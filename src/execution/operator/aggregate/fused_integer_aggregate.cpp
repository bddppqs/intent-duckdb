#include "duckdb/execution/operator/aggregate/fused_integer_aggregate.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/operator/aggregate/distinct_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/run_aggregate.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/compression/dict_global/column_dictionary.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/table/column_data.hpp"
#include "duckdb/storage/table/row_group.hpp"
#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/common/tuning_defaults.hpp"

#include <atomic>
#include <cstring>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Compile-time defaults (duckdb/common/tuning_defaults.hpp)
//===--------------------------------------------------------------------===//
//! The kernel attaches at all
static bool FusedIntegerAggregateEnabled() {
	return kFusedIntegerAggregate;
}

//! The kernel's DISTINCT class attaches (with the kernel)
static bool FusedDistinctClassEnabled() {
	return kFusedDistinctAggregate;
}

//! The kernel accepts the run channel's run-length encoded input
static bool FusedRunChannelEnabled() {
	return kFusedRunChannel;
}

//! The kernel admits a dictionary-coded VARCHAR group key
static bool FusedVarcharKeysEnabled() {
	return kFusedIntegerAggregateVarcharKeys;
}

//! Phase 1 folds a COUNT-only shape's row whose key equals its partition's last row into that row's count; otherwise
//! the grouped class appends every row
static bool FusedLastKeyFoldEnabled() {
	return kFusedLastKeyFold;
}

//! The last-key fold is refused where the key columns' distinct-count statistics show it cannot pay for its wider row
//! (FusedLastKeyFoldPays); otherwise it is taken wherever it fits
static bool FusedLastKeyFoldGuardEnabled() {
	return kFusedLastKeyFoldStatisticsGuard;
}

//! The grouped class takes up to three keys and 16 key bytes; otherwise (and always for the DISTINCT class) two keys and
//! 12 key bytes
static bool FusedSixteenByteKeysEnabled() {
	return kFusedSixteenByteKeys;
}

//! A key may be computed from one column by a deterministic scalar function (FusedIsComputedKey) below or inside its
//! compress function; otherwise a compress takes a column reference alone
static bool FusedComputedKeysEnabled() {
	return kFusedComputedGroupKeys;
}

//! A grouped DISTINCT shape merges its task tables as a tree as the tasks finish (FusedTreeMerge); otherwise the last
//! task merges every task's group table, then every thread's companion, alone
static bool FusedDistinctTreeMergeEnabled() {
	return kFusedDistinctTreeMerge;
}

//! Each row's hash is computed once and stored in the row for the partition build
static bool FusedStoredHashEnabled() {
	return kFusedStoredHash;
}

//! The partition build chains on the stored hash bits instead of open addressing
static bool FusedStoredHashChainEnabled() {
	return kFusedStoredHashChain;
}

//! Sink and SinkRuns reserve a chunk's bytes by one fetch_add, rolled back when the sum crosses the drain threshold;
//! otherwise by a compare-and-swap retry loop
static const bool fused_atomic_reserve = kFusedAggregateAtomicReserve;

//! Each local state reserves from its own allowance, refilled by one fetch_add of max(shortfall, G) (FusedM2Reserve),
//! and keeps the EXPLAIN ANALYZE counters' per-chunk adds in its own words, both flushed to the global state at its drain and its
//! Combine (needs the fetch_add reservation above)
static const bool fused_local_allowance = kFusedAggregateLocalAllowance && fused_atomic_reserve;

//! An ungrouped count(DISTINCT x) over a published VARCHAR column takes the bitmap class (FusedAdmitBitmap); otherwise
//! every operator is decided without it
static bool FusedDistinctBitmapEnabled() {
	return kFusedDistinctBitmap;
}

//! An ungrouped count(DISTINCT x) over an integer x, the operator's one aggregate, takes the set member (no stored hash,
//! phase 2 counts each partition's distinct x in a set of x alone); otherwise the DISTINCT class's (g, x) build and group
//! fold
static bool FusedDistinctSetEnabled() {
	return kFusedDistinctSet;
}

//===--------------------------------------------------------------------===//
// The row hash and the key words
//===--------------------------------------------------------------------===//
// The key region (the first key_bytes <= 16 bytes of the compact row) is read as one or two little-endian words, the
// bytes past the keys masked off (the second word a 32-bit load up to 12 key bytes, a 64-bit one past 12, where the row
// is at least 16 bytes); the words are hashed with DuckDB's Hash(uint64_t) (MurmurHash64) and combined as
// DuckDB's vector hash combines column hashes (CombineHashScalar, vector_hash.cpp). Phase 1 partitions on the top 12
// bits; phase 2 indexes its table with the bits right below them.
struct FusedKeyShape {
	uint64_t mask0;
	uint64_t mask1;
	bool two_words;
	bool wide;
};

static FusedKeyShape FusedGetKeyShape(idx_t key_bytes) {
	FusedKeyShape shape;
	shape.two_words = key_bytes > 8;
	shape.wide = key_bytes > 12;
	shape.mask0 = key_bytes >= 8 ? ~uint64_t(0) : (uint64_t(1) << (8 * key_bytes)) - 1;
	shape.mask1 = shape.two_words ? (key_bytes >= 16 ? ~uint64_t(0) : (uint64_t(1) << (8 * (key_bytes - 8))) - 1) : 0;
	return shape;
}

static inline hash_t FusedCombineHash(hash_t left, hash_t right) {
	left ^= left >> 32;
	left *= 0xd6e8feb86659fd93U;
	return left ^ right;
}

static inline void FusedLoadKey(const_data_ptr_t row, const FusedKeyShape &shape, uint64_t &key0, uint64_t &key1) {
	key0 = Load<uint64_t>(row) & shape.mask0;
	key1 = shape.two_words ? (shape.wide ? Load<uint64_t>(row + 8) : uint64_t(Load<uint32_t>(row + 8))) & shape.mask1
	                       : 0;
}

#if defined(__GNUC__) || defined(__clang__)
// The row hash is inlined into every caller
#define FUSED_ALWAYS_INLINE __attribute__((always_inline))
#else
#define FUSED_ALWAYS_INLINE
#endif

static inline hash_t FusedHashKey(uint64_t key0, uint64_t key1, bool two_words) FUSED_ALWAYS_INLINE;

static inline hash_t FusedHashKey(uint64_t key0, uint64_t key1, bool two_words) {
	auto hash = Hash<uint64_t>(key0);
	if (two_words) {
		hash = FusedCombineHash(hash, Hash<uint64_t>(key1));
	}
	return hash;
}

static inline idx_t FusedPartitionOf(hash_t hash) {
	return hash >> (64 - FusedIntegerAggregate::PARTITION_BITS);
}

static inline int64_t FusedLoadInput(const_data_ptr_t row, const FusedRowColumn &column) {
	switch (column.width) {
	case 2:
		return Load<int16_t>(row + column.offset);
	case 4:
		return Load<int32_t>(row + column.offset);
	case 8:
		return Load<int64_t>(row + column.offset);
	default:
		return Load<int8_t>(row + column.offset);
	}
}

//! the stored bits are the hash's bits 51..20 - FusedSlotOf's slot bits right below the partition bits, then a
//! tag - so (stored << 20) reproduces every bit FusedSlotOf reads for a table of at most 2^32 slots
static constexpr idx_t FUSED_STORED_HASH_SHIFT = 64 - FusedIntegerAggregate::PARTITION_BITS - 32;

static inline uint32_t FusedStoredBits(hash_t hash) {
	return uint32_t(hash >> FUSED_STORED_HASH_SHIFT);
}

#if defined(__GNUC__) || defined(__clang__)
#define FUSED_PREFETCH_WRITE(pointer) __builtin_prefetch(pointer, 1, 3)
// The phase-2 builds stay out of line
#define FUSED_NOINLINE __attribute__((noinline))
#else
#define FUSED_PREFETCH_WRITE(pointer)
#define FUSED_NOINLINE
#endif

//===--------------------------------------------------------------------===//
// The shape gate
//===--------------------------------------------------------------------===//
FusedIntegerAggregate::FusedIntegerAggregate()
    : key_count(0), key_bytes(0), row_width(0), budget_bytes(0), distinct(false), group_bytes(0), fold(false),
      mixed(false), gid_keys(0), hash_stored(false), hash_offset(0), chain(false), run_kind(false),
      run_length_offset(0), last_key_fold(false), bitmap(false), bitmap_words(0) {
}

static bool FusedIsIntegerKeyType(PhysicalType type) {
	switch (type) {
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
		return true;
	default:
		return false;
	}
}

static bool FusedIsInputType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
		return true;
	default:
		return false;
	}
}

//! A computed key of one column, f(BOUND_REF, constant...): f a deterministic scalar function (CONSISTENT) with a
//! statistics callback, exactly one child a BOUND_REF and every other child a constant
static bool FusedIsComputedKey(const Expression &expr) {
	if (!FusedComputedKeysEnabled() || expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &function = expr.Cast<BoundFunctionExpression>();
	if (function.function.GetStability() != FunctionStability::CONSISTENT || !function.function.HasStatisticsCallback()) {
		return false;
	}
	idx_t references = 0;
	for (auto &child : function.children) {
		if (child->GetExpressionType() == ExpressionType::BOUND_REF) {
			references++;
		} else if (child->GetExpressionType() != ExpressionType::VALUE_CONSTANT) {
			return false;
		}
	}
	return references == 1;
}

//! The index the computed key's one BOUND_REF child reads
static idx_t FusedComputedKeyColumn(const Expression &expr) {
	for (auto &child : expr.Cast<BoundFunctionExpression>().children) {
		if (child->GetExpressionType() == ExpressionType::BOUND_REF) {
			return child->Cast<BoundReferenceExpression>().index;
		}
	}
	throw InternalException("FusedComputedKeyColumn: no BOUND_REF child");
}

//! The computed key's statistics, its function's own callback over the column's statistics (a constant child's
//! from its value), run on a copy of the expression (a callback may rewrite the expression it is given): null unless
//! the range is bounded (numeric statistics with a minimum and a maximum)
static unique_ptr<BaseStatistics> FusedComputedKeyStatistics(ClientContext &context, const Expression &expr,
                                                             const BaseStatistics &column_stats) {
	auto copy = expr.Copy();
	auto &function = copy->Cast<BoundFunctionExpression>();
	vector<BaseStatistics> child_stats;
	for (auto &child : function.children) {
		child_stats.push_back(child->GetExpressionType() == ExpressionType::BOUND_REF
		                          ? column_stats.Copy()
		                          : BaseStatistics::FromConstant(child->Cast<BoundConstantExpression>().value));
	}
	FunctionStatisticsInput input(function, function.bind_info.get(), child_stats, &copy);
	auto stats = function.function.GetStatisticsCallback()(context, input);
	if (!stats || stats->GetStatsType() != StatisticsType::NUMERIC_STATS || !NumericStats::HasMinMax(*stats)) {
		return nullptr;
	}
	return stats;
}

//! The compressed-materialization function a key may pass through: compress(BOUND_REF, constant) or
//! compress(computed key, constant)
static bool FusedIsCompressFunction(const Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &function = expr.Cast<BoundFunctionExpression>();
	auto &name = function.function.name;
	if (name != "__internal_compress_integral_utinyint" && name != "__internal_compress_integral_usmallint" &&
	    name != "__internal_compress_integral_uinteger" && name != "__internal_compress_integral_ubigint") {
		return false;
	}
	return function.children.size() == 2 &&
	       (function.children[0]->GetExpressionType() == ExpressionType::BOUND_REF ||
	        FusedIsComputedKey(*function.children[0])) &&
	       function.children[1]->GetExpressionType() == ExpressionType::VALUE_CONSTANT;
}

//! The operators between the aggregate and its input: PROJECTION* / FILTER* over one TABLE_SCAN of a DuckDB table (the
//! walk of CanUsePartitionedAggregate)
struct FusedChain {
	vector<reference<PhysicalOperator>> operators;
	optional_ptr<PhysicalTableScan> scan;
};

static bool FusedWalkChain(PhysicalOperator &child, FusedChain &chain) {
	reference<PhysicalOperator> current(child);
	while (current.get().type == PhysicalOperatorType::PROJECTION ||
	       current.get().type == PhysicalOperatorType::FILTER) {
		if (current.get().children.size() != 1) {
			return false;
		}
		chain.operators.push_back(current);
		current = current.get().children[0];
	}
	if (current.get().type != PhysicalOperatorType::TABLE_SCAN) {
		return false;
	}
	auto &scan = current.get().Cast<PhysicalTableScan>();
	if (scan.function.name != "seq_scan" || !scan.bind_data ||
	    (!scan.function.statistics && !scan.function.statistics_extended)) {
		return false;
	}
	chain.scan = &scan;
	return true;
}

//! Resolves column `index` of the aggregate's input through the chain to a base-table column and reads its statistics
//! through the scan's statistics function (none for transaction-local data): null when the column does not resolve. A
//! key may pass through one compress function (NULL-preserving, its statistics a copy of the source's); an input may not.
//! Below or inside that compress, a key may pass through one computed key of one column (FusedIsComputedKey), and its
//! statistics are the computed key's (FusedComputedKeyStatistics)
static unique_ptr<BaseStatistics> FusedResolveColumnStatistics(ClientContext &context, const FusedChain &chain,
                                                               idx_t index, bool allow_compress) {
	bool compressed = false;
	optional_ptr<const Expression> computed;
	for (auto &operator_ref : chain.operators) {
		auto &op = operator_ref.get();
		if (op.type == PhysicalOperatorType::FILTER) {
			continue;
		}
		auto &projection = op.Cast<PhysicalProjection>();
		if (index >= projection.select_list.size()) {
			return nullptr;
		}
		auto &expr = *projection.select_list[index];
		if (expr.GetExpressionType() == ExpressionType::BOUND_REF) {
			index = expr.Cast<BoundReferenceExpression>().index;
			continue;
		}
		if (allow_compress && !compressed && FusedIsCompressFunction(expr)) {
			auto &source = *expr.Cast<BoundFunctionExpression>().children[0];
			if (source.GetExpressionType() == ExpressionType::BOUND_REF) {
				index = source.Cast<BoundReferenceExpression>().index;
			} else {
				computed = &source;
				index = FusedComputedKeyColumn(source);
			}
			compressed = true;
			continue;
		}
		if (compressed && !computed && FusedIsComputedKey(expr)) {
			computed = &expr;
			index = FusedComputedKeyColumn(expr);
			continue;
		}
		return nullptr;
	}
	auto &scan = *chain.scan;
	idx_t scan_index = index;
	if (!scan.projection_ids.empty()) {
		if (index >= scan.projection_ids.size()) {
			return nullptr;
		}
		scan_index = scan.projection_ids[index];
	}
	if (scan_index >= scan.column_ids.size()) {
		return nullptr;
	}
	auto &column_index = scan.column_ids[scan_index];
	if (column_index.IsRowIdColumn() || column_index.IsVirtualColumn() || column_index.HasChildren() ||
	    !column_index.HasPrimaryIndex()) {
		return nullptr;
	}
	unique_ptr<BaseStatistics> stats;
	if (scan.function.statistics_extended) {
		TableFunctionGetStatisticsInput input(scan.bind_data.get(), column_index);
		stats = scan.function.statistics_extended(context, input);
	} else {
		stats = scan.function.statistics(context, scan.bind_data.get(), column_index.GetPrimaryIndex());
	}
	if (computed && stats) {
		return FusedComputedKeyStatistics(context, *computed, *stats);
	}
	return stats;
}

//! the column resolves (FusedResolveColumnStatistics) and its statistics say it cannot hold NULL
static bool FusedResolvesToNonNullColumn(ClientContext &context, const FusedChain &chain, idx_t index,
                                         bool allow_compress) {
	auto stats = FusedResolveColumnStatistics(context, chain, index, allow_compress);
	return stats && !stats->CanHaveNull();
}

//! Whether the last-key fold can pay for its count. The count widens the compact row from `plain` to `folded` bytes (the
//! stored hash bits beside both), and the fold saves a row only for an input row whose key repeats: at most the input
//! rows less its groups, and a composite key has at least as many groups as the largest distinct-count statistic of its
//! columns. The fold is refused when folding every repeated row would still move no fewer row bytes:
//! (rows - distinct) x folded <= rows x (folded - plain) - for a 16 -> 20 byte row, a key column with more than
//! 0.8 x rows distinct values. The statistics describe the whole table, so only an unfiltered scan is read; a row that
//! does not widen, a filtered input or an unknown statistic keeps the fold
static bool FusedLastKeyFoldPays(ClientContext &context, const FusedChain &chain, const FusedIntegerAggregate &fused,
                                 idx_t offset, idx_t rows) {
	const idx_t hash_bytes = FusedStoredHashEnabled() ? FusedIntegerAggregate::STORED_HASH_BYTES : 0;
	const idx_t plain = MaxValue<idx_t>(8, AlignValue<idx_t>(offset, 4)) + hash_bytes;
	const idx_t folded = MaxValue<idx_t>(8, AlignValue<idx_t>(offset + sizeof(uint32_t), 4)) + hash_bytes;
	if (!FusedLastKeyFoldGuardEnabled() || folded == plain || rows == 0) {
		return true;
	}
	for (auto &op : chain.operators) {
		if (op.get().type == PhysicalOperatorType::FILTER) {
			return true;
		}
	}
	if (chain.scan->table_filters && !chain.scan->table_filters->filters.empty()) {
		return true;
	}
	idx_t distinct = 0;
	for (idx_t key_idx = 0; key_idx < fused.key_count; key_idx++) {
		auto stats = FusedResolveColumnStatistics(context, chain, fused.columns[key_idx].chunk_index, true);
		if (stats) {
			distinct = MaxValue<idx_t>(distinct, stats->GetDistinctCount());
		}
	}
	distinct = MinValue<idx_t>(distinct, rows);
	return (rows - distinct) * folded > rows * (folded - plain);
}

//! the base-table column (its primary index) that column `index` of the aggregate's input resolves to
//! through BOUND_REFs (the walk of FusedResolvesToNonNullColumn, no compress function), or invalid
static optional_idx FusedResolveBaseColumn(const FusedChain &chain, idx_t index) {
	for (auto &operator_ref : chain.operators) {
		auto &op = operator_ref.get();
		if (op.type == PhysicalOperatorType::FILTER) {
			continue;
		}
		auto &projection = op.Cast<PhysicalProjection>();
		if (index >= projection.select_list.size() ||
		    projection.select_list[index]->GetExpressionType() != ExpressionType::BOUND_REF) {
			return optional_idx();
		}
		index = projection.select_list[index]->Cast<BoundReferenceExpression>().index;
	}
	auto &scan = *chain.scan;
	idx_t scan_index = index;
	if (!scan.projection_ids.empty()) {
		if (index >= scan.projection_ids.size()) {
			return optional_idx();
		}
		scan_index = scan.projection_ids[index];
	}
	if (scan_index >= scan.column_ids.size()) {
		return optional_idx();
	}
	auto &column_index = scan.column_ids[scan_index];
	if (column_index.IsRowIdColumn() || column_index.IsVirtualColumn() || column_index.HasChildren() ||
	    !column_index.HasPrimaryIndex()) {
		return optional_idx();
	}
	return optional_idx(column_index.GetPrimaryIndex());
}

//! a VARCHAR key type: a plain VARCHAR (no alias) with the binary collation - neither the type's own collation nor
//! a default collation - with VARCHAR keys enabled (FusedVarcharKeysEnabled)
static bool FusedIsGidKeyType(ClientContext &context, const LogicalType &type) {
	if (!FusedVarcharKeysEnabled() || type.id() != LogicalTypeId::VARCHAR || type.HasAlias()) {
		return false;
	}
	return StringType::GetCollation(type).empty() && Settings::Get<DefaultCollationSetting>(context).empty();
}

//! exactly one distinct aggregate, count(DISTINCT x) with one child and no FILTER or ORDER BY, whose radix
//! table is its own (the only entry of table_map); its index, or invalid
static optional_idx FusedDistinctIndex(const DistinctAggregateCollectionInfo &info, const DistinctAggregateData &data,
                                       const vector<unique_ptr<Expression>> &aggregates) {
	if (info.indices.size() != 1 || info.table_count != 1 || info.table_map.size() != 1) {
		return optional_idx();
	}
	const auto aggr_idx = info.indices[0];
	if (aggr_idx >= aggregates.size()) {
		return optional_idx();
	}
	auto &aggr = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();
	if (aggr.aggr_type != AggregateType::DISTINCT || aggr.function.name != "count" || aggr.children.size() != 1 ||
	    aggr.filter || aggr.order_bys) {
		return optional_idx();
	}
	auto entry = info.table_map.find(aggr_idx);
	if (entry == info.table_map.end() || entry->second >= data.radix_tables.size() ||
	    !data.radix_tables[entry->second]) {
		return optional_idx();
	}
	return optional_idx(aggr_idx);
}

//! the admission checks over the keys and the aggregates of one aggregate operator; with a distinct index, the DISTINCT
//! class: zero to two keys, x (the distinct aggregate's child) after them in the key region, the distinct aggregate
//! beside zero to three non-distinct ones, and the operator's own group estimate at most 2^20
static unique_ptr<FusedIntegerAggregate> FusedAdmitShape(ClientContext &context, PhysicalOperator &child,
                                                         const vector<unique_ptr<Expression>> &groups,
                                                         const vector<unique_ptr<Expression>> &aggregates,
                                                         const vector<LogicalType> &types, optional_idx distinct_index,
                                                         TupleDataValidityType group_validity, idx_t group_estimate, bool run_kind) {
	const bool distinct = distinct_index.IsValid();
	// one to three keys (the DISTINCT class: zero to two), each a BOUND_REF of an integer physical type (INT8,
	// HUGEINT and every non-integer are refused); at most 16 key bytes. The DISTINCT class, and every class without
	// FusedSixteenByteKeysEnabled, keeps two keys and 12 key bytes
	auto fused = make_uniq<FusedIntegerAggregate>();
	const bool key16 = FusedSixteenByteKeysEnabled() && !distinct;
	const idx_t maximum_keys = key16 ? 3 : 2;
	const idx_t maximum_key_bytes = key16 ? FusedIntegerAggregate::MAXIMUM_KEY_BYTES : 12;
	const idx_t key_count = groups.size();
	if (key_count > maximum_keys || (!distinct && key_count < 1)) {
		return nullptr;
	}
	idx_t offset = 0;
	for (auto &group : groups) {
		if (group->GetExpressionType() != ExpressionType::BOUND_REF) {
			return nullptr;
		}
		const auto physical_type = group->return_type.InternalType();
		const bool gid = !FusedIsIntegerKeyType(physical_type) && distinct && FusedIsGidKeyType(context, group->return_type);
		if (!FusedIsIntegerKeyType(physical_type) && !gid) {
			return nullptr;
		}
		FusedRowColumn column;
		column.chunk_index = group->Cast<BoundReferenceExpression>().index;
		column.offset = offset;
		column.width = gid ? sizeof(uint16_t) : GetTypeIdSize(physical_type);
		column.gid = gid;
		offset += column.width;
		fused->columns.push_back(column);
		fused->gid_keys += gid ? 1 : 0;
	}
	fused->group_bytes = offset;
	if (distinct) {
		auto &x = *aggregates[distinct_index.GetIndex()]->Cast<BoundAggregateExpression>().children[0];
		if (x.GetExpressionType() != ExpressionType::BOUND_REF || !FusedIsIntegerKeyType(x.return_type.InternalType())) {
			return nullptr;
		}
		FusedRowColumn column;
		column.chunk_index = x.Cast<BoundReferenceExpression>().index;
		column.offset = offset;
		column.width = GetTypeIdSize(x.return_type.InternalType());
		offset += column.width;
		fused->columns.push_back(column);
	}
	if (offset > maximum_key_bytes) {
		return nullptr;
	}
	if (run_kind && key_count != 1) {
		return nullptr;
	}
	fused->key_count = key_count;
	fused->key_bytes = offset;
	// every key must resolve through the child chain to a base-table column that cannot hold NULL, as the planner's
	// group validity says; so must the DISTINCT argument x (through BOUND_REFs only)
	FusedChain chain;
	if (!FusedWalkChain(child, chain)) {
		return nullptr;
	}
	if (group_validity != TupleDataValidityType::CANNOT_HAVE_NULL_VALUES) {
		return nullptr;
	}
	for (idx_t key_idx = 0; key_idx < key_count; key_idx++) {
		if (fused->columns[key_idx].gid) {
			// a VARCHAR key resolves through BOUND_REFs alone to a base-table column whose statistics say it
			// cannot hold NULL and whose distinct-count estimate is known and at most MAXIMUM_GIDS
			auto stats = FusedResolveColumnStatistics(context, chain, fused->columns[key_idx].chunk_index, false);
			if (!stats || stats->CanHaveNull() || stats->GetDistinctCount() == 0 ||
			    stats->GetDistinctCount() > FusedIntegerAggregate::MAXIMUM_GIDS) {
				return nullptr;
			}
			continue;
		}
		if (!FusedResolvesToNonNullColumn(context, chain, fused->columns[key_idx].chunk_index, true)) {
			return nullptr;
		}
	}
	if (distinct && !FusedResolvesToNonNullColumn(context, chain, fused->columns[key_count].chunk_index, false)) {
		return nullptr;
	}
	// one to four aggregates, none DISTINCT, FILTER or ORDER BY, each named by its input's logical type:
	// count_star; count(x) over SMALLINT / INTEGER / BIGINT; sum(x) over SMALLINT (the int64 state);
	// sum_no_overflow(x) over INTEGER / BIGINT; avg(x) over SMALLINT with no bind data (the int64 state and the
	// double / double finalize); every input a BOUND_REF resolving to a column that cannot hold NULL. In the DISTINCT
	// class the distinct aggregate is one of the four, its result BIGINT
	if (aggregates.empty() || aggregates.size() > FusedIntegerAggregate::MAXIMUM_AGGREGATES) {
		return nullptr;
	}
	// the DISTINCT class attaches only when count(DISTINCT x) is the operator's one aggregate, so every
	// attached shape folds; a non-distinct aggregate beside it takes the general path (without the fold the single pass
	// is slower than the general path). With kFusedMixedDistinct, zero to three non-distinct aggregates of the kinds above may
	// stand beside the distinct one (MAXIMUM_AGGREGATES counts it), none reading x (checked below); they are
	// computed by the companion, so the (g, x) rows still fold
	const bool mixed = distinct && aggregates.size() != 1;
	if (mixed && !kFusedMixedDistinct) {
		return nullptr;
	}
	// a mixed shape with a VARCHAR key takes the general path
	if (mixed && fused->gid_keys > 0) {
		return nullptr;
	}
	// x's chunk column and its base-table column, which no non-distinct aggregate may read (the drain's
	// re-expanded rows carry x = 0, and the regular Sink reads every aggregate's child column)
	const idx_t x_chunk_index = distinct ? fused->columns[key_count].chunk_index : 0;
	const optional_idx x_base = distinct ? FusedResolveBaseColumn(chain, x_chunk_index) : optional_idx();
	if (mixed && !x_base.IsValid()) {
		return nullptr;
	}
	if (types.size() != key_count + aggregates.size()) {
		return nullptr;
	}
	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggr = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();
		auto &result_type = types[key_count + aggr_idx];
		if (run_kind && aggr.function.name != "count_star") {
			return nullptr;
		}
		FusedAggregateOutput output;
		output.kind = FusedAggregateKind::COUNT;
		output.sum_index = 0;
		if (distinct && aggr_idx == distinct_index.GetIndex()) {
			if (result_type.id() != LogicalTypeId::BIGINT) {
				return nullptr;
			}
			output.kind = FusedAggregateKind::DISTINCT_COUNT;
			fused->outputs.push_back(output);
			continue;
		}
		if (aggr.aggr_type != AggregateType::NON_DISTINCT || aggr.filter || aggr.order_bys) {
			return nullptr;
		}
		auto &name = aggr.function.name;
		if (name == "count_star") {
			if (!aggr.children.empty() || result_type.id() != LogicalTypeId::BIGINT) {
				return nullptr;
			}
			fused->outputs.push_back(output);
			continue;
		}
		if (aggr.children.size() != 1 || aggr.children[0]->GetExpressionType() != ExpressionType::BOUND_REF) {
			return nullptr;
		}
		auto &input = *aggr.children[0];
		auto &input_type = input.return_type;
		if (name == "count" && FusedIsInputType(input_type) && result_type.id() == LogicalTypeId::BIGINT) {
			output.kind = FusedAggregateKind::COUNT;
		} else if (name == "sum" && input_type.id() == LogicalTypeId::SMALLINT &&
		           result_type.id() == LogicalTypeId::HUGEINT) {
			output.kind = FusedAggregateKind::SUM;
		} else if (name == "sum_no_overflow" &&
		           (input_type.id() == LogicalTypeId::INTEGER || input_type.id() == LogicalTypeId::BIGINT) &&
		           result_type.id() == LogicalTypeId::HUGEINT) {
			output.kind = FusedAggregateKind::SUM;
		} else if (name == "avg" && input_type.id() == LogicalTypeId::SMALLINT && !aggr.bind_info &&
		           result_type.id() == LogicalTypeId::DOUBLE) {
			output.kind = FusedAggregateKind::AVG;
		} else {
			return nullptr;
		}
		const auto chunk_index = input.Cast<BoundReferenceExpression>().index;
		if (!FusedResolvesToNonNullColumn(context, chain, chunk_index, false)) {
			return nullptr;
		}
		if (mixed) {
			// an input that is x (the same chunk column, or one resolving to x's base-table column) is
			// refused: count(x), sum(x) or avg(x) beside count(DISTINCT x) takes the general path
			const auto input_base = FusedResolveBaseColumn(chain, chunk_index);
			if (chunk_index == x_chunk_index || !input_base.IsValid() ||
			    input_base.GetIndex() == x_base.GetIndex()) {
				return nullptr;
			}
			// the companion input list: each input chunk column once, never in the compact row (the column branch
			// below is not taken), so the (g, x) row and its phase-2 entry carry no input
			idx_t input_idx = 0;
			while (input_idx < fused->companion_inputs.size() &&
			       fused->companion_inputs[input_idx].chunk_index != chunk_index) {
				input_idx++;
			}
			if (input_idx == fused->companion_inputs.size()) {
				FusedRowColumn column;
				column.chunk_index = chunk_index;
				column.offset = 0;
				column.width = GetTypeIdSize(input_type.InternalType());
				fused->companion_inputs.push_back(column);
			}
			if (output.kind != FusedAggregateKind::COUNT) {
				idx_t sum_idx = 0;
				while (sum_idx < fused->companion_sums.size() && fused->companion_sums[sum_idx] != input_idx) {
					sum_idx++;
				}
				if (sum_idx == fused->companion_sums.size()) {
					fused->companion_sums.push_back(input_idx);
				}
				output.sum_index = sum_idx;
			}
			fused->outputs.push_back(output);
			continue;
		}
		// the compact row holds each distinct input column once, after the keys (the DISTINCT class: x included)
		idx_t column_idx = key_count;
		while (column_idx < fused->columns.size() && fused->columns[column_idx].chunk_index != chunk_index) {
			column_idx++;
		}
		if (column_idx == fused->columns.size()) {
			FusedRowColumn column;
			column.chunk_index = chunk_index;
			column.offset = offset;
			column.width = GetTypeIdSize(input_type.InternalType());
			offset += column.width;
			fused->columns.push_back(column);
		}
		if (output.kind != FusedAggregateKind::COUNT) {
			idx_t sum_idx = 0;
			while (sum_idx < fused->sum_columns.size() && fused->sum_columns[sum_idx] != column_idx) {
				sum_idx++;
			}
			if (sum_idx == fused->sum_columns.size()) {
				fused->sum_columns.push_back(column_idx);
			}
			output.sum_index = sum_idx;
		}
		fused->outputs.push_back(output);
	}
	if (run_kind) {
		fused->run_kind = true;
		fused->run_length_offset = offset;
		offset += sizeof(uint32_t);
	}
	// the last-key fold: a COUNT-only shape of the grouped class (no sum state, not DISTINCT, not the run kind) carries
	// a uint32_t count at run_length_offset, so phase 1 can fold a row into its partition's last row; taken only while
	// the row keeps room for the stored hash bits, so the count never refuses a shape nor turns another mechanism off;
	// and only where the statistics leave enough repeated rows to pay for the count's bytes (FusedLastKeyFoldPays)
	if (!run_kind && !distinct && fused->sum_columns.empty() && FusedLastKeyFoldEnabled() &&
	    MaxValue<idx_t>(8, AlignValue<idx_t>(offset + sizeof(uint32_t), 4)) + FusedIntegerAggregate::STORED_HASH_BYTES <=
	        FusedIntegerAggregate::MAXIMUM_ROW_BYTES &&
	    FusedLastKeyFoldPays(context, chain, *fused, offset, child.estimated_cardinality)) {
		fused->last_key_fold = true;
		fused->run_length_offset = offset;
		offset += sizeof(uint32_t);
	}
	// the compact row: at most 32 bytes, padded to a multiple of 4 and at least 8 (the key words are read in place)
	fused->row_width = MaxValue<idx_t>(8, AlignValue<idx_t>(offset, 4));
	if (fused->row_width > FusedIntegerAggregate::MAXIMUM_ROW_BYTES) {
		return nullptr;
	}
	// the chain is PROJECTION* / FILTER* over one TABLE_SCAN (walked above); every key passed through BOUND_REFs
	// or one compress function, every input through BOUND_REFs (checked by the resolution above)
	if (!chain.scan) {
		return nullptr;
	}
	// at least 2^20 estimated input rows, and estimate x 32 B + threads x 24.25 MiB <= B = GetMaxMemory() / 4 (the run
	// kind charges each thread twice: the executor's fused local state and the run partial's); the DISTINCT class's
	// group estimate at most 2^20
	const idx_t estimate = child.estimated_cardinality;
	if (estimate < FusedIntegerAggregate::MINIMUM_ESTIMATED_ROWS) {
		return nullptr;
	}
	const auto threads = NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
	const idx_t budget = BufferManager::GetBufferManager(context).GetMaxMemory() / 4;
	const idx_t thread_charge = threads * FusedIntegerAggregate::PER_THREAD_CHARGE_BYTES * (run_kind ? 2 : 1);
	if (thread_charge > budget ||
	    estimate > (budget - thread_charge) / FusedIntegerAggregate::ESTIMATE_ROW_BYTES) {
		return nullptr;
	}
	if (distinct && group_estimate > FusedIntegerAggregate::MAXIMUM_ESTIMATED_GROUPS) {
		return nullptr;
	}
	fused->budget_bytes = budget;
	fused->input_types = child.GetTypes();
	fused->output_types = types;
	fused->distinct = distinct;
	fused->mixed = mixed;
	// every attached DISTINCT shape folds; a mixed shape's inputs never enter the compact
	// row (the companion reads them from the chunk), so a skipped row loses nothing
	fused->fold = distinct && kFusedDistinctFold;
	return fused;
}

//! arms the members - the stored hash in both classes, the chained partition build in the non-DISTINCT class - on an
//! admitted shape: they add no admission and refuse nothing; a row that would pass MAXIMUM_ROW_BYTES with the stored
//! bits keeps the plain row and recomputes the hash (it stores no hash). The DISTINCT class takes the stored hash
//! alone: its (g, x) build reads the stored bits, its group table still hashes g
static void FusedArmMembers(FusedIntegerAggregate &fused) {
	// the run kind keeps its own row {key, run length} and phase 2 - SinkRuns stores no hash bits and its count folds
	// run lengths - so the members stay idle on it
	if (fused.run_kind) {
		return;
	}
	if (FusedStoredHashEnabled() &&
	    fused.row_width + FusedIntegerAggregate::STORED_HASH_BYTES <= FusedIntegerAggregate::MAXIMUM_ROW_BYTES) {
		fused.hash_offset = fused.row_width;
		fused.row_width += FusedIntegerAggregate::STORED_HASH_BYTES;
		fused.hash_stored = true;
		fused.chain = !fused.distinct && FusedStoredHashChainEnabled();
	}
}

void FusedIntegerAggregate::TryAttach(ClientContext &context, PhysicalHashAggregate &op,
                                      TupleDataValidityType group_validity) {
	if (!FusedIntegerAggregateEnabled()) {
		return;
	}
	auto &data = op.grouped_aggregate_data;
	const bool run_kind = op.run_aggregate && op.run_aggregate->IsGrouped() && FusedRunChannelEnabled();
	if (op.children.size() != 1 || op.grouping_sets.size() != 1 || op.groupings.size() != 1 ||
	    !data.grouping_functions.empty() || (op.run_aggregate && !run_kind)) {
		return;
	}
	if (op.grouping_sets[0].size() != data.groups.size() || !op.groupings[0].table_data.null_groups.empty()) {
		return;
	}
	optional_idx distinct_index;
	if (op.distinct_collection_info || op.groupings[0].distinct_data) {
		if (run_kind || !FusedDistinctClassEnabled() || !op.distinct_collection_info || !op.groupings[0].distinct_data ||
		    data.groups.empty()) {
			return;
		}
		distinct_index =
		    FusedDistinctIndex(*op.distinct_collection_info, *op.groupings[0].distinct_data, data.aggregates);
		if (!distinct_index.IsValid()) {
			return;
		}
	}
	auto code_keys = dict_global::FindCodeKeys(&op);
	if (distinct_index.IsValid() && code_keys && !dict_global::FusedDistinctEnabled()) {
		return;
	}
	// the group cap over a code key: the grouped DISTINCT class ends phase 2 in one merge of the tasks' group tables and
	// one task's emission of every group, both serial and linear in the groups, so the class caps the groups at
	// MAXIMUM_ESTIMATED_GROUPS;
	// the operator's estimate is not a bound, but a code key's groups are at most its dictionary's codes, so the class
	// is refused when the code keys' code counts multiply past the cap (the operator keeps its code-typed keys and takes
	// the general aggregate)
	if (distinct_index.IsValid() && code_keys) {
		idx_t domain = 1;
		for (auto &key : code_keys->keys) {
			const idx_t codes = MaxValue<idx_t>(key.dict ? key.dict->count : MAXIMUM_ESTIMATED_GROUPS + 1, 1);
			if (codes > MAXIMUM_ESTIMATED_GROUPS / domain) {
				return;
			}
			domain *= codes;
		}
	}
	auto fused = FusedAdmitShape(context, op.children[0].get(), data.groups, data.aggregates, op.types,
	                             distinct_index, group_validity, op.estimated_cardinality, run_kind);
	if (fused) {
		// over code keys the kernel's Sink receives the converted chunk (each code key an INTEGER column), so
		// the drain rebuilds its chunks with those types, never with the child's VARCHAR ones
		if (code_keys) {
			fused->input_types = code_keys->input_types;
		}
		FusedArmMembers(*fused);
		op.fused = std::move(fused);
	}
}

//! The coverage bound: the reasons an execution of the bitmap class is refused (the marker's refusal=)
enum class FusedBitmapRefusal : uint8_t { NONE = 0, LOCAL_STORAGE, STALE_DICT, COVERAGE, UPDATES, MEMORY };

static const char *FusedBitmapRefusalName(uint8_t refusal) {
	switch (static_cast<FusedBitmapRefusal>(refusal)) {
	case FusedBitmapRefusal::NONE:
		return "none";
	case FusedBitmapRefusal::LOCAL_STORAGE:
		return "local_storage";
	case FusedBitmapRefusal::STALE_DICT:
		return "stale_dict";
	case FusedBitmapRefusal::COVERAGE:
		return "coverage";
	case FusedBitmapRefusal::UPDATES:
		return "updates";
	case FusedBitmapRefusal::MEMORY:
		return "memory";
	}
	return "unknown";
}

//! The coverage bound of the lookup path (zero uncovered rows): why the rows a scan of `table` emits for the column of `dict`
//! may hold a string `dict` does not, or NONE - (i) the transaction holds local storage for the table; at an execution,
//! the column's publication is no longer `dict` (a checkpoint moved the epoch, or a republish); (ii) committed rows lie
//! outside the segments `dict` was built from (rows_covered below GetTotalRows(): an append not yet checkpointed, a
//! segment the build skipped), or the column carries an update. Refused, the generic path answers, so the overflow set
//! stays empty on every execution the bound admits
static FusedBitmapRefusal FusedBitmapCoverage(ClientContext &context, DataTable &table,
                                              const dict_global::ColumnDictionary &dict, bool execution) {
	if (LocalStorage::Get(context, table.db).Find(table)) {
		return FusedBitmapRefusal::LOCAL_STORAGE;
	}
	if (execution && dict.persisted) {
		// a column read through stored translations that are no longer the column's: the execution reads strings
		if (dict_global::PublishPersisted(table, dict.storage_index).get() != &dict) {
			return FusedBitmapRefusal::STALE_DICT;
		}
	} else if (execution && dict_global::Published(table, dict.storage_index).get() != &dict) {
		return FusedBitmapRefusal::STALE_DICT;
	}
	if (dict.rows_covered != table.GetTotalRows()) {
		return FusedBitmapRefusal::COVERAGE;
	}
	auto row_groups = table.GetRowGroupCollection()->GetRowGroups();
	for (auto node = row_groups->GetRootSegment(); node; node = row_groups->GetNextSegment(*node)) {
		// a column not loaded carries no update (an update loads its column first): never loaded here for the check
		auto &row_group = node->GetNode();
		if (row_group.IsColumnLoaded(dict.storage_index) &&
		    row_group.GetRawColumnData(dict.storage_index).HasUpdates()) {
			return FusedBitmapRefusal::UPDATES;
		}
	}
	return FusedBitmapRefusal::NONE;
}

//! The bitmap class's gate, tried on the ungrouped operator once its distinct aggregate is found (FusedDistinctIndex):
//! the distinct aggregate is the operator's one aggregate, its result BIGINT; x is a BOUND_REF of a plain VARCHAR (no
//! alias) with the binary collation - neither the type's own nor a default collation; x resolves to an admitted column
//! the scan publishes for this plan and whose dictionary is published (dict_global::PlanPublishedColumn), so the code
//! domain [0, codes) is known; the memory bound: one bitmap per thread, threads x 8 x ceil(codes / 64) bytes within the
//! kernel's budget GetMaxMemory() / 4; the work bound: phase 2 reads threads x ceil(codes / 64) words, at most the
//! child's estimated input rows (the rows the generic path would sink); the coverage bound, FusedBitmapCoverage, the
//! plan-time fast path of the per-execution re-check. Null when any condition fails: the operator is decided as before
static unique_ptr<FusedIntegerAggregate> FusedAdmitBitmap(ClientContext &context, PhysicalUngroupedAggregate &op,
                                                          idx_t distinct_index) {
	if (op.aggregates.size() != 1 || op.types.size() != 1 || op.types[0].id() != LogicalTypeId::BIGINT) {
		return nullptr;
	}
	auto &x = *op.aggregates[distinct_index]->Cast<BoundAggregateExpression>().children[0];
	if (x.GetExpressionType() != ExpressionType::BOUND_REF || x.return_type.id() != LogicalTypeId::VARCHAR ||
	    x.return_type.HasAlias() || !StringType::GetCollation(x.return_type).empty() ||
	    !Settings::Get<DefaultCollationSetting>(context).empty()) {
		return nullptr;
	}
	auto &child = op.children[0].get();
	const auto chunk_index = x.Cast<BoundReferenceExpression>().index;
	optional_ptr<DataTable> table;
	auto dict = dict_global::PlanPublishedColumn(child, chunk_index, x.return_type, table);
	if (!dict || dict->count == 0) {
		return nullptr;
	}
	if (FusedBitmapCoverage(context, *table, *dict, false) != FusedBitmapRefusal::NONE) {
		return nullptr;
	}
	const idx_t words = (dict->count + 63) / 64;
	const auto threads = NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
	const idx_t budget = BufferManager::GetBufferManager(context).GetMaxMemory() / 4;
	if (threads * words * sizeof(uint64_t) > budget || threads * words > child.estimated_cardinality) {
		return nullptr;
	}
	auto fused = make_uniq<FusedIntegerAggregate>();
	FusedRowColumn column;
	column.chunk_index = chunk_index;
	column.offset = 0;
	column.width = sizeof(uint32_t);
	fused->columns.push_back(column);
	FusedAggregateOutput output;
	output.kind = FusedAggregateKind::DISTINCT_COUNT;
	output.sum_index = 0;
	fused->outputs.push_back(output);
	fused->budget_bytes = budget;
	fused->input_types = child.GetTypes();
	fused->output_types = op.types;
	fused->bitmap = true;
	fused->bitmap_dict = std::move(dict);
	fused->bitmap_words = words;
	fused->bitmap_table = table.get();
	// the scan reads the column's codes only in the executions the sink admits (GetGlobalSinkState decides before the
	// scan starts; a refused execution reads strings for the generic path); a prepared plan re-binds once those
	// translations go stale
	fused->bitmap_codes_only = make_shared_ptr<atomic<bool>>(false);
	dict_global::MarkCodesOnly(child, chunk_index, fused->bitmap_dict, fused->bitmap_codes_only);
	if (fused->bitmap_dict->persisted) {
		dict_global::NoteStoredTranslationPlan(context);
	}
	return fused;
}

void FusedIntegerAggregate::TryAttachUngrouped(ClientContext &context, PhysicalUngroupedAggregate &op) {
	if (!FusedIntegerAggregateEnabled() || !FusedDistinctClassEnabled()) {
		return;
	}
	// the ungrouped operator with one distinct aggregate, and the run channel not attached (it refuses every
	// distinct shape)
	if (op.children.size() != 1 || !op.distinct_collection_info || !op.distinct_data || op.run_aggregate) {
		return;
	}
	auto distinct_index = FusedDistinctIndex(*op.distinct_collection_info, *op.distinct_data, op.aggregates);
	if (!distinct_index.IsValid()) {
		return;
	}
	// a VARCHAR x the global dictionary publishes takes the bitmap class when its gate holds; refused, or with the class
	// off, FusedAdmitShape decides the operator as before (it refuses every VARCHAR x)
	if (FusedDistinctBitmapEnabled()) {
		auto bitmap = FusedAdmitBitmap(context, op, distinct_index.GetIndex());
		if (bitmap) {
			op.fused = std::move(bitmap);
			return;
		}
	}
	const vector<unique_ptr<Expression>> no_groups;
	auto fused = FusedAdmitShape(context, op.children[0].get(), no_groups, op.aggregates, op.types, distinct_index,
	                             TupleDataValidityType::CANNOT_HAVE_NULL_VALUES, op.estimated_cardinality, false);
	if (fused) {
		if (FusedDistinctSetEnabled() && fused->distinct && !fused->mixed && fused->key_count == 0 &&
		    fused->gid_keys == 0 && fused->key_bytes <= sizeof(uint64_t)) {
			// the set member: the row stays x alone (one key word, read in place), so no stored hash is kept
			fused->distinct_set = true;
		} else {
			FusedArmMembers(*fused);
		}
		op.fused = std::move(fused);
	}
}

//===--------------------------------------------------------------------===//
// States
//===--------------------------------------------------------------------===//
// A fused local state's own words, kept past the STANDARD_VECTOR_SIZE row hashes of its `hashes` buffer (which no
// chunk indexes past STANDARD_VECTOR_SIZE), so neither the local nor the global state needs a new field. The
// allowance is bytes this state holds on reserved_bytes by an earlier grant and has not yet reserved for a chunk; the
// counter words are the EXPLAIN ANALYZE counters' per-chunk adds not yet added to the global state. Only the state's own thread
// touches them; FusedM2Flush returns the allowance and adds the counters (at the state's drain and its Combine).
static constexpr idx_t FUSED_M2_ALLOWANCE = 0;
static constexpr idx_t FUSED_M2_INPUT_ROWS = 1;
static constexpr idx_t FUSED_M2_FOLDED_ROWS = 2;
static constexpr idx_t FUSED_M2_COMPANION_ROWS = 3;
static constexpr idx_t FUSED_M2_HASH_STORED_ROWS = 4;
static constexpr idx_t FUSED_M2_RUN_ROWS = 5;
static constexpr idx_t FUSED_M2_RUN_LENGTH_SUM = 6;
static constexpr idx_t FUSED_M2_WORDS = 7;
// the grant G = min(1 MiB, drain threshold / 4096). A state's allowance stays below G (a refill of G is taken
// only while the allowance is short of the chunk's bytes), so the unused part of reserved_bytes stays below
// (fused local states) x G, and the reservations kept never pass the drain threshold
static constexpr idx_t FUSED_M2_GRANT_MAXIMUM = idx_t(1) << 20;
static constexpr idx_t FUSED_M2_GRANT_DIVISOR = 4096;

static inline hash_t *FusedM2Words(FusedAggregateLocalState &lstate) {
	return lstate.hashes.get() + STANDARD_VECTOR_SIZE;
}

FusedAggregateGlobalState::FusedAggregateGlobalState(BufferManager &buffer_manager_p)
    : buffer_manager(buffer_manager_p), abandoned(false), drain_threshold(0), input_rows(0), folded_rows(0),
      companion_rows(0), crossing(uint8_t(FusedCrossing::NONE)), drained_rows(0), combines_taken(0), reserved_bytes(0),
      slabs_allocated(0), partitions_nonempty(0), hash_stored_rows(0), chain_rows(0), chain_new(0), table_probes(0),
      partitions_built(0), run_rows(0), run_length_sum(0), finalized(false),
      gid_arena(buffer_manager_p.GetBufferAllocator()), gid_count(0), gid_maps(0), gid_rows_flat(0) {
}

FusedAggregateLocalState::FusedAggregateLocalState(ExecutionContext &context, const FusedIntegerAggregate &fused)
    : FusedAggregateLocalState(fused) {
}

FusedAggregateLocalState::FusedAggregateLocalState(const FusedIntegerAggregate &fused)
    : drained(false), resinking(false), resinking_distinct_only(false), resinking_regular_only(false) {
	if (fused.bitmap) {
		// the bitmap class keeps none of the row path's buffers (its bitmap is allocated at the first chunk)
		return;
	}
	const auto partitions = FusedIntegerAggregate::PARTITION_COUNT;
	// allocated here, never inside an append
	lines = make_unsafe_uniq_array_uninitialized<data_t>(partitions * FusedIntegerAggregate::LINE_BYTES);
	fill = make_unsafe_uniq_array<uint8_t>(partitions);
	cursor = make_unsafe_uniq_array<data_ptr_t>(partitions);
	room = make_unsafe_uniq_array<uint16_t>(partitions);
	tail = make_unsafe_uniq_array<data_ptr_t>(partitions);
	head = make_unsafe_uniq_array<data_ptr_t>(partitions);
	for (idx_t p = 0; p < partitions; p++) {
		fill[p] = 0;
		cursor[p] = nullptr;
		room[p] = 0;
		tail[p] = nullptr;
		head[p] = nullptr;
	}
	pool.reserve(FusedIntegerAggregate::POOL_MINIMUM_FREE_CHUNKS + FusedIntegerAggregate::CHUNKS_PER_SLAB);
	row_buffer = make_unsafe_uniq_array<data_t>(STANDARD_VECTOR_SIZE * fused.row_width);
	memset(row_buffer.get(), 0, STANDARD_VECTOR_SIZE * fused.row_width);
	// the state's own words live past the chunk's hashes (FusedM2Words), zero until its first chunk
	hashes = make_unsafe_uniq_array_uninitialized<hash_t>(STANDARD_VECTOR_SIZE + FUSED_M2_WORDS);
	memset(hashes.get() + STANDARD_VECTOR_SIZE, 0, FUSED_M2_WORDS * sizeof(hash_t));
	formats.resize(fused.columns.size());
	if (fused.mixed) {
		companion_formats.resize(fused.companion_inputs.size());
		group_rows = make_unsafe_uniq_array<data_t>(STANDARD_VECTOR_SIZE * 16);
		memset(group_rows.get(), 0, STANDARD_VECTOR_SIZE * 16);
		companion_entries = make_unsafe_uniq_array_uninitialized<uint64_t *>(STANDARD_VECTOR_SIZE);
	}
	gid_caches.resize(fused.columns.size());
	for (idx_t column_idx = 0; column_idx < fused.columns.size(); column_idx++) {
		if (fused.columns[column_idx].gid) {
			gid_caches[column_idx].rows = make_unsafe_uniq_array<uint16_t>(STANDARD_VECTOR_SIZE);
		}
	}
}

unique_ptr<FusedAggregateGlobalState> FusedIntegerAggregate::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<FusedAggregateGlobalState>(BufferManager::GetBufferManager(context));
	state->drain_threshold = budget_bytes;
	if (bitmap) {
		// a prepared plan runs again without re-planning, so every execution re-checks the coverage bound - local
		// storage, the publication's identity, the coverage and updates, and the memory bound at the executing threads
		// and memory limit - before any row is sunk; refused, the execution is abandoned at its start and takes the
		// generic path (Sink, Combine and Finalize return false under `abandoned`, and FusedSourceState hands the
		// source to the hash aggregate)
		auto refusal = FusedBitmapCoverage(context, *bitmap_table, *bitmap_dict, true);
		if (refusal == FusedBitmapRefusal::NONE) {
			const auto threads = NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
			const idx_t budget = BufferManager::GetBufferManager(context).GetMaxMemory() / 4;
			if (threads * bitmap_words * sizeof(uint64_t) > budget) {
				refusal = FusedBitmapRefusal::MEMORY;
			}
		}
		state->bitmap_refusal = static_cast<uint8_t>(refusal);
		state->bitmap_engaged = refusal == FusedBitmapRefusal::NONE;
		if (bitmap_codes_only) {
			// the scan below reads this decision at its own initialisation, after the sink's global state exists
			bitmap_codes_only->store(state->bitmap_engaged, std::memory_order_release);
		}
		if (!state->bitmap_engaged) {
			state->abandoned = true;
		}
	}
	return state;
}

unique_ptr<FusedAggregateLocalState> FusedIntegerAggregate::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<FusedAggregateLocalState>(context, *this);
}

//! A chunk's EXPLAIN ANALYZE counter add, into the state's own word (with the local allowance) or the global counter
static inline void FusedM2Count(atomic<idx_t> &counter, FusedAggregateLocalState &lstate, idx_t word, idx_t rows) {
	if (fused_local_allowance) {
		FusedM2Words(lstate)[word] += rows;
	} else {
		counter += rows;
	}
}

//! Returns the state's unused allowance to reserved_bytes and adds its counter words to the global counters, leaving
//! them all zero
static void FusedM2Flush(FusedAggregateGlobalState &gstate, FusedAggregateLocalState &lstate) {
	auto words = FusedM2Words(lstate);
	if (words[FUSED_M2_ALLOWANCE] != 0) {
		gstate.reserved_bytes -= words[FUSED_M2_ALLOWANCE];
	}
	atomic<idx_t> *const counters[FUSED_M2_WORDS] = {nullptr,
	                                                 &gstate.input_rows,
	                                                 &gstate.folded_rows,
	                                                 &gstate.companion_rows,
	                                                 &gstate.hash_stored_rows,
	                                                 &gstate.run_rows,
	                                                 &gstate.run_length_sum};
	for (idx_t word = FUSED_M2_INPUT_ROWS; word < FUSED_M2_WORDS; word++) {
		if (words[word] != 0) {
			*counters[word] += words[word];
		}
	}
	memset(words, 0, FUSED_M2_WORDS * sizeof(hash_t));
}

//! Reserves a chunk's `need` bytes from the state's allowance; when it is short, one fetch_add of
//! grant = max(shortfall, G) refills it. A grant whose sum passes the drain threshold keeps only the shortfall when
//! previous + shortfall still fits (the per-chunk fetch_add reservation's test, on the same previous value) and returns
//! the rest; else it is returned whole and the chunk is refused (a crossing). `current` is set on a refill to the
//! counter as the per-chunk reservation would have read it (previous less this state's allowance), for the refused
//! reservation's reason. True: reserved
static bool FusedM2Reserve(FusedAggregateGlobalState &gstate, FusedAggregateLocalState &lstate, idx_t need,
                           idx_t &current) {
	auto &allowance = FusedM2Words(lstate)[FUSED_M2_ALLOWANCE];
	if (need <= allowance) {
		allowance -= need;
		return true;
	}
	const idx_t shortfall = need - allowance;
	const idx_t grant = MaxValue<idx_t>(
	    shortfall, MinValue<idx_t>(FUSED_M2_GRANT_MAXIMUM, gstate.drain_threshold / FUSED_M2_GRANT_DIVISOR));
	const idx_t previous = gstate.reserved_bytes.fetch_add(grant);
	current = previous - allowance;
	if (previous + grant <= gstate.drain_threshold) {
		allowance = grant - shortfall;
		return true;
	}
	if (previous + shortfall <= gstate.drain_threshold) {
		gstate.reserved_bytes.fetch_sub(grant - shortfall);
		allowance = 0;
		return true;
	}
	gstate.reserved_bytes.fetch_sub(grant);
	return false;
}

//===--------------------------------------------------------------------===//
// Phase 1: partition chunks, the pool, the append
//===--------------------------------------------------------------------===//
// A partition chunk is CHUNK_BYTES: a {next chunk, rows} header, then packed compact rows. A partition takes its first
// chunk when its first row arrives, and its next chunk as soon as a flush leaves no room for another line, so a flush
// (and Combine's flush of the partial lines) always fits and each appended row takes at most one chunk from the pool.
static inline data_ptr_t FusedChunkNext(const_data_ptr_t chunk) {
	return Load<data_ptr_t>(chunk);
}

static inline idx_t FusedChunkRows(const_data_ptr_t chunk) {
	return Load<uint64_t>(chunk + sizeof(data_ptr_t));
}

static inline data_ptr_t FusedTakeChunk(FusedAggregateLocalState &lstate) {
	D_ASSERT(!lstate.pool.empty());
	auto chunk = lstate.pool.back();
	lstate.pool.pop_back();
	Store<data_ptr_t>(nullptr, chunk);
	Store<uint64_t>(0, chunk + sizeof(data_ptr_t));
	return chunk;
}

//! Tops the pool up to POOL_MINIMUM_FREE_CHUNKS whole slabs at a time; the only allocation of phase 1, before any row
//! of the input chunk is appended. Slabs come from the buffer manager, so memory_limit accounting sees them.
static void FusedTopUpPool(FusedAggregateGlobalState &gstate, FusedAggregateLocalState &lstate) {
	// 63 chunks plus 64 bytes of alignment slack: one 256 KiB buffer-manager allocation per slab
	static constexpr idx_t SLAB_BYTES =
	    FusedIntegerAggregate::CHUNKS_PER_SLAB * FusedIntegerAggregate::CHUNK_BYTES + 64;
	while (lstate.pool.size() < FusedIntegerAggregate::POOL_MINIMUM_FREE_CHUNKS) {
		++gstate.slabs_allocated;
		auto handle = gstate.buffer_manager.Allocate(MemoryTag::HASH_TABLE, SLAB_BYTES, true);
		auto base = AlignPointer<64>(handle.Ptr());
		lstate.slabs.push_back(std::move(handle));
		for (idx_t chunk_idx = 0; chunk_idx < FusedIntegerAggregate::CHUNKS_PER_SLAB; chunk_idx++) {
			lstate.pool.push_back(base + chunk_idx * FusedIntegerAggregate::CHUNK_BYTES);
		}
	}
}

// The compact row copies each key and input as its own bytes, by width: a signed column (INT16, INT32, INT64) is read
// and written through the unsigned type of its width, which the typed accessors' type check refuses in a release build
// (StorageTypeCompatible), so the gather, the drain's scatter and phase 2's key emit use the unchecked accessors
template <class T>
static void FusedGatherColumn(const UnifiedVectorFormat &format, idx_t count, data_ptr_t rows, idx_t row_width,
                              idx_t offset) {
	auto source = UnifiedVectorFormat::GetDataUnsafe<T>(format);
	auto &sel = *format.sel;
	auto target = rows + offset;
	for (idx_t i = 0; i < count; i++) {
		Store<T>(source[sel.get_index(i)], target + i * row_width);
	}
}

static bool FusedHasNull(const UnifiedVectorFormat &format, idx_t count) {
	if (format.validity.AllValid()) {
		return false;
	}
	for (idx_t i = 0; i < count; i++) {
		if (!format.validity.RowIsValid(format.sel->get_index(i))) {
			return true;
		}
	}
	return false;
}

template <idx_t W>
static void FusedAppendRows(FusedAggregateLocalState &lstate, const_data_ptr_t rows, const hash_t *hashes,
                            idx_t count) {
	static constexpr idx_t ROWS_PER_LINE = FusedIntegerAggregate::LINE_BYTES / W;
	static constexpr idx_t LINE_ROW_BYTES = ROWS_PER_LINE * W;
	static constexpr idx_t LINES_PER_CHUNK =
	    (FusedIntegerAggregate::CHUNK_BYTES - FusedIntegerAggregate::CHUNK_HEADER_BYTES) / LINE_ROW_BYTES;
	auto lines = lstate.lines.get();
	auto fill = lstate.fill.get();
	auto cursor = lstate.cursor.get();
	auto room = lstate.room.get();
	for (idx_t i = 0; i < count; i++) {
		const auto p = FusedPartitionOf(hashes[i]);
		auto f = fill[p];
		const auto line = lines + p * FusedIntegerAggregate::LINE_BYTES;
		if (f == 0 && !cursor[p]) {
			// the partition's first row: its first chunk
			auto chunk = FusedTakeChunk(lstate);
			lstate.head[p] = chunk;
			lstate.tail[p] = chunk;
			cursor[p] = chunk + FusedIntegerAggregate::CHUNK_HEADER_BYTES;
			room[p] = LINES_PER_CHUNK;
		}
		memcpy(line + f * W, rows + i * W, W);
		if (++f == ROWS_PER_LINE) {
			memcpy(cursor[p], line, LINE_ROW_BYTES);
			cursor[p] += LINE_ROW_BYTES;
			if (--room[p] == 0) {
				// the chunk is full: seal it and take the next one now, so every later flush fits
				auto full = lstate.tail[p];
				Store<uint64_t>(LINES_PER_CHUNK * ROWS_PER_LINE, full + sizeof(data_ptr_t));
				auto chunk = FusedTakeChunk(lstate);
				Store<data_ptr_t>(chunk, full);
				lstate.tail[p] = chunk;
				cursor[p] = chunk + FusedIntegerAggregate::CHUNK_HEADER_BYTES;
				room[p] = LINES_PER_CHUNK;
			}
			f = 0;
		}
		fill[p] = f;
	}
}

static void FusedAppend(FusedAggregateLocalState &lstate, idx_t row_width, idx_t count) {
	auto rows = lstate.row_buffer.get();
	auto hashes = lstate.hashes.get();
	switch (row_width) {
	case 8:
		return FusedAppendRows<8>(lstate, rows, hashes, count);
	case 12:
		return FusedAppendRows<12>(lstate, rows, hashes, count);
	case 16:
		return FusedAppendRows<16>(lstate, rows, hashes, count);
	case 20:
		return FusedAppendRows<20>(lstate, rows, hashes, count);
	case 24:
		return FusedAppendRows<24>(lstate, rows, hashes, count);
	case 28:
		return FusedAppendRows<28>(lstate, rows, hashes, count);
	case 32:
		return FusedAppendRows<32>(lstate, rows, hashes, count);
	default:
		throw InternalException("Fused integer aggregate: unsupported compact row width %llu", row_width);
	}
}

//! FusedAppendRows with the last-key fold. A row whose key words equal its partition's last row's adds its count
//! to that row and is not appended (unless the sum would pass uint32_t). A full line is flushed when its partition's next
//! row is kept, not when it fills, so the partition's last row is always the line's last row. Returns the rows appended.
template <idx_t W>
static idx_t FusedAppendFoldRows(FusedAggregateLocalState &lstate, const_data_ptr_t rows, const hash_t *hashes,
                                 idx_t count, idx_t count_offset, const FusedKeyShape &shape) {
	static constexpr idx_t ROWS_PER_LINE = FusedIntegerAggregate::LINE_BYTES / W;
	static constexpr idx_t LINE_ROW_BYTES = ROWS_PER_LINE * W;
	static constexpr idx_t LINES_PER_CHUNK =
	    (FusedIntegerAggregate::CHUNK_BYTES - FusedIntegerAggregate::CHUNK_HEADER_BYTES) / LINE_ROW_BYTES;
	auto lines = lstate.lines.get();
	auto fill = lstate.fill.get();
	auto cursor = lstate.cursor.get();
	auto room = lstate.room.get();
	idx_t appended = 0;
	for (idx_t i = 0; i < count; i++) {
		const auto row = rows + i * W;
		const auto p = FusedPartitionOf(hashes[i]);
		auto f = fill[p];
		const auto line = lines + p * FusedIntegerAggregate::LINE_BYTES;
		if (f > 0) {
			const auto last = line + (f - 1) * W;
			uint64_t key0, key1, last0, last1;
			FusedLoadKey(row, shape, key0, key1);
			FusedLoadKey(last, shape, last0, last1);
			const auto sum = uint64_t(Load<uint32_t>(last + count_offset)) + Load<uint32_t>(row + count_offset);
			if (key0 == last0 && key1 == last1 && sum <= NumericLimits<uint32_t>::Maximum()) {
				Store<uint32_t>(uint32_t(sum), last + count_offset);
				continue;
			}
			if (f == ROWS_PER_LINE) {
				memcpy(cursor[p], line, LINE_ROW_BYTES);
				cursor[p] += LINE_ROW_BYTES;
				if (--room[p] == 0) {
					// the chunk is full: seal it and take the next one now, so every later flush fits
					auto full = lstate.tail[p];
					Store<uint64_t>(LINES_PER_CHUNK * ROWS_PER_LINE, full + sizeof(data_ptr_t));
					auto chunk = FusedTakeChunk(lstate);
					Store<data_ptr_t>(chunk, full);
					lstate.tail[p] = chunk;
					cursor[p] = chunk + FusedIntegerAggregate::CHUNK_HEADER_BYTES;
					room[p] = LINES_PER_CHUNK;
				}
				f = 0;
			}
		} else if (!cursor[p]) {
			// the partition's first row: its first chunk
			auto chunk = FusedTakeChunk(lstate);
			lstate.head[p] = chunk;
			lstate.tail[p] = chunk;
			cursor[p] = chunk + FusedIntegerAggregate::CHUNK_HEADER_BYTES;
			room[p] = LINES_PER_CHUNK;
		}
		memcpy(line + f * W, row, W);
		fill[p] = UnsafeNumericCast<uint8_t>(f + 1);
		appended++;
	}
	return appended;
}

static idx_t FusedAppendFold(FusedAggregateLocalState &lstate, idx_t row_width, idx_t count_offset,
                             const FusedKeyShape &shape, idx_t count) {
	auto rows = lstate.row_buffer.get();
	auto hashes = lstate.hashes.get();
	switch (row_width) {
	case 8:
		return FusedAppendFoldRows<8>(lstate, rows, hashes, count, count_offset, shape);
	case 12:
		return FusedAppendFoldRows<12>(lstate, rows, hashes, count, count_offset, shape);
	case 16:
		return FusedAppendFoldRows<16>(lstate, rows, hashes, count, count_offset, shape);
	case 20:
		return FusedAppendFoldRows<20>(lstate, rows, hashes, count, count_offset, shape);
	case 24:
		return FusedAppendFoldRows<24>(lstate, rows, hashes, count, count_offset, shape);
	case 28:
		return FusedAppendFoldRows<28>(lstate, rows, hashes, count, count_offset, shape);
	case 32:
		return FusedAppendFoldRows<32>(lstate, rows, hashes, count, count_offset, shape);
	default:
		throw InternalException("FusedAppendFold: unsupported compact row width %llu", row_width);
	}
}

//! Moves every partial line into its partition's current chunk (always room for a line: never allocates)
static void FusedFlushLines(FusedAggregateLocalState &lstate, idx_t row_width) {
	for (idx_t p = 0; p < FusedIntegerAggregate::PARTITION_COUNT; p++) {
		const auto f = lstate.fill[p];
		if (f == 0) {
			continue;
		}
		D_ASSERT(lstate.cursor[p]);
		memcpy(lstate.cursor[p], lstate.lines.get() + p * FusedIntegerAggregate::LINE_BYTES, f * row_width);
		lstate.cursor[p] += f * row_width;
		lstate.fill[p] = 0;
	}
}

//! Writes each partition's current chunk's row count into its header; returns the rows of every partition list
static void FusedSealTails(FusedAggregateLocalState &lstate, idx_t row_width, idx_t *partition_rows) {
	for (idx_t p = 0; p < FusedIntegerAggregate::PARTITION_COUNT; p++) {
		auto tail = lstate.tail[p];
		if (!tail) {
			if (partition_rows) {
				partition_rows[p] = 0;
			}
			continue;
		}
		const auto used = NumericCast<idx_t>(lstate.cursor[p] - (tail + FusedIntegerAggregate::CHUNK_HEADER_BYTES));
		Store<uint64_t>(used / row_width, tail + sizeof(data_ptr_t));
		if (partition_rows) {
			idx_t rows = 0;
			for (auto chunk = lstate.head[p]; chunk; chunk = FusedChunkNext(chunk)) {
				rows += FusedChunkRows(chunk);
			}
			partition_rows[p] = rows;
		}
	}
}

//! the gid of `str` in the operator's map, inserted when new (its bytes copied into the map's arena unless the
//! string is inlined); false for a string past the MAXIMUM_GIDS-th. The caller holds gstate.gid_lock.
static bool FusedGidOf(FusedAggregateGlobalState &gstate, const string_t &str, uint16_t &gid) {
	auto entry = gstate.gid_of.find(str);
	if (entry != gstate.gid_of.end()) {
		gid = entry->second;
		return true;
	}
	if (gstate.gid_strings.size() >= FusedIntegerAggregate::MAXIMUM_GIDS) {
		return false;
	}
	string_t owned = str;
	if (!str.IsInlined()) {
		auto bytes = gstate.gid_arena.Allocate(str.GetSize());
		memcpy(bytes, str.GetData(), str.GetSize());
		owned = string_t(const_char_ptr_cast(bytes), UnsafeNumericCast<uint32_t>(str.GetSize()));
	}
	gid = UnsafeNumericCast<uint16_t>(gstate.gid_strings.size());
	gstate.gid_strings.push_back(owned);
	gstate.gid_of.emplace(owned, gid);
	gstate.gid_count = gstate.gid_strings.size();
	return true;
}

//! each VARCHAR key's rows of one input chunk mapped to their gids (the thread's gid rows). A dictionary vector
//! with an id and a flat child reads its code -> gid table, built once per (thread, dictionary id) by one locked pass
//! over the dictionary's entries (invalid slots skipped); any other key vector (flat, constant, an id-less dictionary)
//! maps its rows one by one through the map under one lock. False when a string would pass MAXIMUM_GIDS (a crossing).
static bool FusedMapGids(const FusedIntegerAggregate &fused, DataChunk &chunk, idx_t count,
                         FusedAggregateGlobalState &gstate, FusedAggregateLocalState &lstate) {
	for (idx_t column_idx = 0; column_idx < fused.columns.size(); column_idx++) {
		auto &column = fused.columns[column_idx];
		if (!column.gid) {
			continue;
		}
		auto &cache = lstate.gid_caches[column_idx];
		auto out = cache.rows.get();
		auto &key = chunk.data[column.chunk_index];
		if (key.GetVectorType() == VectorType::DICTIONARY_VECTOR && DictionaryVector::DictionarySize(key).IsValid() &&
		    !DictionaryVector::DictionaryId(key).empty() &&
		    DictionaryVector::Child(key).GetVectorType() == VectorType::FLAT_VECTOR) {
			const auto &dictionary_id = DictionaryVector::DictionaryId(key);
			const auto dictionary_size = DictionaryVector::DictionarySize(key).GetIndex();
			if (cache.dictionary_id != dictionary_id || cache.table.size() != dictionary_size) {
				auto &child = DictionaryVector::Child(key);
				auto entries = FlatVector::GetData<string_t>(child);
				auto &validity = FlatVector::Validity(child);
				cache.dictionary_id.clear();
				cache.table.assign(dictionary_size, 0);
				{
					lock_guard<mutex> guard(gstate.gid_lock);
					for (idx_t code = 0; code < dictionary_size; code++) {
						if (!validity.RowIsValid(code)) {
							continue;
						}
						if (!FusedGidOf(gstate, entries[code], cache.table[code])) {
							cache.table.clear();
							return false;
						}
					}
				}
				cache.dictionary_id = dictionary_id;
				gstate.gid_maps++;
			}
			auto &sel = DictionaryVector::SelVector(key);
			auto table = cache.table.data();
			for (idx_t i = 0; i < count; i++) {
				out[i] = table[sel.get_index(i)];
			}
			continue;
		}
		auto &format = lstate.formats[column_idx];
		auto strings = UnifiedVectorFormat::GetData<string_t>(format);
		auto &sel = *format.sel;
		{
			lock_guard<mutex> guard(gstate.gid_lock);
			for (idx_t i = 0; i < count; i++) {
				if (!FusedGidOf(gstate, strings[sel.get_index(i)], out[i])) {
					return false;
				}
			}
		}
		gstate.gid_rows_flat += count;
	}
	return true;
}

//! The compact rows of one input chunk: each key and input copied by width into the row buffer (a VARCHAR
//! key's gids from the thread's gid rows)
static void FusedGatherRows(const FusedIntegerAggregate &fused, FusedAggregateLocalState &lstate, idx_t count) {
	auto rows = lstate.row_buffer.get();
	for (idx_t column_idx = 0; column_idx < fused.columns.size(); column_idx++) {
		auto &column = fused.columns[column_idx];
		auto &format = lstate.formats[column_idx];
		if (column.gid) {
			auto gids = lstate.gid_caches[column_idx].rows.get();
			auto target = rows + column.offset;
			for (idx_t i = 0; i < count; i++) {
				Store<uint16_t>(gids[i], target + i * fused.row_width);
			}
			continue;
		}
		switch (column.width) {
		case 1:
			FusedGatherColumn<uint8_t>(format, count, rows, fused.row_width, column.offset);
			break;
		case 2:
			FusedGatherColumn<uint16_t>(format, count, rows, fused.row_width, column.offset);
			break;
		case 4:
			FusedGatherColumn<uint32_t>(format, count, rows, fused.row_width, column.offset);
			break;
		default:
			FusedGatherColumn<uint64_t>(format, count, rows, fused.row_width, column.offset);
			break;
		}
	}
}

//! the DISTINCT class's fold: a row whose key words equal the previous row's inside the chunk is skipped, so each run
//! of equal keys appends once; the first row always stays and no state crosses chunks. Compacts the row buffer in place
//! and returns the rows kept. A folding row is its key words: no non-distinct input is in the compact row (a mixed
//! shape's inputs are the companion's, read from the chunk).
static idx_t FusedFoldRows(data_ptr_t rows, idx_t count, idx_t row_width, const FusedKeyShape &shape) {
	if (count == 0) {
		return 0;
	}
	uint64_t previous0, previous1;
	FusedLoadKey(rows, shape, previous0, previous1);
	idx_t kept = 1;
	for (idx_t i = 1; i < count; i++) {
		const auto row = rows + i * row_width;
		uint64_t key0, key1;
		FusedLoadKey(row, shape, key0, key1);
		if (key0 == previous0 && key1 == previous1) {
			continue;
		}
		if (kept != i) {
			memcpy(rows + kept * row_width, row, row_width);
		}
		kept++;
		previous0 = key0;
		previous1 = key1;
	}
	return kept;
}

//! The last-key fold's chunk pass (the DISTINCT class's fold, carrying a count): every row's count is set to 1, and a row whose key words equal the
//! previous kept row's adds its count to that row and is skipped. Compacts the row buffer in place; returns the rows kept
//! (a count stays at most STANDARD_VECTOR_SIZE here)
static idx_t FusedFoldCountRows(data_ptr_t rows, idx_t count, idx_t row_width, idx_t count_offset,
                                const FusedKeyShape &shape) {
	if (count == 0) {
		return 0;
	}
	Store<uint32_t>(1, rows + count_offset);
	uint64_t previous0, previous1;
	FusedLoadKey(rows, shape, previous0, previous1);
	auto kept_row = rows;
	idx_t kept = 1;
	for (idx_t i = 1; i < count; i++) {
		const auto row = rows + i * row_width;
		uint64_t key0, key1;
		FusedLoadKey(row, shape, key0, key1);
		if (key0 == previous0 && key1 == previous1) {
			Store<uint32_t>(Load<uint32_t>(kept_row + count_offset) + 1, kept_row + count_offset);
			continue;
		}
		kept_row = rows + kept * row_width;
		if (kept != i) {
			memcpy(kept_row, row, row_width);
		}
		Store<uint32_t>(1, kept_row + count_offset);
		kept++;
		previous0 = key0;
		previous1 = key1;
	}
	return kept;
}

//===--------------------------------------------------------------------===//
// The companion: the mixed DISTINCT shape's per-thread g-keyed table over every input row, in the DISTINCT class's
// group entry layout {g words, occupancy marker, count, one int64 sum state per summed or averaged companion input}
//===--------------------------------------------------------------------===//
static inline idx_t FusedGroupKeyWords(const FusedIntegerAggregate &fused);
static inline idx_t FusedGroupWords(const FusedIntegerAggregate &fused);
static idx_t FusedTableBits(idx_t capacity);
static inline idx_t FusedGroupSlotOf(hash_t hash, idx_t capacity);
static void FusedAllocateGroupTable(BufferManager &buffer_manager, FusedGroupTable &table, idx_t capacity, idx_t words);

//! The capacity the thread's companion needs before `count` more rows are added: the one entry of the ungrouped form,
//! else the power of two (at least TABLE_MINIMUM_CAPACITY) holding occupancy + count at no more than half load, so
//! FusedGroupAdd's own growth rule never fires inside an add
static idx_t FusedCompanionTarget(const FusedIntegerAggregate &fused, optional_ptr<FusedGroupTable> table, idx_t count) {
	if (FusedGroupKeyWords(fused) == 0) {
		return 1;
	}
	const idx_t occupancy = table ? table->occupancy : 0;
	return NextPowerOfTwo(MaxValue<idx_t>(FusedIntegerAggregate::TABLE_MINIMUM_CAPACITY, 2 * (occupancy + count)));
}

//! One growth event: allocates the thread's companion at `capacity` in one Allocate (its first allocation included) and
//! re-inserts every occupied entry. Throws OutOfMemoryException (the buffer manager's) with the old table intact; the
//! caller is inside the chunk's crossing decision
static void FusedCompanionGrow(const FusedIntegerAggregate &fused, BufferManager &buffer_manager,
                               FusedAggregateLocalState &lstate, idx_t capacity) {
	const auto key_words = FusedGroupKeyWords(fused);
	const auto words = FusedGroupWords(fused);
	auto grown = make_uniq<FusedGroupTable>();
	FusedAllocateGroupTable(buffer_manager, *grown, capacity, words);
	auto old = lstate.companion.get();
	if (old && key_words == 0) {
		memcpy(grown->entries, old->entries, words * sizeof(uint64_t));
		grown->occupancy = old->occupancy;
	} else if (old) {
		const auto mask = grown->capacity - 1;
		for (idx_t slot = 0; slot < old->capacity; slot++) {
			auto old_entry = old->entries + slot * words;
			if (old_entry[key_words] == 0) {
				continue;
			}
			auto target = FusedGroupSlotOf(
			    FusedHashKey(old_entry[0], key_words == 2 ? old_entry[1] : 0, key_words == 2), grown->capacity);
			while (grown->entries[target * words + key_words] != 0) {
				target = (target + 1) & mask;
			}
			memcpy(grown->entries + target * words, old_entry, words * sizeof(uint64_t));
			grown->occupancy++;
		}
	}
	lstate.companion = std::move(grown);
}

template <class T>
static void FusedCompanionSum(const UnifiedVectorFormat &format, uint64_t *const *entries, idx_t count, idx_t sum_word) {
	auto source = UnifiedVectorFormat::GetDataUnsafe<T>(format);
	auto &sel = *format.sel;
	for (idx_t i = 0; i < count; i++) {
		// int64 states with wrapping addition, as FusedBuildTable's
		entries[i][sum_word] += uint64_t(int64_t(source[sel.get_index(i)]));
	}
}

//! Adds every row of the input chunk to the thread's companion from the chunk's own unified formats (g's key words
//! and the inputs read from the chunk, never from the compacted row buffer): the group found or inserted (its
//! occupancy marker set to 1), count += 1, each sum state += its input. The pre-grow made room: nothing allocates here
static void FusedCompanionAdd(const FusedIntegerAggregate &fused, FusedAggregateLocalState &lstate, idx_t count) {
	auto &table = *lstate.companion;
	const auto key_words = FusedGroupKeyWords(fused);
	const auto words = FusedGroupWords(fused);
	const auto count_word = key_words + 1;
	auto entries = lstate.companion_entries.get();
	if (key_words == 0) {
		// the ungrouped form: one entry
		auto entry = table.entries;
		entry[0] = 1;
		table.occupancy = 1;
		entry[count_word] += count;
		for (idx_t i = 0; i < count; i++) {
			entries[i] = entry;
		}
	} else {
		// g's bytes at the compact row's g offsets (16 bytes per row), read back as the key words FusedFoldPartition
		// reads from a (g, x) entry, so a companion group and its (g, x) group carry the same words
		static constexpr idx_t GROUP_ROW_BYTES = 16;
		auto rows = lstate.group_rows.get();
		for (idx_t key_idx = 0; key_idx < fused.key_count; key_idx++) {
			auto &column = fused.columns[key_idx];
			auto &format = lstate.formats[key_idx];
			switch (column.width) {
			case 1:
				FusedGatherColumn<uint8_t>(format, count, rows, GROUP_ROW_BYTES, column.offset);
				break;
			case 2:
				FusedGatherColumn<uint16_t>(format, count, rows, GROUP_ROW_BYTES, column.offset);
				break;
			case 4:
				FusedGatherColumn<uint32_t>(format, count, rows, GROUP_ROW_BYTES, column.offset);
				break;
			default:
				FusedGatherColumn<uint64_t>(format, count, rows, GROUP_ROW_BYTES, column.offset);
				break;
			}
		}
		const auto shape = FusedGetKeyShape(fused.group_bytes);
		auto hashes = lstate.hashes.get();
		for (idx_t i = 0; i < count; i++) {
			uint64_t key0, key1;
			FusedLoadKey(rows + i * GROUP_ROW_BYTES, shape, key0, key1);
			hashes[i] = FusedHashKey(key0, key1, shape.two_words);
		}
		// FusedGroupSlotOf's slot, its shift computed once per chunk
		const auto shift = 64 - FusedTableBits(table.capacity);
		const auto mask = table.capacity - 1;
		static constexpr idx_t PREFETCH_DISTANCE = 16;
		for (idx_t i = 0; i < count; i++) {
			if (i + PREFETCH_DISTANCE < count) {
				FUSED_PREFETCH_WRITE(table.entries + (hashes[i + PREFETCH_DISTANCE] >> shift) * words);
			}
			uint64_t key0, key1;
			FusedLoadKey(rows + i * GROUP_ROW_BYTES, shape, key0, key1);
			auto slot = hashes[i] >> shift;
			uint64_t *entry;
			while (true) {
				entry = table.entries + slot * words;
				if (entry[key_words] == 0) {
					entry[0] = key0;
					if (key_words == 2) {
						entry[1] = key1;
					}
					entry[key_words] = 1;
					table.occupancy++;
					break;
				}
				if (entry[0] == key0 && (key_words == 1 || entry[1] == key1)) {
					break;
				}
				slot = (slot + 1) & mask;
			}
			entry[count_word]++;
			entries[i] = entry;
		}
	}
	// the sum states, one companion input at a time
	for (idx_t sum_idx = 0; sum_idx < fused.companion_sums.size(); sum_idx++) {
		const auto input_idx = fused.companion_sums[sum_idx];
		auto &format = lstate.companion_formats[input_idx];
		const auto sum_word = count_word + 1 + sum_idx;
		switch (fused.companion_inputs[input_idx].width) {
		case 2:
			FusedCompanionSum<int16_t>(format, entries, count, sum_word);
			break;
		case 4:
			FusedCompanionSum<int32_t>(format, entries, count, sum_word);
			break;
		default:
			FusedCompanionSum<int64_t>(format, entries, count, sum_word);
			break;
		}
	}
}

//===--------------------------------------------------------------------===//
// The drain
//===--------------------------------------------------------------------===//
template <class T>
static void FusedScatterColumn(Vector &vector, idx_t start, const_data_ptr_t rows, idx_t count, idx_t row_width,
                               idx_t offset) {
	auto target = FlatVector::GetDataUnsafe<T>(vector) + start;
	auto source = rows + offset;
	for (idx_t i = 0; i < count; i++) {
		target[i] = Load<T>(source + i * row_width);
	}
}

//! Appends compact rows to the rebuilt chunk (the aggregate's input layout: each key and input at its own column). A
//! VARCHAR key column is rebuilt as the map's string for each row's gid (a view into the map's arena, which the
//! regular sink copies), the gids resolved under the map's lock once per call - never the gid bytes in a string_t slot
static void FusedScatterRows(const FusedIntegerAggregate &fused, FusedAggregateGlobalState &gstate, DataChunk &out,
                             const_data_ptr_t rows, idx_t count) {
	const auto start = out.size();
	for (auto &column : fused.columns) {
		auto &vector = out.data[column.chunk_index];
		if (column.gid) {
			auto target = FlatVector::GetData<string_t>(vector) + start;
			auto source = rows + column.offset;
			lock_guard<mutex> guard(gstate.gid_lock);
			for (idx_t i = 0; i < count; i++) {
				target[i] = gstate.gid_strings[Load<uint16_t>(source + i * fused.row_width)];
			}
			continue;
		}
		switch (column.width) {
		case 1:
			FusedScatterColumn<uint8_t>(vector, start, rows, count, fused.row_width, column.offset);
			break;
		case 2:
			FusedScatterColumn<uint16_t>(vector, start, rows, count, fused.row_width, column.offset);
			break;
		case 4:
			FusedScatterColumn<uint32_t>(vector, start, rows, count, fused.row_width, column.offset);
			break;
		default:
			FusedScatterColumn<uint64_t>(vector, start, rows, count, fused.row_width, column.offset);
			break;
		}
	}
	out.SetCardinality(start + count);
}

template <class T>
static void FusedFillColumn(Vector &vector, idx_t start, idx_t count, T value) {
	auto target = FlatVector::GetDataUnsafe<T>(vector) + start;
	for (idx_t i = 0; i < count; i++) {
		target[i] = value;
	}
}

//! writes one value of `width` bytes (the low bytes of `bits`, two's complement for a signed input) into rows
//! [start, start + count) of a rebuilt chunk's column
static void FusedFillWidth(Vector &vector, idx_t start, idx_t count, idx_t width, uint64_t bits) {
	switch (width) {
	case 1:
		FusedFillColumn<uint8_t>(vector, start, count, uint8_t(bits));
		break;
	case 2:
		FusedFillColumn<uint16_t>(vector, start, count, uint16_t(bits));
		break;
	case 4:
		FusedFillColumn<uint32_t>(vector, start, count, uint32_t(bits));
		break;
	default:
		FusedFillColumn<uint64_t>(vector, start, count, bits);
		break;
	}
}

//! Sets abandoned once (under the fused state's own lock) and takes every list handed over by an earlier Combine, with
//! its companion; records the first crossing's reason; true for the call that set it
static bool FusedAbandon(FusedAggregateGlobalState &gstate, FusedAggregateLocalState &lstate,
                         FusedCrossing reason = FusedCrossing::NONE) {
	lock_guard<mutex> guard(gstate.lock);
	if (gstate.abandoned) {
		return false;
	}
	gstate.crossing = uint8_t(reason);
	gstate.abandoned = true;
	gstate.combines_taken = gstate.handed_over.size();
	for (auto &lists : gstate.handed_over) {
		lstate.taken.push_back(std::move(lists));
	}
	gstate.handed_over.clear();
	return true;
}

//! Re-sinks every row this thread buffered (its taken-over lists, its own lists, its partial lines) as rebuilt chunks
//! through the operator's Sink, whose fused branch hands them to the regular Sink body, then frees them. Runs
//! outside the fused lock; from here on every chunk of this thread goes through the general code path. The DISTINCT
//! class re-sinks through the same body (SinkDistinct, then the regular sink): a folded shape rebuilds the
//! kept rows, which carry the chunk's distinct set. On the mixed shape the re-sink is split by table - the rebuilt
//! (g, x) rows go to the distinct table alone (resinking_distinct_only), then every companion the thread holds (its
//! own and each one it took at FusedAbandon) is re-expanded into the regular sink alone (resinking_regular_only): per
//! entry {g, k = count, sum_s}, k rows with g, x = 0 (no non-distinct aggregate reads x) and each summed or
//! averaged input carrying floor(sum_s / k) on every row plus 1 on the first sum_s mod k rows (a count-only input 0),
//! which gives count_star = k, count(input) = k, sum = sum_s and avg = sum_s / k exactly; then the companions are
//! dropped.
static void FusedDrain(const FusedIntegerAggregate &fused, ExecutionContext &context, OperatorSinkInput &input,
                       const PhysicalOperator &op, FusedAggregateGlobalState &gstate,
                       FusedAggregateLocalState &lstate) {
	FusedM2Flush(gstate, lstate);
	lstate.drained = true;
	lstate.resinking = true;
	auto &out = lstate.drain_chunk;
	if (out.ColumnCount() == 0) {
		out.Initialize(Allocator::Get(context.client), fused.input_types);
	}
	out.Reset();
	idx_t drained = 0;
	auto flush = [&]() {
		if (out.size() == 0) {
			return;
		}
		drained += out.size();
		op.Sink(context, out, input);
		out.Reset();
	};
	auto append = [&](const_data_ptr_t rows, idx_t count) {
		if (fused.last_key_fold) {
			// a folded row re-sinks as its count's rows (exact: the fold takes COUNT-only shapes alone)
			for (; count > 0; count--, rows += fused.row_width) {
				for (auto repeat = Load<uint32_t>(rows + fused.run_length_offset); repeat > 0; repeat--) {
					FusedScatterRows(fused, gstate, out, rows, 1);
					if (out.size() == STANDARD_VECTOR_SIZE) {
						flush();
					}
				}
			}
			return;
		}
		while (count > 0) {
			const auto take = MinValue<idx_t>(count, STANDARD_VECTOR_SIZE - out.size());
			FusedScatterRows(fused, gstate, out, rows, take);
			rows += take * fused.row_width;
			count -= take;
			if (out.size() == STANDARD_VECTOR_SIZE) {
				flush();
			}
		}
	};
	auto append_list = [&](data_ptr_t chunk) {
		for (; chunk; chunk = FusedChunkNext(chunk)) {
			append(chunk + FusedIntegerAggregate::CHUNK_HEADER_BYTES, FusedChunkRows(chunk));
		}
	};
	lstate.resinking_distinct_only = fused.mixed;
	for (auto &lists : lstate.taken) {
		for (idx_t p = 0; p < FusedIntegerAggregate::PARTITION_COUNT; p++) {
			append_list(lists->heads[p]);
		}
	}
	FusedSealTails(lstate, fused.row_width, nullptr);
	for (idx_t p = 0; p < FusedIntegerAggregate::PARTITION_COUNT; p++) {
		append_list(lstate.head[p]);
		append(lstate.lines.get() + p * FusedIntegerAggregate::LINE_BYTES, lstate.fill[p]);
	}
	flush();
	lstate.resinking_distinct_only = false;
	if (fused.mixed) {
		const auto key_words = FusedGroupKeyWords(fused);
		const auto words = FusedGroupWords(fused);
		const auto count_word = key_words + 1;
		const auto &x = fused.columns[fused.key_count];
		vector<idx_t> sum_of_input(fused.companion_inputs.size(), DConstants::INVALID_INDEX);
		for (idx_t sum_idx = 0; sum_idx < fused.companion_sums.size(); sum_idx++) {
			sum_of_input[fused.companion_sums[sum_idx]] = sum_idx;
		}
		auto expand = [&](optional_ptr<FusedGroupTable> table) {
			if (!table) {
				return;
			}
			for (idx_t slot = 0; slot < table->capacity; slot++) {
				auto entry = table->entries + slot * words;
				if (entry[key_words] == 0 || entry[count_word] == 0) {
					continue;
				}
				const idx_t k = entry[count_word];
				// per sum state, Python-style floor division: sum = k * quotient + remainder, 0 <= remainder < k
				int64_t quotient[FusedIntegerAggregate::MAXIMUM_AGGREGATES];
				idx_t remainder[FusedIntegerAggregate::MAXIMUM_AGGREGATES];
				for (idx_t sum_idx = 0; sum_idx < fused.companion_sums.size(); sum_idx++) {
					const auto sum = int64_t(entry[count_word + 1 + sum_idx]);
					const auto divisor = int64_t(k);
					auto q = sum / divisor;
					auto r = sum % divisor;
					if (r < 0) {
						q -= 1;
						r += divisor;
					}
					quotient[sum_idx] = q;
					remainder[sum_idx] = idx_t(r);
				}
				for (idx_t done = 0; done < k;) {
					const auto take = MinValue<idx_t>(k - done, STANDARD_VECTOR_SIZE - out.size());
					const auto start = out.size();
					for (idx_t key_idx = 0; key_idx < fused.key_count; key_idx++) {
						// the key words hold g's compact-row bytes, so each key is read back at its own offset
						auto &column = fused.columns[key_idx];
						uint64_t bits = 0;
						memcpy(&bits, reinterpret_cast<const_data_ptr_t>(entry) + column.offset, column.width);
						FusedFillWidth(out.data[column.chunk_index], start, take, column.width, bits);
					}
					FusedFillWidth(out.data[x.chunk_index], start, take, x.width, 0);
					for (idx_t input_idx = 0; input_idx < fused.companion_inputs.size(); input_idx++) {
						auto &input = fused.companion_inputs[input_idx];
						auto &vector = out.data[input.chunk_index];
						const auto sum_idx = sum_of_input[input_idx];
						if (sum_idx == DConstants::INVALID_INDEX) {
							// a count-only input: any in-range non-NULL value
							FusedFillWidth(vector, start, take, input.width, 0);
							continue;
						}
						const auto plus =
						    remainder[sum_idx] > done ? MinValue<idx_t>(take, remainder[sum_idx] - done) : idx_t(0);
						FusedFillWidth(vector, start, plus, input.width, uint64_t(quotient[sum_idx] + 1));
						FusedFillWidth(vector, start + plus, take - plus, input.width, uint64_t(quotient[sum_idx]));
					}
					out.SetCardinality(start + take);
					done += take;
					if (out.size() == STANDARD_VECTOR_SIZE) {
						flush();
					}
				}
			}
		};
		lstate.resinking_regular_only = true;
		for (auto &lists : lstate.taken) {
			expand(lists->companion.get());
		}
		expand(lstate.companion.get());
		flush();
		lstate.resinking_regular_only = false;
		lstate.companion.reset();
	}
	gstate.drained_rows += drained;
	// free every buffered row
	lstate.taken.clear();
	lstate.pool.clear();
	lstate.slabs.clear();
	for (idx_t p = 0; p < FusedIntegerAggregate::PARTITION_COUNT; p++) {
		lstate.fill[p] = 0;
		lstate.cursor[p] = nullptr;
		lstate.room[p] = 0;
		lstate.tail[p] = nullptr;
		lstate.head[p] = nullptr;
	}
	lstate.resinking = false;
}

//! The run kind's drain: re-sinks every row this state buffered (its taken-over lists, its own lists, its partial
//! lines), run and ordinary alike, through the radix path's SinkRuns on this state's radix local state with each row's
//! run length (1 for an ordinary row; at most STANDARD_VECTOR_SIZE, since RLEScanRuns clips a run at its vector, so it
//! fits RunSink's uint16_t) in batches of at most STANDARD_VECTOR_SIZE rows - one path, exact by the radix path's own
//! grouped run update, never expanded to rows - then frees them. From here on every batch or chunk of this state goes
//! through the general code path.
static void FusedDrainRuns(const FusedIntegerAggregate &fused, ClientContext &context, const PhysicalHashAggregate &op,
                           FusedAggregateGlobalState &gstate, FusedAggregateLocalState &lstate) {
	FusedM2Flush(gstate, lstate);
	lstate.drained = true;
	auto &radix = op.groupings[0].table_data;
	auto &out = lstate.drain_chunk;
	if (out.ColumnCount() == 0) {
		vector<LogicalType> types;
		types.push_back(radix.group_types[0]);
		out.Initialize(Allocator::Get(context), types);
	}
	if (!lstate.drain_counts) {
		lstate.drain_counts = make_unsafe_uniq_array_uninitialized<uint16_t>(STANDARD_VECTOR_SIZE);
	}
	out.Reset();
	auto counts = lstate.drain_counts.get();
	auto &key = fused.columns[0];
	idx_t drained = 0;
	auto flush = [&]() {
		if (out.size() == 0) {
			return;
		}
		drained += out.size();
		radix.SinkRuns(context, *gstate.run_radix_global, *lstate.run_radix_local, out, counts, out.size(),
		               op.run_aggregate->grouped_run_updates);
		out.Reset();
	};
	auto append = [&](const_data_ptr_t rows, idx_t count) {
		while (count > 0) {
			const auto start = out.size();
			const auto take = MinValue<idx_t>(count, STANDARD_VECTOR_SIZE - start);
			switch (key.width) {
			case 1:
				FusedScatterColumn<uint8_t>(out.data[0], start, rows, take, fused.row_width, key.offset);
				break;
			case 2:
				FusedScatterColumn<uint16_t>(out.data[0], start, rows, take, fused.row_width, key.offset);
				break;
			case 4:
				FusedScatterColumn<uint32_t>(out.data[0], start, rows, take, fused.row_width, key.offset);
				break;
			default:
				FusedScatterColumn<uint64_t>(out.data[0], start, rows, take, fused.row_width, key.offset);
				break;
			}
			for (idx_t i = 0; i < take; i++) {
				counts[start + i] =
				    UnsafeNumericCast<uint16_t>(Load<uint32_t>(rows + i * fused.row_width + fused.run_length_offset));
			}
			out.SetCardinality(start + take);
			rows += take * fused.row_width;
			count -= take;
			if (out.size() == STANDARD_VECTOR_SIZE) {
				flush();
			}
		}
	};
	auto append_list = [&](data_ptr_t chunk) {
		for (; chunk; chunk = FusedChunkNext(chunk)) {
			append(chunk + FusedIntegerAggregate::CHUNK_HEADER_BYTES, FusedChunkRows(chunk));
		}
	};
	for (auto &lists : lstate.taken) {
		for (idx_t p = 0; p < FusedIntegerAggregate::PARTITION_COUNT; p++) {
			append_list(lists->heads[p]);
		}
	}
	FusedSealTails(lstate, fused.row_width, nullptr);
	for (idx_t p = 0; p < FusedIntegerAggregate::PARTITION_COUNT; p++) {
		append_list(lstate.head[p]);
		append(lstate.lines.get() + p * FusedIntegerAggregate::LINE_BYTES, lstate.fill[p]);
	}
	flush();
	gstate.drained_rows += drained;
	// free every buffered row
	lstate.taken.clear();
	lstate.pool.clear();
	lstate.slabs.clear();
	for (idx_t p = 0; p < FusedIntegerAggregate::PARTITION_COUNT; p++) {
		lstate.fill[p] = 0;
		lstate.cursor[p] = nullptr;
		lstate.room[p] = 0;
		lstate.tail[p] = nullptr;
		lstate.head[p] = nullptr;
	}
}

//===--------------------------------------------------------------------===//
// The stored hash
//===--------------------------------------------------------------------===//
//! each row's hash, computed once, and its stored bits written into the row at hash_offset
static void FusedStoreHash(data_ptr_t rows, hash_t *hashes, idx_t count, idx_t row_width, idx_t hash_offset,
                           const FusedKeyShape &shape) {
	for (idx_t i = 0; i < count; i++) {
		const auto row = rows + i * row_width;
		uint64_t key0, key1;
		FusedLoadKey(row, shape, key0, key1);
		const auto hash = FusedHashKey(key0, key1, shape.two_words);
		hashes[i] = hash;
		Store<uint32_t>(FusedStoredBits(hash), row + hash_offset);
	}
}

//===--------------------------------------------------------------------===//
// The bitmap class, phase 1
//===--------------------------------------------------------------------===//
//! Sets the chunk's code bits in the thread's bitmap (buffer-manager memory, allocated and zeroed at its first chunk): a
//! vector over the published child reads its codes from its selection (code 0, a NULL row, is set here and excluded in
//! phase 2); any other vector looks each non-NULL string up, a string the dictionary does not hold kept in the thread's
//! overflow set
static void FusedBitmapSink(const FusedIntegerAggregate &fused, DataChunk &chunk, FusedAggregateGlobalState &gstate,
                            FusedAggregateLocalState &lstate) {
	const idx_t count = chunk.size();
	if (count == 0) {
		return;
	}
	lstate.bitmap_rows += count;
	if (!lstate.bitmap) {
		const idx_t bytes = fused.bitmap_words * sizeof(uint64_t);
		lstate.bitmap_handle = gstate.buffer_manager.Allocate(MemoryTag::HASH_TABLE, bytes, true);
		lstate.bitmap = reinterpret_cast<uint64_t *>(lstate.bitmap_handle.Ptr());
		memset(lstate.bitmap, 0, bytes);
	}
	auto bits = lstate.bitmap;
	auto &dict = *fused.bitmap_dict;
	auto &x = chunk.data[fused.columns[0].chunk_index];
	if (x.GetVectorType() == VectorType::DICTIONARY_VECTOR && &DictionaryVector::Child(x) == &dict.child->data) {
		auto &sel = DictionaryVector::SelVector(x);
		for (idx_t i = 0; i < count; i++) {
			const idx_t code = sel.get_index(i);
			D_ASSERT(code < dict.count);
			// a set bit is only read, so a run of one code (a hot value) stores nothing
			const uint64_t mask = uint64_t(1) << (code & 63);
			auto &word = bits[code >> 6];
			if (!(word & mask)) {
				word |= mask;
			}
		}
		return;
	}
	lstate.bitmap_lookups += count;
	UnifiedVectorFormat format;
	x.ToUnifiedFormat(count, format);
	auto strings = UnifiedVectorFormat::GetData<string_t>(format);
	for (idx_t i = 0; i < count; i++) {
		const auto idx = format.sel->get_index(i);
		if (!format.validity.RowIsValid(idx)) {
			continue;
		}
		auto &value = strings[idx];
		const auto code = dict.Lookup(value, Hash(value));
		if (code == dict_global::ColumnDictionary::INVALID_CODE) {
			lstate.bitmap_overflow.insert(value.GetString());
			continue;
		}
		bits[code >> 6] |= uint64_t(1) << (code & 63);
	}
}

//! Combine: the thread's bitmap, overflow strings and input rows handed over under the lock
static void FusedBitmapHandOver(FusedAggregateGlobalState &gstate, FusedAggregateLocalState &lstate) {
	lock_guard<mutex> guard(gstate.lock);
	gstate.input_rows += lstate.bitmap_rows;
	gstate.bitmap_lookups += lstate.bitmap_lookups;
	lstate.bitmap_rows = 0;
	lstate.bitmap_lookups = 0;
	if (lstate.bitmap) {
		gstate.bitmaps.push_back(std::move(lstate.bitmap_handle));
		lstate.bitmap = nullptr;
	}
	for (auto &text : lstate.bitmap_overflow) {
		gstate.bitmap_overflow.insert(text);
	}
	lstate.bitmap_overflow.clear();
}

//===--------------------------------------------------------------------===//
// Sink, Combine, Finalize
//===--------------------------------------------------------------------===//
bool FusedIntegerAggregate::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input,
                                 const PhysicalOperator &op, FusedAggregateGlobalState &gstate,
                                 FusedAggregateLocalState &lstate) const {
	if (bitmap) {
		if (gstate.abandoned) {
			// this execution was refused at its start - the hash aggregate's Sink body takes the chunk
			return false;
		}
		// the bitmap class never crosses, so every chunk stays in the fused path
		FusedBitmapSink(*this, chunk, gstate, lstate);
		return true;
	}
	if (lstate.resinking) {
		return false;
	}
	const idx_t count = chunk.size();
	FusedM2Count(gstate.input_rows, lstate, FUSED_M2_INPUT_ROWS, count);
	if (lstate.drained) {
		return false;
	}
	if (gstate.abandoned) {
		if (run_kind) {
			FusedDrainRuns(*this, context.client, op.Cast<PhysicalHashAggregate>(), gstate, lstate);
		} else {
			FusedDrain(*this, context, input, op, gstate, lstate);
		}
		return false;
	}
	if (count == 0) {
		return true;
	}
	// the input chunk is the unit of atomicity: both checks and the pool top-up precede any append
	// (i) a NULL key or input (only a plan the data has outrun can deliver one) is a crossing, in the companion's
	// input columns too
	bool crossing = false;
	auto reason = FusedCrossing::NONE;
	for (idx_t column_idx = 0; column_idx < columns.size(); column_idx++) {
		auto &format = lstate.formats[column_idx];
		chunk.data[columns[column_idx].chunk_index].ToUnifiedFormat(count, format);
		if (FusedHasNull(format, count)) {
			crossing = true;
		}
	}
	for (idx_t input_idx = 0; input_idx < companion_inputs.size(); input_idx++) {
		auto &format = lstate.companion_formats[input_idx];
		chunk.data[companion_inputs[input_idx].chunk_index].ToUnifiedFormat(count, format);
		if (FusedHasNull(format, count)) {
			crossing = true;
		}
	}
	// each VARCHAR key's rows mapped to their gids before anything is gathered, reserved or appended; a string
	// past MAXIMUM_GIDS (the estimate outrun at run time) is a crossing
	if (!crossing && gid_keys > 0 && !FusedMapGids(*this, chunk, count, gstate, lstate)) {
		crossing = true;
	}
	// a folding shape gathers its compact rows first (thread-local) and folds adjacent equal keys, so only the
	// kept rows are reserved and appended
	idx_t append_count = count;
	if (fold && !crossing) {
		FusedGatherRows(*this, lstate, count);
		append_count = FusedFoldRows(lstate.row_buffer.get(), count, row_width, FusedGetKeyShape(key_bytes));
	} else if (last_key_fold && !crossing) {
		// the last-key fold: the same, each kept row counting the rows it took
		FusedGatherRows(*this, lstate, count);
		append_count = FusedFoldCountRows(lstate.row_buffer.get(), count, row_width, run_length_offset,
		                                  FusedGetKeyShape(key_bytes));
	}
	// (ii) the chunk's rows' bytes are reserved against the drain threshold; a reservation that would cross it is a
	// crossing with nothing of the chunk appended. The companion's growth bytes are reserved with them (the
	// table pre-grown, inside this decision, to hold occupancy + count at no more than half load), so the whole
	// operator stays under the budget the admission set; a refused reservation names which part crossed
	const idx_t bytes = append_count * row_width;
	idx_t grow_capacity = 0;
	idx_t grow_bytes = 0;
	if (mixed && !crossing) {
		const auto target = FusedCompanionTarget(*this, lstate.companion.get(), count);
		if (!lstate.companion || target > lstate.companion->capacity) {
			grow_capacity = target;
			grow_bytes = target * FusedGroupWords(*this) * sizeof(uint64_t);
		}
	}
	bool reserved = false;
	if (!crossing) {
		// the counter as the refused reservation read it (the fetch_add's previous value, or the loop's last load), for
		// the reason
		idx_t current = 0;
		if (fused_local_allowance) {
			// from the state's allowance, refilled by one grant when short; the rows' and growth bytes together
			reserved = FusedM2Reserve(gstate, lstate, bytes + grow_bytes, current);
		} else if (fused_atomic_reserve) {
			// one atomic add, rolled back when the sum crosses the threshold (no retry loop); the companion's growth
			// bytes are reserved with the rows', as the loop reserves them
			current = gstate.reserved_bytes.fetch_add(bytes + grow_bytes);
			if (current + bytes + grow_bytes <= gstate.drain_threshold) {
				reserved = true;
			} else {
				gstate.reserved_bytes.fetch_sub(bytes + grow_bytes);
				reserved = false;
			}
		} else {
			current = gstate.reserved_bytes.load();
			while (current + bytes + grow_bytes <= gstate.drain_threshold) {
				if (gstate.reserved_bytes.compare_exchange_weak(current, current + bytes + grow_bytes)) {
					reserved = true;
					break;
				}
			}
		}
		if (!reserved) {
			crossing = true;
			reason = current + bytes <= gstate.drain_threshold ? FusedCrossing::COMPANION_GROWTH : FusedCrossing::ROWS;
		}
	}
	// the companion's growth, one Allocate at its target size; a failed allocation is a crossing with nothing of the
	// chunk appended or added
	if (!crossing && grow_capacity > 0) {
		try {
			FusedCompanionGrow(*this, gstate.buffer_manager, lstate, grow_capacity);
		} catch (OutOfMemoryException &) {
			gstate.reserved_bytes -= bytes + grow_bytes;
			crossing = true;
			reason = FusedCrossing::OOM;
		}
	}
	// the pool: at least POOL_MINIMUM_FREE_CHUNKS free chunks, one per row an append can take; a failed allocation
	// is a crossing with nothing appended
	if (!crossing) {
		try {
			FusedTopUpPool(gstate, lstate);
		} catch (OutOfMemoryException &) {
			gstate.reserved_bytes -= bytes + grow_bytes;
			crossing = true;
			reason = FusedCrossing::OOM;
		}
	}
	if (crossing) {
		FusedAbandon(gstate, lstate, reason);
		if (run_kind) {
			FusedDrainRuns(*this, context.client, op.Cast<PhysicalHashAggregate>(), gstate, lstate);
		} else {
			FusedDrain(*this, context, input, op, gstate, lstate);
		}
		return false;
	}
	// only after the crossing decision, every row of the chunk is added to the companion (from the chunk's own
	// unified formats), so a crossing chunk is never half-added and no row is counted twice
	if (mixed) {
		FusedCompanionAdd(*this, lstate, count);
		FusedM2Count(gstate.companion_rows, lstate, FUSED_M2_COMPANION_ROWS, count);
	}
	// the compact rows, their hashes, then the append into the partitions through the write-combining lines
	auto rows = lstate.row_buffer.get();
	if (fold) {
		FusedM2Count(gstate.folded_rows, lstate, FUSED_M2_FOLDED_ROWS, count - append_count);
	} else if (!last_key_fold) {
		FusedGatherRows(*this, lstate, count);
	}
	if (run_kind) {
		for (idx_t i = 0; i < count; i++) {
			Store<uint32_t>(1, rows + i * row_width + run_length_offset);
		}
	}
	const auto shape = FusedGetKeyShape(key_bytes);
	auto hashes = lstate.hashes.get();
	if (hash_stored) {
		FusedStoreHash(rows, hashes, append_count, row_width, hash_offset, shape);
		FusedM2Count(gstate.hash_stored_rows, lstate, FUSED_M2_HASH_STORED_ROWS, append_count);
	} else {
		for (idx_t i = 0; i < append_count; i++) {
			uint64_t key0, key1;
			FusedLoadKey(rows + i * row_width, shape, key0, key1);
			hashes[i] = FusedHashKey(key0, key1, shape.two_words);
		}
	}
	if (last_key_fold) {
		// the rows the partition fold took return their reserved bytes
		const auto appended = FusedAppendFold(lstate, row_width, run_length_offset, shape, append_count);
		const idx_t unused = (append_count - appended) * row_width;
		if (fused_local_allowance) {
			FusedM2Words(lstate)[FUSED_M2_ALLOWANCE] += unused;
		} else {
			gstate.reserved_bytes -= unused;
		}
		FusedM2Count(gstate.folded_rows, lstate, FUSED_M2_FOLDED_ROWS, count - appended);
		return true;
	}
	FusedAppend(lstate, row_width, append_count);
	return true;
}

bool FusedIntegerAggregate::SinkRuns(ClientContext &context, Vector &values, const uint16_t *counts, idx_t run_count,
                                     const PhysicalHashAggregate &op, FusedAggregateGlobalState &gstate,
                                     FusedAggregateLocalState &lstate) const {
	D_ASSERT(run_kind && run_count <= STANDARD_VECTOR_SIZE);
	idx_t length_sum = 0;
	for (idx_t i = 0; i < run_count; i++) {
		length_sum += counts[i];
	}
	FusedM2Count(gstate.run_rows, lstate, FUSED_M2_RUN_ROWS, run_count);
	FusedM2Count(gstate.run_length_sum, lstate, FUSED_M2_RUN_LENGTH_SUM, length_sum);
	if (lstate.drained) {
		return false;
	}
	if (gstate.abandoned) {
		FusedDrainRuns(*this, context, op, gstate, lstate);
		return false;
	}
	if (run_count == 0) {
		return true;
	}
	// the batch is the unit of atomicity, as Sink's input chunk: the checks and the pool top-up precede any append
	// (i) a NULL key is a crossing (the run branch never delivers one: its columns cannot hold NULL)
	auto &format = lstate.formats[0];
	values.ToUnifiedFormat(run_count, format);
	bool crossing = FusedHasNull(format, run_count);
	// (ii) the batch's rows' bytes are reserved against the drain threshold
	const idx_t bytes = run_count * row_width;
	bool reserved = false;
	if (!crossing) {
		if (fused_local_allowance) {
			idx_t current = 0;
			reserved = FusedM2Reserve(gstate, lstate, bytes, current);
		} else if (fused_atomic_reserve) {
			const idx_t previous = gstate.reserved_bytes.fetch_add(bytes);
			if (previous + bytes <= gstate.drain_threshold) {
				reserved = true;
			} else {
				gstate.reserved_bytes.fetch_sub(bytes);
				reserved = false;
			}
		} else {
			auto current = gstate.reserved_bytes.load();
			while (current + bytes <= gstate.drain_threshold) {
				if (gstate.reserved_bytes.compare_exchange_weak(current, current + bytes)) {
					reserved = true;
					break;
				}
			}
		}
		crossing = !reserved;
	}
	// (iii) the pool: one free chunk per row an append can take; a failed allocation is a crossing
	if (!crossing) {
		try {
			FusedTopUpPool(gstate, lstate);
		} catch (OutOfMemoryException &) {
			gstate.reserved_bytes -= bytes;
			crossing = true;
		}
	}
	if (crossing) {
		FusedAbandon(gstate, lstate);
		FusedDrainRuns(*this, context, op, gstate, lstate);
		return false;
	}
	// the compact rows {key, run length}, their hashes, then the append
	auto rows = lstate.row_buffer.get();
	auto &key = columns[0];
	switch (key.width) {
	case 1:
		FusedGatherColumn<uint8_t>(format, run_count, rows, row_width, key.offset);
		break;
	case 2:
		FusedGatherColumn<uint16_t>(format, run_count, rows, row_width, key.offset);
		break;
	case 4:
		FusedGatherColumn<uint32_t>(format, run_count, rows, row_width, key.offset);
		break;
	default:
		FusedGatherColumn<uint64_t>(format, run_count, rows, row_width, key.offset);
		break;
	}
	for (idx_t i = 0; i < run_count; i++) {
		Store<uint32_t>(counts[i], rows + i * row_width + run_length_offset);
	}
	const auto shape = FusedGetKeyShape(key_bytes);
	auto hashes = lstate.hashes.get();
	for (idx_t i = 0; i < run_count; i++) {
		uint64_t key0, key1;
		FusedLoadKey(rows + i * row_width, shape, key0, key1);
		hashes[i] = FusedHashKey(key0, key1, shape.two_words);
	}
	FusedAppend(lstate, row_width, run_count);
	return true;
}

//! Combine's hand-over: the partial lines go into their partitions' current chunks (allocation-free), the lists are
//! sealed and, under the fused lock, handed over with their slabs; false when the operator is abandoned (the caller
//! drains the lists outside the lock)
static bool FusedHandOver(const FusedIntegerAggregate &fused, FusedAggregateGlobalState &gstate,
                          FusedAggregateLocalState &lstate) {
	FusedFlushLines(lstate, fused.row_width);
	auto lists = make_uniq<FusedThreadPartitions>();
	lists->rows.resize(FusedIntegerAggregate::PARTITION_COUNT);
	FusedSealTails(lstate, fused.row_width, lists->rows.data());
	lists->heads.assign(lstate.head.get(), lstate.head.get() + FusedIntegerAggregate::PARTITION_COUNT);
	{
		lock_guard<mutex> guard(gstate.lock);
		if (gstate.abandoned) {
			return false;
		}
		lists->slabs = std::move(lstate.slabs);
		lists->companion = std::move(lstate.companion);
		gstate.handed_over.push_back(std::move(lists));
	}
	// the local state keeps nothing: the executor resets it right after Combine
	lstate.pool.clear();
	for (idx_t p = 0; p < FusedIntegerAggregate::PARTITION_COUNT; p++) {
		lstate.cursor[p] = nullptr;
		lstate.tail[p] = nullptr;
		lstate.head[p] = nullptr;
	}
	return true;
}

bool FusedIntegerAggregate::Combine(ExecutionContext &context, OperatorSinkInput &input,
                                    const PhysicalOperator &op, FusedAggregateGlobalState &gstate,
                                    FusedAggregateLocalState &lstate) const {
	if (bitmap) {
		if (gstate.abandoned) {
			// refused at its start - no bitmap to hand over; the hash aggregate's Combine runs
			return false;
		}
		// no allowance words, no partition lists
		FusedBitmapHandOver(gstate, lstate);
		return true;
	}
	// the state is done sinking: its unused allowance and counters go to the global state before every path below
	FusedM2Flush(gstate, lstate);
	if (run_kind) {
		return CombineRuns(context.client, op.Cast<PhysicalHashAggregate>(), gstate, lstate);
	}
	if (lstate.drained) {
		return false;
	}
	if (!FusedHandOver(*this, gstate, lstate)) {
		FusedDrain(*this, context, input, op, gstate, lstate);
		return false;
	}
	return true;
}

bool FusedIntegerAggregate::CombineRuns(ClientContext &context, const PhysicalHashAggregate &op,
                                        FusedAggregateGlobalState &gstate, FusedAggregateLocalState &lstate) const {
	D_ASSERT(run_kind);
	// the state is done sinking: its unused allowance and counters go to the global state before every path below
	FusedM2Flush(gstate, lstate);
	if (lstate.drained) {
		return false;
	}
	if (!FusedHandOver(*this, gstate, lstate)) {
		FusedDrainRuns(*this, context, op, gstate, lstate);
		return false;
	}
	return true;
}

bool FusedIntegerAggregate::Finalize(FusedAggregateGlobalState &gstate) const {
	if (bitmap) {
		if (gstate.abandoned) {
			// refused at its start - every row is in the hash aggregate's states; FinalizeDistinct runs
			return false;
		}
		// phase 2's stripes, counted in partitions_nonempty (the source's MaxThreads reads it); none without a
		// bitmap, and then one source task emits the overflow count alone
		gstate.partitions_nonempty =
		    gstate.bitmaps.empty() ? 0 : (bitmap_words + BITMAP_STRIPE_WORDS - 1) / BITMAP_STRIPE_WORDS;
		gstate.finalized = true;
		return true;
	}
	if (gstate.abandoned) {
		if (!gstate.handed_over.empty()) {
			throw InternalException("Fused integer aggregate: partition lists left behind by the drain");
		}
		return false;
	}
	gstate.partition_rows.assign(PARTITION_COUNT, 0);
	for (auto &lists : gstate.handed_over) {
		for (idx_t p = 0; p < PARTITION_COUNT; p++) {
			gstate.partition_rows[p] += lists->rows[p];
		}
	}
	idx_t nonempty = 0;
	for (idx_t p = 0; p < PARTITION_COUNT; p++) {
		nonempty += gstate.partition_rows[p] != 0;
	}
	gstate.partitions_nonempty = nonempty;
	gstate.finalized = true;
	return true;
}

//===--------------------------------------------------------------------===//
// Phase 2: one table per partition, sized once, scanned once
//===--------------------------------------------------------------------===//
class FusedAggregateGlobalSourceState : public GlobalSourceState {
public:
	FusedAggregateGlobalSourceState(ClientContext &context, FusedAggregateGlobalState &sink_p)
	    : sink(sink_p), next_partition(0), claimed(0), arrived(false), tasks_started(0), tasks_done(0), merged(false),
	      merge_done(false), distinct_entries(0), groups(0), tasks_merged(0), companion_groups(0), next_companion(0),
	      parked(false) {
		const auto threads = NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
		max_threads = MaxValue<idx_t>(1, MinValue<idx_t>(threads, sink.partitions_nonempty.load()));
	}

	idx_t MaxThreads() override {
		return max_threads;
	}

	FusedAggregateGlobalState &sink;
	atomic<idx_t> next_partition;
	atomic<idx_t> claimed;
	idx_t max_threads;

	//! The DISTINCT class's merge. Before its first fetch of next_partition a task counts itself under the lock;
	//! a task whose fetch finds no partition hands its group table in under the lock, and the one that brings
	//! tasks_done up to tasks_started while merged is unset sets merged and takes every handed-in table: every task
	//! that fetched counted itself first, so every claimed partition is in a handed-in table, and the merge runs once
	atomic<bool> arrived;
	mutex merge_lock;
	idx_t tasks_started;
	idx_t tasks_done;
	vector<unique_ptr<FusedGroupTable>> handed_in;
	bool merged;
	//! Set by the merger (under the lock) after the merge and before its first emitted chunk; the phase-2 counters are
	//! emitted only from then on (recorded only)
	bool merge_done;
	idx_t distinct_entries;
	idx_t groups;
	idx_t tasks_merged;
	//! the companion entries merged into the final group table (equal to groups on every mixed operator)
	idx_t companion_groups;
	//! the tree merge (FusedTreeMerge): the next sink thread's companion a task claims, and the
	//! one parked pair (a task's group table and its companions folded into one), under merge_lock
	atomic<idx_t> next_companion;
	bool parked;
	unique_ptr<FusedGroupTable> parked_groups;
	unique_ptr<FusedGroupTable> parked_companion;
};

class FusedAggregateLocalSourceState : public LocalSourceState {
public:
	explicit FusedAggregateLocalSourceState(BufferManager &buffer_manager_p)
	    : buffer_manager(buffer_manager_p), table(nullptr), buffer_entries(0), capacity(0), table_bits(0),
	      occupancy(0), active(false), scan_position(0), counted(false), distinct_entries(0), emit_done(false),
	      chain_directory(nullptr), chain_directory_entries(0), chain_groups(nullptr), chain_group_entries(0),
	      chain_group_count(0) {
	}

	BufferManager &buffer_manager;
	//! The task's table buffer (buffer-manager memory), reused across its partitions: all-zero between partitions
	BufferHandle handle;
	uint64_t *table;
	idx_t buffer_entries;
	//! The current partition's table
	idx_t capacity;
	idx_t table_bits;
	idx_t occupancy;
	bool active;
	idx_t scan_position;
	//! Occupied entries of one output chunk
	uint64_t *emit[STANDARD_VECTOR_SIZE];

	//! whether the task has counted itself, its group table, the tables it merged (the last task) and the one it
	//! emits from, its (g, x) entries folded
	bool counted;
	unique_ptr<FusedGroupTable> groups;
	vector<unique_ptr<FusedGroupTable>> merged_tables;
	optional_ptr<FusedGroupTable> emit_table;
	idx_t distinct_entries;
	bool emit_done;

	//! the chained build: the task's directory (4-byte group ordinals, 0 = empty) and its append-only group array
	//! ({group row, tag << 32 | next, count, sums}), buffer-manager memory reused across its partitions; the current
	//! partition's groups; one output chunk's group rows
	BufferHandle chain_directory_handle;
	uint32_t *chain_directory;
	idx_t chain_directory_entries;
	BufferHandle chain_groups_handle;
	uint64_t *chain_groups;
	idx_t chain_group_entries;
	idx_t chain_group_count;
	const_data_ptr_t emit_rows[STANDARD_VECTOR_SIZE];
};

unique_ptr<GlobalSourceState> FusedIntegerAggregate::GetGlobalSourceState(ClientContext &context,
                                                                           FusedAggregateGlobalState &gstate) const {
	return make_uniq<FusedAggregateGlobalSourceState>(context, gstate);
}

unique_ptr<LocalSourceState> FusedIntegerAggregate::GetLocalSourceState(ExecutionContext &context) const {
	return make_uniq<FusedAggregateLocalSourceState>(BufferManager::GetBufferManager(context.client));
}

//! Entry words: the key words, the row count, one int64 sum state per summed or averaged input column
static inline idx_t FusedKeyWords(const FusedIntegerAggregate &fused) {
	return fused.key_bytes > 8 ? 2 : 1;
}

static inline idx_t FusedEntryWords(const FusedIntegerAggregate &fused) {
	return FusedKeyWords(fused) + 1 + fused.sum_columns.size();
}

static void FusedAllocateTable(FusedAggregateLocalSourceState &lstate, idx_t entries, idx_t entry_words) {
	const idx_t bytes = entries * entry_words * sizeof(uint64_t);
	lstate.handle = lstate.buffer_manager.Allocate(MemoryTag::HASH_TABLE, bytes, true);
	lstate.table = reinterpret_cast<uint64_t *>(lstate.handle.Ptr());
	memset(lstate.table, 0, bytes);
	lstate.buffer_entries = entries;
}

static idx_t FusedTableBits(idx_t capacity) {
	idx_t bits = 0;
	while ((idx_t(1) << bits) < capacity) {
		bits++;
	}
	return bits;
}

static inline idx_t FusedSlotOf(hash_t hash, idx_t table_bits) {
	return (hash << FusedIntegerAggregate::PARTITION_BITS) >> (64 - table_bits);
}

//! Doubles the table past its cap (more than 2^20 groups in one of 4096 partitions: an adversarial hash
//! distribution)
static void FusedGrowTable(const FusedIntegerAggregate &fused, FusedAggregateLocalSourceState &lstate) {
	const auto shape = FusedGetKeyShape(fused.key_bytes);
	const auto words = FusedEntryWords(fused);
	const auto count_word = FusedKeyWords(fused);
	const auto old_capacity = lstate.capacity;
	BufferHandle old_handle = std::move(lstate.handle);
	auto old_table = lstate.table;
	lstate.capacity = old_capacity * 2;
	lstate.table_bits = FusedTableBits(lstate.capacity);
	FusedAllocateTable(lstate, lstate.capacity, words);
	const auto mask = lstate.capacity - 1;
	for (idx_t slot = 0; slot < old_capacity; slot++) {
		auto entry = old_table + slot * words;
		if (entry[count_word] == 0) {
			continue;
		}
		const auto hash = FusedHashKey(entry[0], shape.two_words ? entry[1] : 0, shape.two_words);
		auto target = FusedSlotOf(hash, lstate.table_bits);
		while (lstate.table[target * words + count_word] != 0) {
			target = (target + 1) & mask;
		}
		memcpy(lstate.table + target * words, entry, words * sizeof(uint64_t));
	}
}

//! Builds partition p's table over every handed-over list, sized once from its row count
static void FusedBuildTable(const FusedIntegerAggregate &fused, FusedAggregateGlobalState &gstate,
                            FusedAggregateLocalSourceState &lstate, idx_t p) {
	const auto rows = gstate.partition_rows[p];
	const auto words = FusedEntryWords(fused);
	const auto count_word = FusedKeyWords(fused);
	const auto sums = fused.sum_columns.size();
	idx_t capacity = NextPowerOfTwo(MaxValue<idx_t>(FusedIntegerAggregate::TABLE_MINIMUM_CAPACITY, 2 * rows));
	capacity = MinValue<idx_t>(capacity, FusedIntegerAggregate::TABLE_MAXIMUM_CAPACITY);
	if (capacity > lstate.buffer_entries) {
		lstate.handle.Destroy();
		lstate.table = nullptr;
		FusedAllocateTable(lstate, capacity, words);
	}
	lstate.capacity = capacity;
	lstate.table_bits = FusedTableBits(capacity);
	lstate.occupancy = 0;
	const auto shape = FusedGetKeyShape(fused.key_bytes);
	const auto row_width = fused.row_width;
	// the run kind folds each row's run length (a new entry starts at it, a hit adds it); the ordinary kind 1 / +1; so
	// does a last-key-folded row's count
	const bool run_kind = fused.run_kind || fused.last_key_fold;
	const auto run_length_offset = fused.run_length_offset;
	FusedRowColumn sum_input[FusedIntegerAggregate::MAXIMUM_AGGREGATES];
	for (idx_t s = 0; s < sums; s++) {
		sum_input[s] = fused.columns[fused.sum_columns[s]];
	}
	uint64_t key0[FusedIntegerAggregate::TABLE_BATCH];
	uint64_t key1[FusedIntegerAggregate::TABLE_BATCH];
	hash_t hash[FusedIntegerAggregate::TABLE_BATCH];
	for (auto &lists : gstate.handed_over) {
		for (auto chunk = lists->heads[p]; chunk; chunk = FusedChunkNext(chunk)) {
			const auto chunk_rows = FusedChunkRows(chunk);
			const auto base = chunk + FusedIntegerAggregate::CHUNK_HEADER_BYTES;
			for (idx_t batch_start = 0; batch_start < chunk_rows; batch_start += FusedIntegerAggregate::TABLE_BATCH) {
				const auto batch = MinValue<idx_t>(FusedIntegerAggregate::TABLE_BATCH, chunk_rows - batch_start);
				const auto batch_rows = base + batch_start * row_width;
				for (idx_t j = 0; j < batch; j++) {
					FusedLoadKey(batch_rows + j * row_width, shape, key0[j], key1[j]);
					hash[j] = FusedHashKey(key0[j], key1[j], shape.two_words);
					FUSED_PREFETCH_WRITE(lstate.table + FusedSlotOf(hash[j], lstate.table_bits) * words);
				}
				for (idx_t j = 0; j < batch; j++) {
					const auto row = batch_rows + j * row_width;
					const auto mask = lstate.capacity - 1;
					auto slot = FusedSlotOf(hash[j], lstate.table_bits);
					while (true) {
						auto entry = lstate.table + slot * words;
						if (entry[count_word] == 0) {
							entry[0] = key0[j];
							if (shape.two_words) {
								entry[1] = key1[j];
							}
							entry[count_word] = run_kind ? uint64_t(Load<uint32_t>(row + run_length_offset)) : 1;
							for (idx_t s = 0; s < sums; s++) {
								entry[count_word + 1 + s] = uint64_t(FusedLoadInput(row, sum_input[s]));
							}
							if (++lstate.occupancy * 2 > lstate.capacity) {
								FusedGrowTable(fused, lstate);
							}
							break;
						}
						if (entry[0] == key0[j] && (!shape.two_words || entry[1] == key1[j])) {
							entry[count_word] += run_kind ? uint64_t(Load<uint32_t>(row + run_length_offset)) : 1;
							for (idx_t s = 0; s < sums; s++) {
								entry[count_word + 1 + s] += uint64_t(FusedLoadInput(row, sum_input[s]));
							}
							break;
						}
						slot = (slot + 1) & mask;
					}
				}
			}
		}
	}
}

//! For the non-DISTINCT class without the chained build and for the DISTINCT class's (g, x) build: FusedBuildTable
//! reading each row's stored bits instead of recomputing its hash
//! (FusedGrowTable still rehashes its entries, which carry no hash, onto the same slot bits); the probe iterations are
//! counted per partition (table_probes)
static FUSED_NOINLINE void FusedBuildTableStored(const FusedIntegerAggregate &fused, FusedAggregateGlobalState &gstate,
                                                 FusedAggregateLocalSourceState &lstate, idx_t p) {
	const auto rows = gstate.partition_rows[p];
	const auto words = FusedEntryWords(fused);
	const auto count_word = FusedKeyWords(fused);
	const auto sums = fused.sum_columns.size();
	idx_t capacity = NextPowerOfTwo(MaxValue<idx_t>(FusedIntegerAggregate::TABLE_MINIMUM_CAPACITY, 2 * rows));
	capacity = MinValue<idx_t>(capacity, FusedIntegerAggregate::TABLE_MAXIMUM_CAPACITY);
	if (capacity > lstate.buffer_entries) {
		lstate.handle.Destroy();
		lstate.table = nullptr;
		FusedAllocateTable(lstate, capacity, words);
	}
	lstate.capacity = capacity;
	lstate.table_bits = FusedTableBits(capacity);
	lstate.occupancy = 0;
	const auto shape = FusedGetKeyShape(fused.key_bytes);
	const auto row_width = fused.row_width;
	const auto hash_offset = fused.hash_offset;
	// a last-key-folded row's count (a new entry starts at it, a hit adds it)
	const bool counted = fused.last_key_fold;
	const auto run_length_offset = fused.run_length_offset;
	FusedRowColumn sum_input[FusedIntegerAggregate::MAXIMUM_AGGREGATES];
	for (idx_t s = 0; s < sums; s++) {
		sum_input[s] = fused.columns[fused.sum_columns[s]];
	}
	uint64_t key0[FusedIntegerAggregate::TABLE_BATCH];
	uint64_t key1[FusedIntegerAggregate::TABLE_BATCH];
	hash_t hash[FusedIntegerAggregate::TABLE_BATCH];
	idx_t probes = 0;
	for (auto &lists : gstate.handed_over) {
		for (auto chunk = lists->heads[p]; chunk; chunk = FusedChunkNext(chunk)) {
			const auto chunk_rows = FusedChunkRows(chunk);
			const auto base = chunk + FusedIntegerAggregate::CHUNK_HEADER_BYTES;
			for (idx_t batch_start = 0; batch_start < chunk_rows; batch_start += FusedIntegerAggregate::TABLE_BATCH) {
				const auto batch = MinValue<idx_t>(FusedIntegerAggregate::TABLE_BATCH, chunk_rows - batch_start);
				const auto batch_rows = base + batch_start * row_width;
				for (idx_t j = 0; j < batch; j++) {
					const auto row = batch_rows + j * row_width;
					FusedLoadKey(row, shape, key0[j], key1[j]);
					hash[j] = hash_t(Load<uint32_t>(row + hash_offset)) << FUSED_STORED_HASH_SHIFT;
					FUSED_PREFETCH_WRITE(lstate.table + FusedSlotOf(hash[j], lstate.table_bits) * words);
				}
				for (idx_t j = 0; j < batch; j++) {
					const auto row = batch_rows + j * row_width;
					const auto mask = lstate.capacity - 1;
					auto slot = FusedSlotOf(hash[j], lstate.table_bits);
					while (true) {
						probes++;
						auto entry = lstate.table + slot * words;
						if (entry[count_word] == 0) {
							entry[0] = key0[j];
							if (shape.two_words) {
								entry[1] = key1[j];
							}
							entry[count_word] = counted ? uint64_t(Load<uint32_t>(row + run_length_offset)) : 1;
							for (idx_t s = 0; s < sums; s++) {
								entry[count_word + 1 + s] = uint64_t(FusedLoadInput(row, sum_input[s]));
							}
							if (++lstate.occupancy * 2 > lstate.capacity) {
								FusedGrowTable(fused, lstate);
							}
							break;
						}
						if (entry[0] == key0[j] && (!shape.two_words || entry[1] == key1[j])) {
							entry[count_word] += counted ? uint64_t(Load<uint32_t>(row + run_length_offset)) : 1;
							for (idx_t s = 0; s < sums; s++) {
								entry[count_word + 1 + s] += uint64_t(FusedLoadInput(row, sum_input[s]));
							}
							break;
						}
						slot = (slot + 1) & mask;
					}
				}
			}
		}
	}
	gstate.table_probes += probes;
}

template <class T>
static void FusedEmitKey(Vector &vector, uint64_t *const *entries, idx_t count, idx_t offset) {
	auto target = FlatVector::GetDataUnsafe<T>(vector);
	for (idx_t i = 0; i < count; i++) {
		// the key words hold the row's key bytes (little-endian), so the key is read back at its own offset
		T value;
		memcpy(&value, reinterpret_cast<const_data_ptr_t>(entries[i]) + offset, sizeof(T));
		target[i] = value;
	}
}

//! Writes one output chunk from occupied entries: the keys, then count as BIGINT, sum as HUGEINT from its int64 state,
//! avg as double(sum) / double(count) (IntegerAverageOperation::Finalize with no bind data)
static void FusedEmit(const FusedIntegerAggregate &fused, DataChunk &chunk, uint64_t *const *entries, idx_t count) {
	const auto count_word = FusedKeyWords(fused);
	for (idx_t key_idx = 0; key_idx < fused.key_count; key_idx++) {
		auto &column = fused.columns[key_idx];
		auto &vector = chunk.data[key_idx];
		switch (column.width) {
		case 1:
			FusedEmitKey<uint8_t>(vector, entries, count, column.offset);
			break;
		case 2:
			FusedEmitKey<uint16_t>(vector, entries, count, column.offset);
			break;
		case 4:
			FusedEmitKey<uint32_t>(vector, entries, count, column.offset);
			break;
		default:
			FusedEmitKey<uint64_t>(vector, entries, count, column.offset);
			break;
		}
	}
	for (idx_t aggr_idx = 0; aggr_idx < fused.outputs.size(); aggr_idx++) {
		auto &output = fused.outputs[aggr_idx];
		auto &vector = chunk.data[fused.key_count + aggr_idx];
		const auto sum_word = count_word + 1 + output.sum_index;
		switch (output.kind) {
		case FusedAggregateKind::COUNT: {
			auto target = FlatVector::GetData<int64_t>(vector);
			for (idx_t i = 0; i < count; i++) {
				target[i] = int64_t(entries[i][count_word]);
			}
			break;
		}
		case FusedAggregateKind::SUM: {
			auto target = FlatVector::GetData<hugeint_t>(vector);
			for (idx_t i = 0; i < count; i++) {
				target[i] = Hugeint::Convert(int64_t(entries[i][sum_word]));
			}
			break;
		}
		case FusedAggregateKind::AVG: {
			auto target = FlatVector::GetData<double>(vector);
			for (idx_t i = 0; i < count; i++) {
				target[i] = double(int64_t(entries[i][sum_word])) / double(entries[i][count_word]);
			}
			break;
		}
		case FusedAggregateKind::DISTINCT_COUNT:
			throw InternalException("Fused integer aggregate: a distinct count outside the DISTINCT class");
		}
	}
	chunk.SetCardinality(count);
}

//===--------------------------------------------------------------------===//
// The in-place chained partition table
//===--------------------------------------------------------------------===//
// Per partition: a directory of 4-byte group ordinals (0 = empty) at the open-addressing table's initial capacity for
// the partition (min(next_pow2(max(1024, 2 x rows)), 2^21) buckets, sized once from the exact row count and never
// grown), and an append-only group array - per group its group row's address (the partition row that introduced the
// group, whose key words are the group's keys: never copied), its stored bits (the tag) above the ordinal of the next
// group in its bucket, its count and one int64 sum state per summed or averaged input column. The array holds at most
// half the directory's buckets and doubles only past that (more than 2^20 groups in one partition, where the
// open-addressing table doubles), so its bytes and the directory's never pass that table's at the same row and group
// count.
static inline idx_t FusedChainGroupWords(const FusedIntegerAggregate &fused) {
	return 3 + fused.sum_columns.size();
}

//! (Re)allocates the group array with `entries` groups, keeping its first `keep` groups (the directory and the links
//! hold ordinals, never addresses, so a moved array stays linked)
static void FusedChainAllocateGroups(FusedAggregateLocalSourceState &lstate, idx_t entries, idx_t words, idx_t keep) {
	auto handle = lstate.buffer_manager.Allocate(MemoryTag::HASH_TABLE, entries * words * sizeof(uint64_t), true);
	auto groups = reinterpret_cast<uint64_t *>(handle.Ptr());
	if (keep > 0) {
		memcpy(groups, lstate.chain_groups, keep * words * sizeof(uint64_t));
	}
	lstate.chain_groups_handle = std::move(handle);
	lstate.chain_groups = groups;
	lstate.chain_group_entries = entries;
}

static inline const_data_ptr_t FusedChainGroupRow(const uint64_t *group) {
	return reinterpret_cast<const_data_ptr_t>(static_cast<uintptr_t>(group[0]));
}

static inline bool FusedChainSameKey(const_data_ptr_t row, const_data_ptr_t group_row, const FusedKeyShape &shape) {
	uint64_t key0, key1, group0, group1;
	FusedLoadKey(row, shape, key0, key1);
	FusedLoadKey(group_row, shape, group0, group1);
	return key0 == group0 && key1 == group1;
}

//! Links partition p's rows: the bucket from each row's stored bits, its chain walked comparing the stored bits and then
//! the key words (read from the group row), a new key appended as a group at the head of its bucket (chain_new), a
//! duplicate's count and sums added to its group (chain_dup); no open-addressing probe, no key copy, no rehash. The
//! group count fits the 4-byte ordinals: a partition's rows are bounded by the drain threshold's bytes.
static FUSED_NOINLINE void FusedChainBuild(const FusedIntegerAggregate &fused, FusedAggregateGlobalState &gstate,
                                           FusedAggregateLocalSourceState &lstate, idx_t p) {
	const auto rows = gstate.partition_rows[p];
	const auto words = FusedChainGroupWords(fused);
	const auto sums = fused.sum_columns.size();
	idx_t buckets = NextPowerOfTwo(MaxValue<idx_t>(FusedIntegerAggregate::TABLE_MINIMUM_CAPACITY, 2 * rows));
	buckets = MinValue<idx_t>(buckets, FusedIntegerAggregate::TABLE_MAXIMUM_CAPACITY);
	if (buckets > lstate.chain_directory_entries) {
		lstate.chain_directory_handle.Destroy();
		lstate.chain_directory = nullptr;
		lstate.chain_directory_handle =
		    lstate.buffer_manager.Allocate(MemoryTag::HASH_TABLE, buckets * sizeof(uint32_t), true);
		lstate.chain_directory = reinterpret_cast<uint32_t *>(lstate.chain_directory_handle.Ptr());
		lstate.chain_directory_entries = buckets;
	}
	const auto directory = lstate.chain_directory;
	memset(directory, 0, buckets * sizeof(uint32_t));
	const auto bucket_shift = 32 - FusedTableBits(buckets);
	const idx_t limit = MaxValue<idx_t>(1, MinValue<idx_t>(rows, buckets / 2));
	if (limit > lstate.chain_group_entries) {
		lstate.chain_groups_handle.Destroy();
		lstate.chain_groups = nullptr;
		FusedChainAllocateGroups(lstate, limit, words, 0);
	}
	const auto shape = FusedGetKeyShape(fused.key_bytes);
	const auto row_width = fused.row_width;
	const auto hash_offset = fused.hash_offset;
	// a last-key-folded row's count (a new group starts at it, a duplicate adds it)
	const bool counted = fused.last_key_fold;
	const auto run_length_offset = fused.run_length_offset;
	FusedRowColumn sum_input[FusedIntegerAggregate::MAXIMUM_AGGREGATES];
	for (idx_t s = 0; s < sums; s++) {
		sum_input[s] = fused.columns[fused.sum_columns[s]];
	}
	auto groups = lstate.chain_groups;
	idx_t group_count = 0;
	for (auto &lists : gstate.handed_over) {
		for (auto chunk = lists->heads[p]; chunk; chunk = FusedChunkNext(chunk)) {
			const auto chunk_rows = FusedChunkRows(chunk);
			auto row = chunk + FusedIntegerAggregate::CHUNK_HEADER_BYTES;
			for (idx_t i = 0; i < chunk_rows; i++, row += row_width) {
				const auto stored = Load<uint32_t>(row + hash_offset);
				const auto bucket = stored >> bucket_shift;
				const uint32_t head = directory[bucket];
				uint64_t *group = nullptr;
				for (auto ordinal = head; ordinal != 0;) {
					auto candidate = groups + (ordinal - 1) * words;
					const auto link = candidate[1];
					// the stored bits first, then the key words of both rows (read only on a tag match)
					if (uint32_t(link >> 32) == stored && FusedChainSameKey(row, FusedChainGroupRow(candidate), shape)) {
						group = candidate;
						break;
					}
					ordinal = uint32_t(link);
				}
				if (group) {
					group[2] += counted ? uint64_t(Load<uint32_t>(row + run_length_offset)) : 1;
					for (idx_t s = 0; s < sums; s++) {
						group[3 + s] += uint64_t(FusedLoadInput(row, sum_input[s]));
					}
					continue;
				}
				if (group_count == lstate.chain_group_entries) {
					FusedChainAllocateGroups(lstate, lstate.chain_group_entries * 2, words, group_count);
					groups = lstate.chain_groups;
				}
				// a new key: its row becomes the group row, linked at the head of its bucket
				group = groups + group_count * words;
				group[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(row));
				group[1] = (uint64_t(stored) << 32) | head;
				group[2] = counted ? uint64_t(Load<uint32_t>(row + run_length_offset)) : 1;
				for (idx_t s = 0; s < sums; s++) {
					group[3 + s] = uint64_t(FusedLoadInput(row, sum_input[s]));
				}
				directory[bucket] = uint32_t(++group_count);
			}
		}
	}
	lstate.chain_group_count = group_count;
	gstate.chain_rows += rows;
	gstate.chain_new += group_count;
}

template <class T>
static void FusedEmitRowKey(Vector &vector, const const_data_ptr_t *group_rows, idx_t count, idx_t offset) {
	auto target = FlatVector::GetDataUnsafe<T>(vector);
	for (idx_t i = 0; i < count; i++) {
		T value;
		memcpy(&value, group_rows[i] + offset, sizeof(T));
		target[i] = value;
	}
}

//! One output chunk from the chained build's groups: the keys read from each group row at their row offsets, then the
//! aggregates as FusedEmit writes them (count as BIGINT, sum as HUGEINT from its int64 state, avg as double(sum) /
//! double(count))
static void FusedChainEmit(const FusedIntegerAggregate &fused, DataChunk &chunk, const const_data_ptr_t *group_rows,
                           uint64_t *const *groups, idx_t count) {
	for (idx_t key_idx = 0; key_idx < fused.key_count; key_idx++) {
		auto &column = fused.columns[key_idx];
		auto &vector = chunk.data[key_idx];
		switch (column.width) {
		case 1:
			FusedEmitRowKey<uint8_t>(vector, group_rows, count, column.offset);
			break;
		case 2:
			FusedEmitRowKey<uint16_t>(vector, group_rows, count, column.offset);
			break;
		case 4:
			FusedEmitRowKey<uint32_t>(vector, group_rows, count, column.offset);
			break;
		default:
			FusedEmitRowKey<uint64_t>(vector, group_rows, count, column.offset);
			break;
		}
	}
	for (idx_t aggr_idx = 0; aggr_idx < fused.outputs.size(); aggr_idx++) {
		auto &output = fused.outputs[aggr_idx];
		auto &vector = chunk.data[fused.key_count + aggr_idx];
		const auto sum_word = 3 + output.sum_index;
		switch (output.kind) {
		case FusedAggregateKind::COUNT: {
			auto target = FlatVector::GetData<int64_t>(vector);
			for (idx_t i = 0; i < count; i++) {
				target[i] = int64_t(groups[i][2]);
			}
			break;
		}
		case FusedAggregateKind::SUM: {
			auto target = FlatVector::GetData<hugeint_t>(vector);
			for (idx_t i = 0; i < count; i++) {
				target[i] = Hugeint::Convert(int64_t(groups[i][sum_word]));
			}
			break;
		}
		case FusedAggregateKind::AVG: {
			auto target = FlatVector::GetData<double>(vector);
			for (idx_t i = 0; i < count; i++) {
				target[i] = double(int64_t(groups[i][sum_word])) / double(groups[i][2]);
			}
			break;
		}
		case FusedAggregateKind::DISTINCT_COUNT:
			throw InternalException("Fused integer aggregate: a distinct count in the chained partition build");
		}
	}
	chunk.SetCardinality(count);
}

//===--------------------------------------------------------------------===//
// Phase 2 of the DISTINCT class: each partition's (g, x) table folded into a per-task group table, the tables
// merged once by the last task
//===--------------------------------------------------------------------===//
static inline idx_t FusedGroupKeyWords(const FusedIntegerAggregate &fused) {
	return fused.group_bytes == 0 ? 0 : fused.group_bytes > 8 ? 2 : 1;
}

//! the group entry's sum states - one per summed or averaged companion input on a mixed shape, else one per sum
//! column (the four group-table sites read it: FusedGroupWords, FusedGroupAdd, the merges and FusedDistinctEmit's
//! sum_index, which indexes the same list)
static inline idx_t FusedSumStates(const FusedIntegerAggregate &fused) {
	return fused.mixed ? fused.companion_sums.size() : fused.sum_columns.size();
}

//! Group entry words: the g key words, distinct, count, one int64 sum state per summed or averaged input column
static inline idx_t FusedGroupWords(const FusedIntegerAggregate &fused) {
	return FusedGroupKeyWords(fused) + 2 + FusedSumStates(fused);
}

static void FusedAllocateGroupTable(BufferManager &buffer_manager, FusedGroupTable &table, idx_t capacity,
                                    idx_t words) {
	const idx_t bytes = capacity * words * sizeof(uint64_t);
	table.handle = buffer_manager.Allocate(MemoryTag::HASH_TABLE, bytes, true);
	table.entries = reinterpret_cast<uint64_t *>(table.handle.Ptr());
	memset(table.entries, 0, bytes);
	table.capacity = capacity;
	table.occupancy = 0;
}

static inline idx_t FusedGroupSlotOf(hash_t hash, idx_t capacity) {
	return hash >> (64 - FusedTableBits(capacity));
}

//! Finds or inserts group (g0, g1) and adds the deltas to its entry (distinct first, so an inserted entry is occupied
//! before the table may grow); a table half full doubles
static void FusedGroupAdd(const FusedIntegerAggregate &fused, BufferManager &buffer_manager, FusedGroupTable &table,
                          uint64_t g0, uint64_t g1, uint64_t distinct, uint64_t count, const uint64_t *sums) {
	const auto key_words = FusedGroupKeyWords(fused);
	const auto words = FusedGroupWords(fused);
	const auto sum_count = FusedSumStates(fused);
	uint64_t *entry;
	if (key_words == 0) {
		// the ungrouped form: one entry
		entry = table.entries;
		table.occupancy = 1;
	} else {
		const auto mask = table.capacity - 1;
		auto slot = FusedGroupSlotOf(FusedHashKey(g0, g1, key_words == 2), table.capacity);
		while (true) {
			entry = table.entries + slot * words;
			if (entry[key_words] == 0) {
				entry[0] = g0;
				if (key_words == 2) {
					entry[1] = g1;
				}
				table.occupancy++;
				break;
			}
			if (entry[0] == g0 && (key_words == 1 || entry[1] == g1)) {
				break;
			}
			slot = (slot + 1) & mask;
		}
	}
	entry[key_words] += distinct;
	entry[key_words + 1] += count;
	for (idx_t s = 0; s < sum_count; s++) {
		entry[key_words + 2 + s] += sums[s];
	}
	if (key_words == 0 || table.occupancy * 2 <= table.capacity) {
		return;
	}
	// grow: re-insert every occupied entry into a table twice the size
	FusedGroupTable grown;
	FusedAllocateGroupTable(buffer_manager, grown, table.capacity * 2, words);
	const auto mask = grown.capacity - 1;
	for (idx_t slot = 0; slot < table.capacity; slot++) {
		auto old_entry = table.entries + slot * words;
		if (old_entry[key_words] == 0) {
			continue;
		}
		auto target = FusedGroupSlotOf(FusedHashKey(old_entry[0], key_words == 2 ? old_entry[1] : 0, key_words == 2),
		                               grown.capacity);
		while (grown.entries[target * words + key_words] != 0) {
			target = (target + 1) & mask;
		}
		memcpy(grown.entries + target * words, old_entry, words * sizeof(uint64_t));
		grown.occupancy++;
	}
	table.handle = std::move(grown.handle);
	table.entries = grown.entries;
	table.capacity = grown.capacity;
	table.occupancy = grown.occupancy;
}

//! Folds the task's (g, x) table of one partition into its group table: each occupied entry is one distinct (g, x),
//! so its group gains distinct += 1 and the entry's count and sums; the entry's count word is cleared (the buffer is
//! all-zero again for the next build). On a mixed shape: distinct += 1 alone - the entry's count counts kept rows,
//! and its (g, x) entry carries no sum word, so the count and the FusedSumStates sums passed are zero (the companion
//! brings them at the merge)
static void FusedFoldPartition(const FusedIntegerAggregate &fused, FusedAggregateLocalSourceState &lstate) {
	const auto words = FusedEntryWords(fused);
	const auto count_word = FusedKeyWords(fused);
	const auto group_shape = FusedGetKeyShape(fused.group_bytes);
	const auto sum_count = fused.sum_columns.size();
	uint64_t sums[FusedIntegerAggregate::MAXIMUM_AGGREGATES];
	for (idx_t s = 0; s < FusedIntegerAggregate::MAXIMUM_AGGREGATES; s++) {
		sums[s] = 0;
	}
	auto &groups = *lstate.groups;
	for (idx_t slot = 0; slot < lstate.capacity; slot++) {
		auto entry = lstate.table + slot * words;
		if (entry[count_word] == 0) {
			continue;
		}
		// the entry's key words hold the row's key bytes, g first: g is read back from the entry's own bytes
		uint64_t g0, g1;
		FusedLoadKey(reinterpret_cast<const_data_ptr_t>(entry), group_shape, g0, g1);
		for (idx_t s = 0; s < sum_count; s++) {
			sums[s] = entry[count_word + 1 + s];
		}
		FusedGroupAdd(fused, lstate.buffer_manager, groups, g0, g1, 1, fused.mixed ? 0 : entry[count_word], sums);
		entry[count_word] = 0;
		lstate.distinct_entries++;
	}
}

//! Merges every handed-in table into the fullest one and returns it
static FusedGroupTable &FusedMergeGroupTables(const FusedIntegerAggregate &fused, BufferManager &buffer_manager,
                                              vector<unique_ptr<FusedGroupTable>> &tables) {
	const auto key_words = FusedGroupKeyWords(fused);
	const auto words = FusedGroupWords(fused);
	idx_t base_idx = 0;
	for (idx_t i = 1; i < tables.size(); i++) {
		if (tables[i]->occupancy > tables[base_idx]->occupancy) {
			base_idx = i;
		}
	}
	auto &base = *tables[base_idx];
	for (idx_t i = 0; i < tables.size(); i++) {
		if (i == base_idx) {
			continue;
		}
		auto &other = *tables[i];
		for (idx_t slot = 0; slot < other.capacity; slot++) {
			auto entry = other.entries + slot * words;
			if (entry[key_words] == 0) {
				continue;
			}
			const uint64_t g0 = key_words >= 1 ? entry[0] : 0;
			const uint64_t g1 = key_words == 2 ? entry[1] : 0;
			FusedGroupAdd(fused, buffer_manager, base, g0, g1, entry[key_words], entry[key_words + 1],
			              entry + key_words + 2);
		}
	}
	return base;
}

//! The companion merge's second half: each entry of the one combined companion is looked up in the final group table
//! `base` - find-only, never an insert, whose zero distinct word the growth and the scans would read as empty - and its
//! count and sums are added to the group; a companion group without its (g, x) group, or a (g, x) group without a
//! companion entry, is an InternalException (every row passed both sides). Returns the companion entries merged
//! (companion_groups)
static idx_t FusedMergeCombinedCompanion(const FusedIntegerAggregate &fused, optional_ptr<FusedGroupTable> combined,
                                         FusedGroupTable &base) {
	const auto key_words = FusedGroupKeyWords(fused);
	const auto words = FusedGroupWords(fused);
	const auto sum_count = FusedSumStates(fused);
	idx_t merged = 0;
	if (combined) {
		for (idx_t slot = 0; slot < combined->capacity; slot++) {
			auto entry = combined->entries + slot * words;
			if (entry[key_words] == 0) {
				continue;
			}
			uint64_t *target = nullptr;
			if (key_words == 0) {
				target = base.occupancy != 0 ? base.entries : nullptr;
			} else {
				const auto mask = base.capacity - 1;
				auto probe = FusedGroupSlotOf(FusedHashKey(entry[0], key_words == 2 ? entry[1] : 0, key_words == 2),
				                              base.capacity);
				while (base.entries[probe * words + key_words] != 0) {
					auto candidate = base.entries + probe * words;
					if (candidate[0] == entry[0] && (key_words == 1 || candidate[1] == entry[1])) {
						target = candidate;
						break;
					}
					probe = (probe + 1) & mask;
				}
			}
			if (!target) {
				throw InternalException("Fused integer aggregate: a companion group without its (g, x) group");
			}
			target[key_words + 1] += entry[key_words + 1];
			for (idx_t s = 0; s < sum_count; s++) {
				target[key_words + 2 + s] += entry[key_words + 2 + s];
			}
			merged++;
		}
	}
	if (merged != base.occupancy) {
		throw InternalException("Fused integer aggregate: %llu companion groups merged into %llu groups", merged, base.occupancy);
	}
	return merged;
}

//! Merges every thread's companion into the final group table `base` (the (g, x)-derived table the merge
//! above returned, passed explicitly - never a companion): the companions are first folded into one (their occupancy
//! words stay non-zero), then FusedMergeCombinedCompanion adds it to `base`. Returns the companion entries merged
//! (companion_groups)
static idx_t FusedMergeCompanions(const FusedIntegerAggregate &fused, BufferManager &buffer_manager,
                                  FusedAggregateGlobalState &gstate, FusedGroupTable &base) {
	const auto key_words = FusedGroupKeyWords(fused);
	const auto words = FusedGroupWords(fused);
	unique_ptr<FusedGroupTable> combined;
	for (auto &lists : gstate.handed_over) {
		if (!lists->companion || lists->companion->occupancy == 0) {
			continue;
		}
		if (!combined) {
			combined = std::move(lists->companion);
			continue;
		}
		auto &other = *lists->companion;
		for (idx_t slot = 0; slot < other.capacity; slot++) {
			auto entry = other.entries + slot * words;
			if (entry[key_words] == 0) {
				continue;
			}
			const uint64_t g0 = key_words >= 1 ? entry[0] : 0;
			const uint64_t g1 = key_words == 2 ? entry[1] : 0;
			FusedGroupAdd(fused, buffer_manager, *combined, g0, g1, 1, entry[key_words + 1], entry + key_words + 2);
		}
		lists->companion.reset();
	}
	return FusedMergeCombinedCompanion(fused, combined.get(), base);
}

//! Merges the smaller of two group tables into the larger
//! (FusedMergeGroupTables over the pair) and returns the larger; the smaller is freed
static unique_ptr<FusedGroupTable> FusedMergeTwoGroupTables(const FusedIntegerAggregate &fused,
                                                           BufferManager &buffer_manager, unique_ptr<FusedGroupTable> a,
                                                           unique_ptr<FusedGroupTable> b) {
	vector<unique_ptr<FusedGroupTable>> two;
	two.push_back(std::move(a));
	two.push_back(std::move(b));
	auto &base = FusedMergeGroupTables(fused, buffer_manager, two);
	return std::move(&base == two[0].get() ? two[0] : two[1]);
}

//! Folds the smaller of two companions into the larger as
//! FusedMergeCompanions folds each companion into the first (their occupancy words stay non-zero) and returns the
//! larger; the smaller is freed
static unique_ptr<FusedGroupTable> FusedCombineTwoCompanions(const FusedIntegerAggregate &fused,
                                                            BufferManager &buffer_manager,
                                                            unique_ptr<FusedGroupTable> a,
                                                            unique_ptr<FusedGroupTable> b) {
	if (b->occupancy > a->occupancy) {
		std::swap(a, b);
	}
	const auto key_words = FusedGroupKeyWords(fused);
	const auto words = FusedGroupWords(fused);
	auto &other = *b;
	for (idx_t slot = 0; slot < other.capacity; slot++) {
		auto entry = other.entries + slot * words;
		if (entry[key_words] == 0) {
			continue;
		}
		const uint64_t g0 = key_words >= 1 ? entry[0] : 0;
		const uint64_t g1 = key_words == 2 ? entry[1] : 0;
		FusedGroupAdd(fused, buffer_manager, *a, g0, g1, 1, entry[key_words + 1], entry + key_words + 2);
	}
	return a;
}

//! The grouped DISTINCT class's merge as a tree, on the tasks as they finish
//! instead of on the last one alone. After its builds a task claims the sink threads' companions one at a time (a
//! mixed shape) and folds them into one; then, under merge_lock, it takes the parked pair when there is one - merging
//! it into its own outside the lock, the smaller table into the larger, and looking again - else parks its own pair and
//! counts itself done. A pair is parked only when none is, so at most one is, and every task's tables are in the parked
//! pair or in a task still merging; the task that brings tasks_done up to tasks_started while merged is unset takes
//! the parked pair - every task's tables merged into it - into merged_tables[0] and `companion`, and `tables` the
//! tasks merged. Returns false for every other task
static bool FusedTreeMerge(const FusedIntegerAggregate &fused, FusedAggregateGlobalState &gstate,
                           FusedAggregateGlobalSourceState &source, FusedAggregateLocalSourceState &lstate,
                           unique_ptr<FusedGroupTable> &companion, idx_t &tables) {
	auto &buffer_manager = lstate.buffer_manager;
	if (fused.mixed) {
		while (true) {
			const idx_t i = source.next_companion++;
			if (i >= gstate.handed_over.size()) {
				break;
			}
			auto claimed = std::move(gstate.handed_over[i]->companion);
			if (!claimed || claimed->occupancy == 0) {
				continue;
			}
			companion = companion ? FusedCombineTwoCompanions(fused, buffer_manager, std::move(companion),
			                                                  std::move(claimed))
			                      : std::move(claimed);
		}
	}
	auto groups = std::move(lstate.groups);
	while (true) {
		unique_ptr<FusedGroupTable> other_groups;
		unique_ptr<FusedGroupTable> other_companion;
		{
			lock_guard<mutex> guard(source.merge_lock);
			source.distinct_entries += lstate.distinct_entries;
			lstate.distinct_entries = 0;
			if (!source.parked) {
				source.parked_groups = std::move(groups);
				source.parked_companion = std::move(companion);
				source.parked = true;
				source.tasks_done++;
				if (source.tasks_done != source.tasks_started || source.merged) {
					return false;
				}
				source.merged = true;
				source.parked = false;
				lstate.merged_tables.push_back(std::move(source.parked_groups));
				companion = std::move(source.parked_companion);
				tables = source.tasks_done;
				return true;
			}
			other_groups = std::move(source.parked_groups);
			other_companion = std::move(source.parked_companion);
			source.parked = false;
		}
		groups = FusedMergeTwoGroupTables(fused, buffer_manager, std::move(groups), std::move(other_groups));
		if (other_companion) {
			companion = companion ? FusedCombineTwoCompanions(fused, buffer_manager, std::move(companion),
			                                                  std::move(other_companion))
			                      : std::move(other_companion);
		}
	}
}

//! A VARCHAR key written from the group entries' gids as the map's strings (copied into the output vector)
static void FusedEmitGidKey(Vector &vector, uint64_t *const *entries, idx_t count, idx_t offset,
                            FusedAggregateGlobalState &gstate) {
	auto target = FlatVector::GetData<string_t>(vector);
	lock_guard<mutex> guard(gstate.gid_lock);
	for (idx_t i = 0; i < count; i++) {
		uint16_t gid;
		memcpy(&gid, reinterpret_cast<const_data_ptr_t>(entries[i]) + offset, sizeof(gid));
		target[i] = StringVector::AddString(vector, gstate.gid_strings[gid]);
	}
}

//! Writes one output chunk from group entries: the keys (read back from the g words at their row offsets; a
//! VARCHAR key as its gid's string), then each aggregate in the operator's order: count(DISTINCT) and count as BIGINT,
//! sum as HUGEINT from its int64 state, avg as double(sum) / double(count); an empty ungrouped input gives count 0 and
//! NULL sum and avg, as DuckDB's
static void FusedDistinctEmit(const FusedIntegerAggregate &fused, FusedAggregateGlobalState &gstate, DataChunk &chunk,
                              uint64_t *const *entries, idx_t count) {
	const auto key_words = FusedGroupKeyWords(fused);
	for (idx_t key_idx = 0; key_idx < fused.key_count; key_idx++) {
		auto &column = fused.columns[key_idx];
		auto &vector = chunk.data[key_idx];
		if (column.gid) {
			FusedEmitGidKey(vector, entries, count, column.offset, gstate);
			continue;
		}
		switch (column.width) {
		case 1:
			FusedEmitKey<uint8_t>(vector, entries, count, column.offset);
			break;
		case 2:
			FusedEmitKey<uint16_t>(vector, entries, count, column.offset);
			break;
		case 4:
			FusedEmitKey<uint32_t>(vector, entries, count, column.offset);
			break;
		default:
			FusedEmitKey<uint64_t>(vector, entries, count, column.offset);
			break;
		}
	}
	const auto distinct_word = key_words;
	const auto count_word = key_words + 1;
	for (idx_t aggr_idx = 0; aggr_idx < fused.outputs.size(); aggr_idx++) {
		auto &output = fused.outputs[aggr_idx];
		auto &vector = chunk.data[fused.key_count + aggr_idx];
		const auto sum_word = key_words + 2 + output.sum_index;
		switch (output.kind) {
		case FusedAggregateKind::DISTINCT_COUNT: {
			auto target = FlatVector::GetData<int64_t>(vector);
			for (idx_t i = 0; i < count; i++) {
				target[i] = int64_t(entries[i][distinct_word]);
			}
			break;
		}
		case FusedAggregateKind::COUNT: {
			auto target = FlatVector::GetData<int64_t>(vector);
			for (idx_t i = 0; i < count; i++) {
				target[i] = int64_t(entries[i][count_word]);
			}
			break;
		}
		case FusedAggregateKind::SUM: {
			auto target = FlatVector::GetData<hugeint_t>(vector);
			for (idx_t i = 0; i < count; i++) {
				if (entries[i][count_word] == 0) {
					FlatVector::SetNull(vector, i, true);
					continue;
				}
				target[i] = Hugeint::Convert(int64_t(entries[i][sum_word]));
			}
			break;
		}
		case FusedAggregateKind::AVG: {
			auto target = FlatVector::GetData<double>(vector);
			for (idx_t i = 0; i < count; i++) {
				if (entries[i][count_word] == 0) {
					FlatVector::SetNull(vector, i, true);
					continue;
				}
				target[i] = double(int64_t(entries[i][sum_word])) / double(entries[i][count_word]);
			}
			break;
		}
		}
	}
	chunk.SetCardinality(count);
}

static SourceResultType FusedDistinctGetData(const FusedIntegerAggregate &fused, DataChunk &chunk,
                                             FusedAggregateGlobalState &gstate, OperatorSourceInput &input) {
	auto &source = input.global_state.Cast<FusedAggregateGlobalSourceState>();
	auto &lstate = input.local_state.Cast<FusedAggregateLocalSourceState>();
	const auto key_words = FusedGroupKeyWords(fused);
	const auto words = FusedGroupWords(fused);
	if (!lstate.emit_table) {
		if (!lstate.counted) {
			// count this task before its first fetch
			{
				lock_guard<mutex> guard(source.merge_lock);
				source.tasks_started++;
			}
			lstate.counted = true;
			lstate.groups = make_uniq<FusedGroupTable>();
			FusedAllocateGroupTable(lstate.buffer_manager, *lstate.groups,
			                        key_words == 0 ? 1 : FusedIntegerAggregate::TABLE_MINIMUM_CAPACITY, words);
		}
		// build every partition this task claims and fold it into the task's group table
		while (true) {
			idx_t p;
			do {
				p = source.next_partition++;
			} while (p < FusedIntegerAggregate::PARTITION_COUNT && gstate.partition_rows[p] == 0);
			if (p >= FusedIntegerAggregate::PARTITION_COUNT) {
				break;
			}
			source.claimed++;
			if (fused.hash_stored) {
				FusedBuildTableStored(fused, gstate, lstate, p);
			} else {
				FusedBuildTable(fused, gstate, lstate, p);
			}
			FusedFoldPartition(fused, lstate);
		}
		optional_ptr<FusedGroupTable> merged_table;
		idx_t companion_groups = 0;
		idx_t tables_merged = 0;
		if (key_words > 0 && FusedDistinctTreeMergeEnabled()) {
			unique_ptr<FusedGroupTable> companion;
			if (!FusedTreeMerge(fused, gstate, source, lstate, companion, tables_merged)) {
				return SourceResultType::FINISHED;
			}
			merged_table = lstate.merged_tables[0].get();
			companion_groups = fused.mixed ? FusedMergeCombinedCompanion(fused, companion.get(), *merged_table) : 0;
		} else {
			// hand the group table in; the task that completes the count merges, once
			{
				lock_guard<mutex> guard(source.merge_lock);
				source.handed_in.push_back(std::move(lstate.groups));
				source.distinct_entries += lstate.distinct_entries;
				source.tasks_done++;
				if (source.tasks_done != source.tasks_started || source.merged) {
					return SourceResultType::FINISHED;
				}
				source.merged = true;
				lstate.merged_tables = std::move(source.handed_in);
				source.handed_in.clear();
			}
			merged_table = &FusedMergeGroupTables(fused, lstate.buffer_manager, lstate.merged_tables);
			companion_groups =
			    fused.mixed ? FusedMergeCompanions(fused, lstate.buffer_manager, gstate, *merged_table) : 0;
			tables_merged = lstate.merged_tables.size();
		}
		auto &merged = *merged_table;
		{
			lock_guard<mutex> guard(source.merge_lock);
			source.groups = merged.occupancy;
			source.tasks_merged = tables_merged;
			source.companion_groups = companion_groups;
			source.merge_done = true;
		}
		lstate.emit_table = &merged;
		lstate.scan_position = 0;
	}
	auto &table = *lstate.emit_table;
	if (key_words == 0) {
		// the ungrouped form: one row, emitted once
		if (lstate.emit_done) {
			return SourceResultType::FINISHED;
		}
		lstate.emit_done = true;
		lstate.emit[0] = table.entries;
		FusedDistinctEmit(fused, gstate, chunk, lstate.emit, 1);
		return SourceResultType::FINISHED;
	}
	idx_t count = 0;
	while (lstate.scan_position < table.capacity && count < STANDARD_VECTOR_SIZE) {
		auto entry = table.entries + lstate.scan_position * words;
		lstate.scan_position++;
		if (entry[key_words] != 0) {
			lstate.emit[count++] = entry;
		}
	}
	FusedDistinctEmit(fused, gstate, chunk, lstate.emit, count);
	return count == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

//! The chained build's source: claims the next nonempty partition and links it (FusedChainBuild), then scans its
//! groups in place - one visit per group, no empty slot and no clear
static SourceResultType FusedChainScan(const FusedIntegerAggregate &fused, DataChunk &chunk,
                                       FusedAggregateGlobalState &gstate, OperatorSourceInput &input) {
	auto &source = input.global_state.Cast<FusedAggregateGlobalSourceState>();
	auto &lstate = input.local_state.Cast<FusedAggregateLocalSourceState>();
	const auto words = FusedChainGroupWords(fused);
	idx_t count = 0;
	while (count < STANDARD_VECTOR_SIZE) {
		if (!lstate.active) {
			if (count > 0) {
				// the chunk's groups live in the group array, which the next build reuses: emit them first
				break;
			}
			idx_t p;
			do {
				p = source.next_partition++;
			} while (p < FusedIntegerAggregate::PARTITION_COUNT && gstate.partition_rows[p] == 0);
			if (p >= FusedIntegerAggregate::PARTITION_COUNT) {
				break;
			}
			source.claimed++;
			FusedChainBuild(fused, gstate, lstate, p);
			lstate.active = true;
			lstate.scan_position = 0;
		}
		while (lstate.scan_position < lstate.chain_group_count && count < STANDARD_VECTOR_SIZE) {
			auto group = lstate.chain_groups + lstate.scan_position * words;
			lstate.scan_position++;
			lstate.emit[count] = group;
			lstate.emit_rows[count] = FusedChainGroupRow(group);
			count++;
		}
		if (lstate.scan_position == lstate.chain_group_count) {
			lstate.active = false;
		}
	}
	FusedChainEmit(fused, chunk, lstate.emit_rows, lstate.emit, count);
	return count == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// The bitmap class, phase 2
//===--------------------------------------------------------------------===//
static inline idx_t FusedPopCount(uint64_t word) {
#if defined(__GNUC__) || defined(__clang__)
	return idx_t(__builtin_popcountll(word));
#else
	idx_t result = 0;
	for (; word; word &= word - 1) {
		result++;
	}
	return result;
#endif
}

//! Each task ORs the threads' bitmaps over the stripes it claims and counts the bits, code 0 (the NULL slot) excluded;
//! the task that brings tasks_done up to tasks_started emits the one row, the bits counted plus the distinct overflow
//! strings (the DISTINCT class's merge rule: every task counts itself before its first claim, so every claimed stripe
//! is counted by a task that is done by then, and the row is emitted once)
static SourceResultType FusedBitmapGetData(const FusedIntegerAggregate &fused, DataChunk &chunk,
                                           FusedAggregateGlobalState &gstate, OperatorSourceInput &input) {
	auto &source = input.global_state.Cast<FusedAggregateGlobalSourceState>();
	auto &lstate = input.local_state.Cast<FusedAggregateLocalSourceState>();
	if (lstate.emit_done) {
		return SourceResultType::FINISHED;
	}
	lstate.emit_done = true;
	{
		lock_guard<mutex> guard(source.merge_lock);
		source.tasks_started++;
	}
	const idx_t stripes = gstate.partitions_nonempty.load();
	vector<const uint64_t *> sources;
	for (auto &handle : gstate.bitmaps) {
		sources.push_back(reinterpret_cast<const uint64_t *>(handle.Ptr()));
	}
	uint64_t acc[FusedIntegerAggregate::BITMAP_STRIPE_WORDS];
	idx_t bits = 0;
	while (true) {
		const idx_t stripe = source.next_partition++;
		if (stripe >= stripes) {
			break;
		}
		source.claimed++;
		const idx_t begin = stripe * FusedIntegerAggregate::BITMAP_STRIPE_WORDS;
		const idx_t words = MinValue<idx_t>(FusedIntegerAggregate::BITMAP_STRIPE_WORDS, fused.bitmap_words - begin);
		memcpy(acc, sources[0] + begin, words * sizeof(uint64_t));
		for (idx_t t = 1; t < sources.size(); t++) {
			auto src = sources[t] + begin;
			for (idx_t w = 0; w < words; w++) {
				acc[w] |= src[w];
			}
		}
		if (begin == 0) {
			acc[0] &= ~uint64_t(1);
		}
		if (begin + words == fused.bitmap_words && (fused.bitmap_dict->count & 63) != 0) {
			// the last word's bits at or beyond the code count (the marker's padding_bits; every code is
			// below the count, so none is ever set)
			gstate.bitmap_padding_bits += FusedPopCount(acc[words - 1] >> (fused.bitmap_dict->count & 63));
		}
		for (idx_t w = 0; w < words; w++) {
			bits += FusedPopCount(acc[w]);
		}
	}
	idx_t total;
	{
		lock_guard<mutex> guard(source.merge_lock);
		source.distinct_entries += bits;
		source.tasks_done++;
		if (source.tasks_done != source.tasks_started || source.merged) {
			return SourceResultType::FINISHED;
		}
		source.merged = true;
		source.merge_done = true;
		total = source.distinct_entries;
	}
	total += gstate.bitmap_overflow.size();
	chunk.SetCardinality(1);
	FlatVector::GetData<int64_t>(chunk.data[0])[0] = NumericCast<int64_t>(total);
	return SourceResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// The set member of the DISTINCT class, phase 2
//===--------------------------------------------------------------------===//
//! Doubles the task's set past half load (more than 2^20 distinct x in one of 4096 partitions), re-inserting every key
static void FusedGrowSet(FusedAggregateLocalSourceState &lstate) {
	const auto old_capacity = lstate.capacity;
	BufferHandle old_handle = std::move(lstate.handle);
	auto old_table = lstate.table;
	lstate.capacity = old_capacity * 2;
	lstate.table_bits = FusedTableBits(lstate.capacity);
	FusedAllocateTable(lstate, lstate.capacity, 1);
	const auto mask = lstate.capacity - 1;
	for (idx_t slot = 0; slot < old_capacity; slot++) {
		const auto key = old_table[slot];
		if (key == 0) {
			continue;
		}
		auto target = FusedSlotOf(FusedHashKey(key, 0, false), lstate.table_bits);
		while (lstate.table[target] != 0) {
			target = (target + 1) & mask;
		}
		lstate.table[target] = key;
	}
}

//! Counts partition p's distinct x: every row of every handed-over list is inserted into an open-addressing set of its
//! key word alone (x's bytes, read in place), sized once at next_pow2(2 x rows) in [TABLE_MINIMUM_CAPACITY,
//! TABLE_MAXIMUM_CAPACITY] and doubled past half load. The key word 0 marks an empty slot, so x = 0 is counted by a flag.
//! A new key counts at its insert, so the set is never scanned; it is cleared (all-zero) for the next partition
static FUSED_NOINLINE idx_t FusedBuildSet(const FusedIntegerAggregate &fused, FusedAggregateGlobalState &gstate,
                                          FusedAggregateLocalSourceState &lstate, idx_t p) {
	const auto rows = gstate.partition_rows[p];
	idx_t capacity = NextPowerOfTwo(MaxValue<idx_t>(FusedIntegerAggregate::TABLE_MINIMUM_CAPACITY, 2 * rows));
	capacity = MinValue<idx_t>(capacity, FusedIntegerAggregate::TABLE_MAXIMUM_CAPACITY);
	if (capacity > lstate.buffer_entries) {
		lstate.handle.Destroy();
		lstate.table = nullptr;
		FusedAllocateTable(lstate, capacity, 1);
	}
	lstate.capacity = capacity;
	lstate.table_bits = FusedTableBits(capacity);
	const auto shape = FusedGetKeyShape(fused.key_bytes);
	const auto row_width = fused.row_width;
	idx_t occupancy = 0;
	bool zero = false;
	uint64_t key[FusedIntegerAggregate::TABLE_BATCH];
	hash_t hash[FusedIntegerAggregate::TABLE_BATCH];
	for (auto &lists : gstate.handed_over) {
		for (auto chunk = lists->heads[p]; chunk; chunk = FusedChunkNext(chunk)) {
			const auto chunk_rows = FusedChunkRows(chunk);
			const auto base = chunk + FusedIntegerAggregate::CHUNK_HEADER_BYTES;
			for (idx_t batch_start = 0; batch_start < chunk_rows; batch_start += FusedIntegerAggregate::TABLE_BATCH) {
				const auto batch = MinValue<idx_t>(FusedIntegerAggregate::TABLE_BATCH, chunk_rows - batch_start);
				const auto batch_rows = base + batch_start * row_width;
				for (idx_t j = 0; j < batch; j++) {
					uint64_t unused;
					FusedLoadKey(batch_rows + j * row_width, shape, key[j], unused);
					hash[j] = FusedHashKey(key[j], 0, false);
					FUSED_PREFETCH_WRITE(lstate.table + FusedSlotOf(hash[j], lstate.table_bits));
				}
				for (idx_t j = 0; j < batch; j++) {
					if (key[j] == 0) {
						zero = true;
						continue;
					}
					const auto mask = lstate.capacity - 1;
					auto slot = FusedSlotOf(hash[j], lstate.table_bits);
					while (true) {
						const auto present = lstate.table[slot];
						if (present == key[j]) {
							break;
						}
						if (present == 0) {
							lstate.table[slot] = key[j];
							if (++occupancy * 2 > lstate.capacity) {
								FusedGrowSet(lstate);
							}
							break;
						}
						slot = (slot + 1) & mask;
					}
				}
			}
		}
	}
	memset(lstate.table, 0, lstate.capacity * sizeof(uint64_t));
	return occupancy + (zero ? 1 : 0);
}

//! The set member's source: each task counts itself before its first claim, then counts the distinct x of every
//! partition it claims (FusedBuildSet); the task that brings tasks_done up to tasks_started emits the one row, the sum
//! (the merge rule of the bitmap class: every claimed partition is counted by a task that is done by then)
static SourceResultType FusedSetGetData(const FusedIntegerAggregate &fused, DataChunk &chunk,
                                        FusedAggregateGlobalState &gstate, OperatorSourceInput &input) {
	auto &source = input.global_state.Cast<FusedAggregateGlobalSourceState>();
	auto &lstate = input.local_state.Cast<FusedAggregateLocalSourceState>();
	if (lstate.emit_done) {
		return SourceResultType::FINISHED;
	}
	lstate.emit_done = true;
	{
		lock_guard<mutex> guard(source.merge_lock);
		source.tasks_started++;
	}
	idx_t distinct = 0;
	while (true) {
		idx_t p;
		do {
			p = source.next_partition++;
		} while (p < FusedIntegerAggregate::PARTITION_COUNT && gstate.partition_rows[p] == 0);
		if (p >= FusedIntegerAggregate::PARTITION_COUNT) {
			break;
		}
		source.claimed++;
		distinct += FusedBuildSet(fused, gstate, lstate, p);
	}
	idx_t total;
	{
		lock_guard<mutex> guard(source.merge_lock);
		source.distinct_entries += distinct;
		source.tasks_done++;
		if (source.tasks_done != source.tasks_started || source.merged) {
			return SourceResultType::FINISHED;
		}
		source.merged = true;
		source.merge_done = true;
		source.groups = 1;
		source.tasks_merged = source.tasks_done;
		total = source.distinct_entries;
	}
	chunk.SetCardinality(1);
	FlatVector::GetData<int64_t>(chunk.data[0])[0] = NumericCast<int64_t>(total);
	return SourceResultType::FINISHED;
}

SourceResultType FusedIntegerAggregate::GetData(ExecutionContext &context, DataChunk &chunk,
                                                FusedAggregateGlobalState &gstate, OperatorSourceInput &input) const {
	if (bitmap) {
		return FusedBitmapGetData(*this, chunk, gstate, input);
	}
	if (distinct_set) {
		return FusedSetGetData(*this, chunk, gstate, input);
	}
	if (distinct) {
		return FusedDistinctGetData(*this, chunk, gstate, input);
	}
	if (chain) {
		return FusedChainScan(*this, chunk, gstate, input);
	}
	auto &source = input.global_state.Cast<FusedAggregateGlobalSourceState>();
	auto &lstate = input.local_state.Cast<FusedAggregateLocalSourceState>();
	const auto words = FusedEntryWords(*this);
	const auto count_word = FusedKeyWords(*this);
	idx_t count = 0;
	while (count < STANDARD_VECTOR_SIZE) {
		if (!lstate.active) {
			if (count > 0) {
				// the chunk's entries point into the table buffer, which the next build reuses (or frees): emit and
				// clear them first
				break;
			}
			// claim the next non-empty partition and build its table
			idx_t p;
			do {
				p = source.next_partition++;
			} while (p < PARTITION_COUNT && gstate.partition_rows[p] == 0);
			if (p >= PARTITION_COUNT) {
				break;
			}
			source.claimed++;
			if (hash_stored) {
				FusedBuildTableStored(*this, gstate, lstate, p);
			} else {
				FusedBuildTable(*this, gstate, lstate, p);
			}
			lstate.active = true;
			lstate.scan_position = 0;
		}
		// scan it once, clearing each emitted entry (the buffer is all-zero again when the scan ends)
		while (lstate.scan_position < lstate.capacity && count < STANDARD_VECTOR_SIZE) {
			auto entry = lstate.table + lstate.scan_position * words;
			lstate.scan_position++;
			if (entry[count_word] != 0) {
				lstate.emit[count++] = entry;
			}
		}
		if (lstate.scan_position == lstate.capacity) {
			lstate.active = false;
		}
	}
	FusedEmit(*this, chunk, lstate.emit, count);
	for (idx_t i = 0; i < count; i++) {
		lstate.emit[i][count_word] = 0;
	}
	return count == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

ProgressData FusedIntegerAggregate::GetProgress(FusedAggregateGlobalState &gstate,
                                                GlobalSourceState &source_state) const {
	auto &source = source_state.Cast<FusedAggregateGlobalSourceState>();
	ProgressData progress;
	progress.done = double(source.claimed.load());
	progress.total = double(MaxValue<idx_t>(1, gstate.partitions_nonempty.load()));
	return progress;
}

//===--------------------------------------------------------------------===//
// The EXPLAIN text
//===--------------------------------------------------------------------===//
string FusedIntegerAggregate::ParamsString(optional_ptr<FusedAggregateGlobalState> gstate,
                                           OrderPreservationType source_order) const {
	if (bitmap) {
		// the bitmap class's marker: the class, its code domain and words; with a sink state, the input rows, the threads' bitmaps
		// and the distinct overflow strings
		string result = "bitmap=1 codes=" + to_string(bitmap_dict->count) + " words=" + to_string(bitmap_words) +
		                " budget_mib=" + to_string(budget_bytes >> 20);
		if (gstate) {
			// this execution's re-check of the coverage bound
			result += string(" engaged=") + (gstate->bitmap_engaged ? "1" : "0") +
			          " refusal=" + FusedBitmapRefusalName(gstate->bitmap_refusal);
			result += " input_rows=" + to_string(gstate->input_rows.load()) +
			          " lookup_rows=" + to_string(gstate->bitmap_lookups) +
			          " bitmaps=" + to_string(gstate->bitmaps.size()) +
			          " overflow=" + to_string(gstate->bitmap_overflow.size());
			// the overflow strings' bytes and the last word's padding bits (both 0 under the coverage bound)
			idx_t overflow_bytes = 0;
			for (auto &text : gstate->bitmap_overflow) {
				overflow_bytes += text.size();
			}
			result += " overflow_bytes=" + to_string(overflow_bytes) +
			          " padding_bits=" + to_string(gstate->bitmap_padding_bits.load());
		}
		return result;
	}
	string result = "keys=" + to_string(key_count) + (distinct ? " distinct=1" : "") + (distinct_set ? " set=1" : "") +
	                " aggregates=" + to_string(outputs.size()) + " partitions=" + to_string(PARTITION_COUNT) +
	                " budget_mib=" + to_string(budget_bytes >> 20);
	if (hash_stored) {
		result += " hash=stored";
	}
	if (chain) {
		result += " chain=on";
	}
	if (distinct) {
		result += string(" source_order=") + (source_order == OrderPreservationType::NO_ORDER ? "no_order" : "default");
		if (gid_keys > 0) {
			result += " gid_keys=" + to_string(gid_keys);
		}
	}
	result += run_kind ? " kind=run" : " kind=rows";
	if (last_key_fold) {
		// phase 1 folds a row into its partition's last row
		result += " fold=last_key";
	}
	result += fused_atomic_reserve ? " reserve=ldadd" : " reserve=cas";
	if (gstate) {
		result += " input_rows=" + to_string(gstate->input_rows.load());
		if (distinct || last_key_fold) {
			result += " folded_rows=" + to_string(gstate->folded_rows.load());
		}
		if (mixed) {
			result += " companion_rows=" + to_string(gstate->companion_rows.load());
		}
		result += " drained_rows=" + to_string(gstate->drained_rows.load());
		result += " combines_taken=" + to_string(gstate->combines_taken.load());
		result += " reserved_bytes=" + to_string(gstate->reserved_bytes.load());
		result += " slabs_allocated=" + to_string(gstate->slabs_allocated.load());
		result += " partitions_nonempty=" + to_string(gstate->partitions_nonempty.load());
		if (gid_keys > 0) {
			result += " gid_count=" + to_string(gstate->gid_count.load());
			result += " gid_maps=" + to_string(gstate->gid_maps.load());
			result += " gid_rows_flat=" + to_string(gstate->gid_rows_flat.load());
		}
		if (run_kind) {
			result += " run_rows=" + to_string(gstate->run_rows.load());
			result += " run_length_sum=" + to_string(gstate->run_length_sum.load());
		}
	}
	return result;
}

InsertionOrderPreservingMap<string> FusedIntegerAggregate::ExtraSourceParams(GlobalSourceState &source_state) const {
	InsertionOrderPreservingMap<string> result;
	if (bitmap) {
		// no phase-2 counters beyond the marker's
		return result;
	}
	auto &source = source_state.Cast<FusedAggregateGlobalSourceState>();
	if (!distinct) {
		return result;
	}
	lock_guard<mutex> guard(source.merge_lock);
	if (!source.merge_done) {
		// before the merge nothing is emitted, so a flush leaves the profiler's entry untouched
		return result;
	}
	result["Fused Distinct Phase 2"] = "distinct_entries=" + to_string(source.distinct_entries) +
	                                   " groups=" + to_string(source.groups) +
	                                   (mixed ? " companion_groups=" + to_string(source.companion_groups) : string()) +
	                                   " tasks_started=" + to_string(source.tasks_started) +
	                                   " tasks_merged=" + to_string(source.tasks_merged);
	return result;
}

} // namespace duckdb
