//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/table/row_group_reorderer.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/partition_stats.hpp"
#include "duckdb/storage/table/row_group.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"
#include "duckdb/storage/table/segment_tree.hpp"
#include "duckdb/common/enums/order_type.hpp"

#include <condition_variable>

namespace duckdb {
class TableFilterSet;
class CollectionScanState;
struct DynamicFilterData;

enum class OrderByStatistics : uint8_t { MIN, MAX };
enum class OrderByColumnType : uint8_t { NUMERIC, STRING };

struct RowGroupOrderOptions {
	RowGroupOrderOptions(const StorageIndex &column_idx_p, OrderByStatistics order_by_p, OrderType order_type_p,
	                     OrderByNullType null_order_p, OrderByColumnType column_type_p,
	                     optional_idx row_limit_p = optional_idx(), idx_t row_group_offset_p = 0,
	                     idx_t leading_null_group_offset_p = 0)
	    : column_idx(column_idx_p), order_by(order_by_p), order_type(order_type_p), null_order(null_order_p),
	      column_type(column_type_p), row_limit(row_limit_p), row_group_offset(row_group_offset_p),
	      leading_null_group_offset(leading_null_group_offset_p) {
	}

	const StorageIndex column_idx;
	const OrderByStatistics order_by;
	const OrderType order_type;
	const OrderByNullType null_order;
	const OrderByColumnType column_type;
	const optional_idx row_limit;
	const idx_t row_group_offset;
	const idx_t leading_null_group_offset;

	void Serialize(Serializer &serializer) const;
	static unique_ptr<RowGroupOrderOptions> Deserialize(Deserializer &deserializer);
};

struct OffsetPruningResult {
	idx_t offset_remainder;
	idx_t pruned_row_group_count;
	idx_t leading_null_group_offset;
};

//! The wave gate of an ordered parallel scan that carries a Top-N bound (RowGroupReorderer::SetTopNBound), kept by
//! RowGroupCollection::NextParallelScan under the parallel scan state's lock
struct TopNWaveGate {
	//! The Top-N bound on the order column
	shared_ptr<DynamicFilterData> bound;
	//! The scan holding the prefix (the first vectors of the first ordered row group), and whether the prefix is done
	optional_ptr<CollectionScanState> prefix_holder;
	bool prefix_done = false;
	//! The hand-outs whose scans hold a slot
	idx_t in_flight = 0;
	//! Notified when the prefix ends or passes on, and when a slot is released
	std::condition_variable prefix_end;
	std::condition_variable slot_released;
};

class RowGroupReorderer {
public:
	RowGroupReorderer(const RowGroupOrderOptions &options_p, TransactionData transaction_p);
	optional_ptr<SegmentNode<RowGroup>> GetRootSegment(RowGroupSegmentTree &row_groups);
	optional_ptr<SegmentNode<RowGroup>> GetNextRowGroup(SegmentNode<RowGroup> &row_group);
	//! The scan passes only non-empty values of the order column: an ascending order by statistics that only orders
	//! (no limit or offset pruning) keys a VARCHAR row group on its minimum non-empty value, so the row groups holding
	//! the smallest non-empty values are scanned first and a Top-N bound tightens early (kNonEmptyMinRowGroupOrder)
	void SetScanExcludesEmptyString(const TableFilterSet *filters, const vector<StorageIndex> &column_ids);
	//! The scan carries the bound of a Top-N on the order column (a dynamic filter): enable the wave gate of the parallel
	//! scan, which holds the first hand-outs to a short prefix and then bounds the hand-outs in flight while the bound is
	//! set (kTopNWaveGate)
	void SetTopNBound(const TableFilterSet *filters, const vector<StorageIndex> &column_ids);
	//! The armed wave gate, or nullptr
	optional_ptr<TopNWaveGate> GetTopNWaveGate() {
		return wave_gate.get();
	}

	static Value RetrieveStat(const BaseStatistics &stats, OrderByStatistics order_by, OrderByColumnType column_type);
	static OffsetPruningResult GetOffsetAfterPruning(OrderByStatistics order_by, OrderByColumnType column_type,
	                                                 OrderType order_type, OrderByNullType null_order,
	                                                 const StorageIndex &column_idx, idx_t row_offset,
	                                                 vector<PartitionStatistics> &stats);

private:
	const RowGroupOrderOptions options;
	const TransactionData transaction;

	idx_t offset;
	bool initialized;
	bool nonempty_min_key = false;
	vector<reference<SegmentNode<RowGroup>>> ordered_row_groups;
	unique_ptr<TopNWaveGate> wave_gate;
};

} // namespace duckdb
