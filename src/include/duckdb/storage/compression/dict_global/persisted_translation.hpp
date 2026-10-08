//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/compression/dict_global/persisted_translation.hpp
//
// Column translations stored at checkpoint: each admitted VARCHAR column's DICT_FSST segments have their local
// codes translated into one column-wide code space numbered by first occurrence - segment order, then local order;
// code 0 is NULL - and stored with the table: per segment, a bitmap of the local entries that occur first there (their
// codes are implicit: the segment's start code plus their rank) and the codes of the others, which occurred in an
// earlier segment. Every string stays in the segment dictionary where it first occurs, so the column-wide dictionary is
// implicit: code c is found by a search over the segments' start codes and a select on that segment's bitmap, and read
// from that segment's dictionary block. The blob is read on first use inside a statement, never at attach.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"
#include "duckdb/storage/storage_info.hpp"

namespace duckdb {
class BlockHandle;
class BlockManager;
class ClientContext;
class ColumnData;
class ColumnDefinition;
class ColumnSegment;
class DatabaseInstance;
class Deserializer;
class PersistentTableData;
class RowGroupCollection;
class Serializer;
class TableFilter;
struct DataTableInfo;

namespace dict_global {
class ColumnDictionary;
class SegmentTranslation;

//! Whether stored column translations are written and read (a compile-time constant)
bool PersistedTranslationsEnabled();

//! One stored column's translations, as the table data records them: the blob's blocks and the identity of the
//! segments they translate
struct PersistedColumn {
	idx_t storage_index = 0;
	//! a hash of the column's segments (block, offset, count, dict_count) in collection order at the checkpoint
	uint64_t identity = 0;
	//! the code space, slot 0 (NULL) included
	idx_t count = 0;
	idx_t segments = 0;
	idx_t rows = 0;
	idx_t blob_bytes = 0;
	//! the blob's block ids, in blob order
	vector<int64_t> blocks;
	//! the byte-length table's block ids, in order (empty: the entry stores no byte lengths)
	vector<int64_t> length_blocks;
	//! the byte-length table's bytes: its header and one 16-bit length per code
	idx_t length_bytes = 0;
	//! the longest decoded string of the column, in bytes (0 when no length pass ran)
	idx_t max_length = 0;

	void Serialize(Serializer &serializer) const;
	static PersistedColumn Deserialize(Deserializer &deserializer);
};

//! A stored column's translations once read: its blob directory (read inside a statement, on first use) and the reads
//! of one segment's translation and of one code's string
class PersistedTranslations {
public:
	PersistedTranslations(DatabaseInstance &db, BlockManager &block_manager, PersistedColumn column);
	~PersistedTranslations();

	DatabaseInstance &db;
	BlockManager &block_manager;
	const PersistedColumn column;
	const idx_t storage_index;

	struct Entry {
		int64_t block_id;
		uint32_t offset;
		uint32_t dict_count;
		uint32_t start_code;
		uint32_t new_count;
		uint64_t data_offset;
		uint64_t rows;
		uint8_t old_width;
	};
	//! The directory (EnsureDirectory first)
	const vector<Entry> &Entries();
	//! The code of the empty string, or 0 when the column holds none (EnsureDirectory first)
	uint32_t EmptyCode();
	//! The code space, slot 0 (NULL) included
	idx_t Count() const {
		return column.count;
	}
	//! The local -> code array of entry `entry` (dict_count codes, code[0] = 0)
	void Decode(idx_t entry, uint32_t *codes);
	//! The string of code `code` (1 <= code < Count()), into `result`'s string heap
	string_t Fetch(Vector &result, uint32_t code);
	//! The entry of the segment stored at (block_id, offset), or INVALID_INDEX
	idx_t FindEntry(int64_t block_id, uint32_t offset);
	//! Whether the entry stores the byte length of every code
	bool HasLengths() const {
		return !column.length_blocks.empty();
	}
	//! The decoded byte length of every code (Count() entries, slot 0 = 0), read whole on first use; null when the entry
	//! stores none
	const uint16_t *Lengths();
	//! Set when a publication skipped the link walk (the table unchanged since its load): from then on each load of
	//! the column in a row group links its segments (LinkLoadedColumn) before the column is visible to a scan
	atomic<bool> lazy_link {false};

private:
	void EnsureDirectory();
	//! `bytes` bytes of the blob at `offset` (never across a block boundary) - a pinned view
	const_data_ptr_t Read(idx_t offset, idx_t bytes, BufferHandle &handle);

	mutex lock;
	atomic<bool> loaded;
	vector<Entry> entries;
	uint32_t empty_code = 0;
	//! the entries with new codes, by start code (the inverse search)
	vector<uint32_t> inverse_starts;
	vector<uint32_t> inverse_entries;
	unordered_map<uint64_t, idx_t> by_location;
	vector<shared_ptr<BlockHandle>> handles;
	//! the segment blocks Fetch has read (under `lock`): held so an unpinned block stays registered in the buffer pool
	//! (evictable under pressure) instead of being dropped when Fetch returns and read again by the next decode
	unordered_map<int64_t, shared_ptr<BlockHandle>> fetch_blocks;
	//! the byte-length table once read (Lengths(), under `lock`), and the bytes reserved for it in the buffer pool
	atomic<bool> lengths_loaded {false};
	unsafe_unique_array<uint16_t> lengths;
	idx_t lengths_reserved = 0;
};

//===--------------------------------------------------------------------===//
// The table's stored columns
//===--------------------------------------------------------------------===//
//! SingleFileTableDataWriter::FinalizeTable: the table's stored columns after this checkpoint. `unchanged` (the table's
//! metadata is reused): every entry carried. Otherwise each admitted VARCHAR column's entry is carried when its segments'
//! identity is unchanged, else its translations are built (in parallel on the scheduler) and written to new blocks of
//! `block_manager`; the blocks of an entry not carried are freed
vector<PersistedColumn> PersistAtCheckpoint(optional_ptr<ClientContext> context, DatabaseInstance &db,
                                            DataTableInfo &info, RowGroupCollection &collection,
                                            const vector<ColumnDefinition> &columns, BlockManager &block_manager,
                                            bool unchanged);
//! CheckpointReader: the stored columns read with a table's data, kept until the DataTable built from it adopts them
void StashPersisted(const PersistentTableData &data, vector<PersistedColumn> columns);
//! DataTable(..., data): the table adopts the stored columns read with its data
void AdoptPersisted(const PersistentTableData &data, const DataTableInfo &info);
//! ~DataTableInfo: the table's stored-column entries are forgotten
void ReleasePersisted(const DataTableInfo &info) noexcept;
//! An alter of the table (which may renumber its storage indexes): its stored translations are not read until the next
//! checkpoint, which carries the entries whose segments are unchanged and frees the others
void InvalidatePersisted(const DataTableInfo &info) noexcept;
//! DataTable::CommitDropTable: the blocks of the table's stored columns are freed and its entries forgotten
void CommitDropPersisted(const DataTableInfo &info, BlockManager &block_manager);
//! The stored translations of a table's column (read lazily), or null
shared_ptr<PersistedTranslations> FindPersistedTranslations(const DataTableInfo &info, idx_t storage_index);
//! Whether any publication in this process links lazily (PersistedTranslations::lazy_link): the column loads ask
bool LazyLinkActive();
//! RowGroup::LoadColumn, under the row group's lock and before the column is marked loaded: link the segments of the
//! column `storage_index` just loaded to its stored translations' entries when they link lazily
void LinkLoadedColumn(const DataTableInfo &info, idx_t storage_index, ColumnData &column);

//===--------------------------------------------------------------------===//
// Codes-only reads
//===--------------------------------------------------------------------===//
//! Whether the scan on this thread reads the segment's column codes only (its consumers never read the strings)
bool SegmentReadsCodesOnly(ColumnSegment &segment);
//! The segment's translation for a codes-only scan on this thread, or null (the standard per-segment scan)
shared_ptr<SegmentTranslation> CodesOnlyTranslation(ColumnSegment &segment);
//! A pushed filter decided on codes: = or <> with the empty string, IS NULL, IS NOT NULL, their AND, and optional
//! filters (not needed for correctness: dropped)
bool CodeTranslatable(const TableFilter &filter);
//! Whether a NULL, an empty-string and any other code pass `filter` (CodeTranslatable), `empty_code` the column's empty
//! string code (0: the column holds none); a filter CodeTranslatable does not admit throws an InternalException
void CodeClassVerdicts(const TableFilter &filter, uint32_t empty_code, bool &null_passes, bool &empty_passes,
                       bool &other_passes);
//! The rows of the selection whose code passes `filter` (CodeTranslatable), `codes` the vector's codes: sel and
//! sel_count as a segment filter function takes and returns them; a filter CodeTranslatable does not admit throws an
//! InternalException
void FilterCodes(PersistedTranslations &translations, const TableFilter &filter, const sel_t *codes, idx_t count,
                 SelectionVector &sel, idx_t &sel_count);
//! The generic filter path over a vector a codes-only scan emitted: the filter decided on its codes; false for any
//! other vector
bool FilterCodesOnlyVector(Vector &result, idx_t count, SelectionVector &sel, idx_t &sel_count,
                           const TableFilter &filter);

} // namespace dict_global
} // namespace duckdb
