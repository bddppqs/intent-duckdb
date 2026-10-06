//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/table/persisted_row_group_index.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/storage/metadata/metadata_manager.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"

namespace duckdb {
class MetadataWriter;
struct PersistentColumnData;
struct RowGroupPointer;
class SerializationOptions;

//! The row-group index of a table, written at a checkpoint of a file at the persisted-row-group-index storage version
//! (kPersistedRowGroupIndex) into the table's metadata stream right after its row-group pointers:
//! - the positions: the row-group count, then the position of every row-group pointer in the pointer stream;
//! - per column, the column's statistics of every row group as a load of the column computes them (a flag, then the
//!   statistics when they are stored);
//! - the directory, whose position is the table entry's property 107: the row-group count, the position of the
//!   positions and of each column's statistics, and the metadata blocks the positions and the statistics occupy.
//! Nothing of it is read when the database is attached: the directory at the first use of the index (which requests
//! the index's blocks at once, so a cold read of a column's statistics finds them read or in flight), the positions at
//! the first load of every row-group pointer, a column's statistics at the first request for them.
class PersistedRowGroupIndex {
public:
	PersistedRowGroupIndex(MetadataManager &manager, MetaBlockPointer directory);

	MetaBlockPointer GetDirectory() const {
		return directory;
	}
	//! The position of every row-group pointer (read once)
	const vector<MetaBlockPointer> &GetPositions();
	//! Row group k's statistics of column c as a load of the column computes them, or nullptr when they are not stored
	unique_ptr<BaseStatistics> GetStatistics(idx_t row_group, idx_t column, const LogicalType &type);
	//! Appends every metadata block the index occupies (an unchanged table's checkpoint keeps them in use)
	void AppendBlocks(vector<MetaBlockPointer> &blocks);

	//! The statistics a load of a column computes from its persistent data, or nullptr where they are not stored: a
	//! column whose data is not one standard column (nested types, geometry, variant) or that carries updates
	static shared_ptr<BaseStatistics> LoadedStatistics(const LogicalType &type, const PersistentColumnData &data);
	//! Writes the index after the row-group pointers and returns the directory's position. `written` is the writer's
	//! list of the metadata blocks it started, of which those from `written_start` on are the index's
	static MetaBlockPointer Write(MetadataWriter &writer, const vector<MetaBlockPointer> &positions,
	                              const vector<RowGroupPointer> &row_group_pointers, const vector<LogicalType> &types,
	                              const SerializationOptions &options, const vector<MetaBlockPointer> &written,
	                              idx_t written_start);

private:
	void ReadDirectory();
	const vector<unique_ptr<BaseStatistics>> &ReadColumn(idx_t column, const LogicalType &type);

private:
	mutex lock;
	MetadataManager &manager;
	MetaBlockPointer directory;
	bool directory_read = false;
	idx_t row_group_count = 0;
	MetaBlockPointer positions_start;
	vector<MetaBlockPointer> column_starts;
	bool positions_read = false;
	vector<MetaBlockPointer> positions;
	vector<unique_ptr<vector<unique_ptr<BaseStatistics>>>> columns;
	//! The directory read, and each column's statistics once read: published for a copy without the lock (never changed
	//! after they are read)
	atomic<bool> directory_ready {false};
	unique_ptr<atomic<const vector<unique_ptr<BaseStatistics>> *>[]> published;
};

} // namespace duckdb
