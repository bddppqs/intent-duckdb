//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/table/row_group_segment_tree.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/storage/table/segment_tree.hpp"
#include "duckdb/storage/table/row_group.hpp"

namespace duckdb {
struct DataTableInfo;
class PersistentTableData;
class MetadataReader;
class PersistedRowGroupIndex;

class RowGroupSegmentTree : public SegmentTree<RowGroup, true> {
public:
	RowGroupSegmentTree(RowGroupCollection &collection, idx_t base_row_id);
	~RowGroupSegmentTree() override;

	void Initialize(PersistentTableData &data, optional_ptr<vector<MetaBlockPointer>> read_pointers = nullptr);

	MetaBlockPointer GetRootPointer() const {
		return root_pointer;
	}

	//! The table's persisted row-group index (nullptr: the file has none)
	const shared_ptr<PersistedRowGroupIndex> &GetPersistedIndex() const {
		return persisted_index;
	}

protected:
	shared_ptr<RowGroup> LoadSegment() const override;
	//! With a persisted row-group index, the row-group pointers not loaded yet are read from their positions in
	//! parallel on the database's task scheduler
	bool LoadRemainingSegments(vector<shared_ptr<RowGroup>> &result) const override;

	RowGroupCollection &collection;
	mutable idx_t current_row_group;
	mutable idx_t max_row_group;
	mutable unique_ptr<MetadataReader> reader;
	MetaBlockPointer root_pointer;
	//! The metadata blocks read (the collection's, see RowGroupCollection::Initialize)
	mutable optional_ptr<vector<MetaBlockPointer>> read_pointers;
	shared_ptr<PersistedRowGroupIndex> persisted_index;
};

} // namespace duckdb
