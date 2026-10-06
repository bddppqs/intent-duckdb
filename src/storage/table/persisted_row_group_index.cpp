#include "duckdb/storage/table/persisted_row_group_index.hpp"

#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/tuning_defaults.hpp"
#include "duckdb/storage/data_pointer.hpp"
#include "duckdb/storage/metadata/metadata_reader.hpp"
#include "duckdb/storage/metadata/metadata_writer.hpp"
#include "duckdb/storage/table/column_data.hpp"

namespace duckdb {

PersistedRowGroupIndex::PersistedRowGroupIndex(MetadataManager &manager, MetaBlockPointer directory)
    : manager(manager), directory(directory) {
}

void PersistedRowGroupIndex::ReadDirectory() {
	if (directory_read) {
		return;
	}
	MetadataReader reader(manager, directory);
	BinaryDeserializer deserializer(reader);
	deserializer.Begin();
	row_group_count = deserializer.ReadProperty<idx_t>(100, "row_group_count");
	positions_start = deserializer.ReadProperty<MetaBlockPointer>(101, "positions");
	column_starts = deserializer.ReadProperty<vector<MetaBlockPointer>>(102, "column_statistics");
	auto blocks = deserializer.ReadPropertyWithDefault<vector<MetaBlockPointer>>(103, "blocks");
	deserializer.End();
	columns.resize(column_starts.size());
	published = unique_ptr<atomic<const vector<unique_ptr<BaseStatistics>> *>[]>(
	    new atomic<const vector<unique_ptr<BaseStatistics>> *>[column_starts.size()]);
	for (idx_t c = 0; c < column_starts.size(); c++) {
		published[c].store(nullptr, std::memory_order_relaxed);
	}
	directory_read = true;
	directory_ready.store(true, std::memory_order_release);
	if (kPersistedIndexReadAhead) {
		// the positions and every column's statistics requested at once: a later read finds its block read or in flight
		manager.ReadAhead(blocks);
	}
}

const vector<MetaBlockPointer> &PersistedRowGroupIndex::GetPositions() {
	lock_guard<mutex> guard(lock);
	if (positions_read) {
		return positions;
	}
	ReadDirectory();
	MetadataReader reader(manager, positions_start);
	BinaryDeserializer deserializer(reader);
	deserializer.Begin();
	positions = deserializer.ReadProperty<vector<MetaBlockPointer>>(100, "positions");
	deserializer.End();
	if (positions.size() != row_group_count) {
		throw IOException("Row-group index holds %llu positions for %llu row groups. Corrupt file?", positions.size(),
		                  row_group_count);
	}
	positions_read = true;
	return positions;
}

const vector<unique_ptr<BaseStatistics>> &PersistedRowGroupIndex::ReadColumn(idx_t column, const LogicalType &type) {
	ReadDirectory();
	if (column >= columns.size()) {
		throw IOException("Row-group index holds statistics of %llu columns, column %llu requested. Corrupt file?",
		                  columns.size(), column);
	}
	if (columns[column]) {
		return *columns[column];
	}
	auto result = make_uniq<vector<unique_ptr<BaseStatistics>>>();
	result->resize(row_group_count);
	MetadataReader reader(manager, column_starts[column]);
	BinaryDeserializer deserializer(reader);
	deserializer.Set<const LogicalType &>(type);
	deserializer.Begin();
	auto count = deserializer.ReadProperty<idx_t>(100, "row_group_count");
	if (count != row_group_count) {
		throw IOException("Row-group index holds statistics of %llu row groups for %llu. Corrupt file?", count,
		                  row_group_count);
	}
	deserializer.ReadList(101, "statistics", [&](Deserializer::List &list, idx_t k) {
		list.ReadObject([&](Deserializer &object) {
			auto stored = object.ReadProperty<bool>(100, "stored");
			if (stored) {
				(*result)[k] = object.ReadProperty<BaseStatistics>(101, "statistics").ToUnique();
			}
		});
	});
	deserializer.End();
	deserializer.Unset<LogicalType>();
	columns[column] = std::move(result);
	published[column].store(columns[column].get(), std::memory_order_release);
	return *columns[column];
}

unique_ptr<BaseStatistics> PersistedRowGroupIndex::GetStatistics(idx_t row_group, idx_t column,
                                                                 const LogicalType &type) {
	optional_ptr<const vector<unique_ptr<BaseStatistics>>> stats;
	if (directory_ready.load(std::memory_order_acquire) && column < column_starts.size()) {
		// a column already read: copied without the lock
		stats = published[column].load(std::memory_order_acquire);
	}
	if (!stats) {
		lock_guard<mutex> guard(lock);
		stats = &ReadColumn(column, type);
	}
	if (row_group >= stats->size() || !(*stats)[row_group]) {
		return nullptr;
	}
	return (*stats)[row_group]->ToUnique();
}

void PersistedRowGroupIndex::AppendBlocks(vector<MetaBlockPointer> &blocks) {
	lock_guard<mutex> guard(lock);
	ReadDirectory();
	// the index is one run of the table's metadata stream: walk the chain of metadata blocks from the positions to
	// the directory's first block, then read the directory once more to record the blocks it spans
	auto block_size = manager.GetMetadataBlockSize();
	MetaBlockPointer current(positions_start.block_pointer, 0);
	while (current.block_pointer != directory.block_pointer) {
		blocks.push_back(current);
		auto handle = manager.Pin(manager.FromDiskPointer(current));
		auto next = Load<idx_t>(handle.handle.Ptr() + handle.pointer.index * block_size);
		if (next == idx_t(-1)) {
			throw IOException("Row-group index: its directory does not follow its positions. Corrupt file?");
		}
		current = MetaBlockPointer(next, 0);
	}
	MetadataReader reader(manager, directory, &blocks);
	BinaryDeserializer deserializer(reader);
	deserializer.Begin();
	deserializer.ReadProperty<idx_t>(100, "row_group_count");
	deserializer.ReadProperty<MetaBlockPointer>(101, "positions");
	deserializer.ReadProperty<vector<MetaBlockPointer>>(102, "column_statistics");
	deserializer.ReadPropertyWithDefault<vector<MetaBlockPointer>>(103, "blocks");
	deserializer.End();
}

shared_ptr<BaseStatistics> PersistedRowGroupIndex::LoadedStatistics(const LogicalType &type,
                                                                    const PersistentColumnData &data) {
	// a standard column (ColumnData::CreateColumn) whose load merges its segments' statistics and then its validity's
	// into empty statistics (StandardColumnData::InitializeColumn)
	if (type.id() == LogicalTypeId::GEOMETRY || type.id() == LogicalTypeId::VARIANT ||
	    type.id() == LogicalTypeId::VALIDITY || type.InternalType() == PhysicalType::STRUCT ||
	    type.InternalType() == PhysicalType::LIST || type.InternalType() == PhysicalType::ARRAY) {
		return nullptr;
	}
	if (data.HasUpdates() || data.child_columns.size() != 1) {
		return nullptr;
	}
	auto result = make_shared_ptr<BaseStatistics>(BaseStatistics::CreateEmpty(type));
	for (auto &pointer : data.pointers) {
		result->Merge(pointer.statistics);
	}
	for (auto &pointer : data.child_columns[0].pointers) {
		result->Merge(pointer.statistics);
	}
	return result;
}

MetaBlockPointer PersistedRowGroupIndex::Write(MetadataWriter &writer, const vector<MetaBlockPointer> &positions,
                                               const vector<RowGroupPointer> &row_group_pointers,
                                               const vector<LogicalType> &types, const SerializationOptions &options,
                                               const vector<MetaBlockPointer> &written, idx_t written_start) {
	D_ASSERT(positions.size() == row_group_pointers.size());
	auto positions_start = writer.GetMetaBlockPointer();
	{
		BinarySerializer serializer(writer, options);
		serializer.Begin();
		serializer.WriteProperty(100, "positions", positions);
		serializer.End();
	}
	vector<MetaBlockPointer> column_starts;
	for (idx_t c = 0; c < types.size(); c++) {
		column_starts.push_back(writer.GetMetaBlockPointer());
		BinarySerializer serializer(writer, options);
		serializer.Begin();
		serializer.WriteProperty<idx_t>(100, "row_group_count", row_group_pointers.size());
		serializer.WriteList(101, "statistics", row_group_pointers.size(), [&](Serializer::List &list, idx_t k) {
			auto &stats = row_group_pointers[k].column_statistics;
			auto entry = c < stats.size() ? stats[c].get() : nullptr;
			list.WriteObject([&](Serializer &object) {
				object.WriteProperty<bool>(100, "stored", entry != nullptr);
				if (entry) {
					object.WriteProperty(101, "statistics", *entry);
				}
			});
		});
		serializer.End();
	}
	// the blocks the positions and the statistics occupy: the one the positions start in, then every block the writer
	// started since the index began
	vector<MetaBlockPointer> blocks;
	blocks.emplace_back(positions_start.block_pointer, 0);
	for (idx_t i = written_start; i < written.size(); i++) {
		if (written[i].block_pointer != positions_start.block_pointer) {
			blocks.emplace_back(written[i].block_pointer, 0);
		}
	}
	auto directory = writer.GetMetaBlockPointer();
	BinarySerializer serializer(writer, options);
	serializer.Begin();
	serializer.WriteProperty<idx_t>(100, "row_group_count", row_group_pointers.size());
	serializer.WriteProperty(101, "positions", positions_start);
	serializer.WriteProperty(102, "column_statistics", column_starts);
	serializer.WriteProperty(103, "blocks", blocks);
	serializer.End();
	return directory;
}

} // namespace duckdb
