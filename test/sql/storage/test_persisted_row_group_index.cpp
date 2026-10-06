#include "catch.hpp"
#include "test_helpers.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/row_group.hpp"
#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"

using namespace duckdb;

static duckdb::vector<uint8_t> StatisticsBytes(const BaseStatistics &stats) {
	MemoryStream stream;
	BinarySerializer::Serialize(stats, stream);
	return duckdb::vector<uint8_t>(stream.GetData(), stream.GetData() + stream.GetPosition());
}

// The persisted row-group index stores, per row group and column, the statistics a load of the column computes: for
// every row group of a file at the persisted-row-group-index version, the persisted statistics of a column not loaded
// equal (byte for byte, serialized) the statistics of the same column once loaded; a nested column stores none. The
// row groups are loaded in parallel from their persisted positions
TEST_CASE("Persisted row-group statistics equal the loaded column's", "[storage][persisted_row_group_index]") {
	auto path = TestCreatePath("persisted_rg_index_stats.db");
	DeleteDatabase(path);
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads = 4"));
	REQUIRE_NO_FAIL(con.Query("ATTACH '" + path + "' AS p (STORAGE_VERSION 'latest', ROW_GROUP_SIZE 2048)"));
	// integers of several widths (bit-packed, FOR_SCALED, constant, all NULL), strings ('' rows, shared 8-byte
	// prefixes, unicode, a low-cardinality dictionary column), doubles, dates, timestamps, decimals, booleans,
	// hugeints, a list and a struct
	REQUIRE_NO_FAIL(con.Query(
	    "CREATE TABLE p.t AS SELECT i, (i * 1000)::BIGINT AS big, (i % 7)::TINYINT AS tiny, 42 AS k, "
	    "NULL::INTEGER AS z, CASE WHEN i % 5 = 0 THEN '' WHEN i % 11 = 0 THEN NULL "
	    "ELSE 'prefix00-' || (i % 977)::VARCHAR END AS s, "
	    "CASE WHEN i % 3 = 0 THEN 'é' || (i // 2048)::VARCHAR ELSE 'a' END AS u, "
	    "['x', 'y', 'z'][1 + i % 3] AS lowcard, i / 3.0 AS dbl, DATE '2000-01-01' + (i // 1000)::INTEGER AS dt, "
	    "TIMESTAMP '2000-01-01 00:00:00' + INTERVAL (i) SECOND AS ts, (i / 100)::DECIMAL(18, 2) AS dec, "
	    "i % 2 = 0 AS b, i::HUGEINT * 1000000000000 AS h, [i, i] AS l, {'a': i} AS st FROM range(143360) r(i)"));
	REQUIRE_NO_FAIL(con.Query("CHECKPOINT p"));
	REQUIRE_NO_FAIL(con.Query("DETACH p"));
	REQUIRE_NO_FAIL(con.Query("ATTACH '" + path + "' AS p"));

	REQUIRE_NO_FAIL(con.Query("BEGIN TRANSACTION"));
	{
		// every reference into the attached database (the tree, its row groups) ends here, before COMMIT and DETACH
		auto &table = Catalog::GetEntry<TableCatalogEntry>(*con.context, "p", DEFAULT_SCHEMA, "t");
		auto &collection = *table.GetStorage().GetRowGroupCollection();
		auto &types = collection.GetTypes();
		auto row_groups = collection.GetRowGroups();
		REQUIRE(row_groups->GetPersistedIndex());
		idx_t row_group_count = 0;
		idx_t compared = 0;
		for (auto &row_group : row_groups->Segments()) {
			row_group_count++;
			for (idx_t c = 0; c < types.size(); c++) {
				auto nested = types[c].id() == LogicalTypeId::LIST || types[c].id() == LogicalTypeId::STRUCT;
				REQUIRE(!row_group.IsColumnLoaded(c));
				auto persisted = row_group.GetPersistedStatistics(c);
				if (nested) {
					REQUIRE(!persisted);
					continue;
				}
				REQUIRE(persisted);
				// load the column: its statistics are now the loaded column's
				row_group.GetRawColumnData(c);
				REQUIRE(row_group.IsColumnLoaded(c));
				REQUIRE(!row_group.GetPersistedStatistics(c));
				auto loaded = row_group.GetStatistics(c);
				REQUIRE(loaded);
				REQUIRE(persisted->ToString() == loaded->ToString());
				REQUIRE(StatisticsBytes(*persisted) == StatisticsBytes(*loaded));
				compared++;
			}
		}
		// 70 row groups or more: loaded in parallel from their persisted positions (at least 64 remaining, 4 threads)
		REQUIRE(row_group_count >= 70);
		REQUIRE(compared == row_group_count * 14);
	}
	REQUIRE_NO_FAIL(con.Query("COMMIT"));
	REQUIRE_NO_FAIL(con.Query("DETACH p"));
	DeleteDatabase(path);
}
