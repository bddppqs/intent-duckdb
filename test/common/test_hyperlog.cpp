#include "catch.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/types/hyperloglog.hpp"

using namespace duckdb;
using namespace std;

TEST_CASE("Test that hyperloglog works", "[hyperloglog]") {
	HyperLogLog log;
	// add a million elements of the same value
	int x = 4;
	for (size_t i = 0; i < 1000000; i++) {
		log.InsertElement(Hash(x));
	}
	REQUIRE(log.Count() == 1);

	// now add a million different values
	HyperLogLog log2;
	for (size_t i = 0; i < 1000000; i++) {
		x = i;
		log2.InsertElement(Hash(x));
	}
	// the count is approximate, but should be pretty close to a million
	size_t count = log2.Count();
	REQUIRE(count > 950000LL);
	REQUIRE(count < 1050000LL);

	// now we can merge the HLLs
	log.Merge(log2);
	// the count should be pretty much the same
	count = log.Count();
	REQUIRE(count > 950000LL);
	REQUIRE(count < 1050000LL);

	// now test composability of the merge
	// add everything to one big_hll one
	// add chunks to small_hll ones and then merge them
	// the result should be the same
	HyperLogLog big_hll;
	HyperLogLog small_hll[16];
	for (size_t i = 0; i < 1000000; i++) {
		x = ((2 * i) + 3) % (i + 3 / 2);
		big_hll.InsertElement(Hash(x));
		small_hll[i % 16].InsertElement(Hash(x));
	}
	// now merge them into one big_hll HyperLogLog
	for (idx_t i = 1; i < 16; i++) {
		small_hll[0].Merge(small_hll[i]);
	}
	// the result should be identical to the big_hll one
	REQUIRE(small_hll[0].Count() == big_hll.Count());
}

TEST_CASE("Test different hyperloglog version serialization", "[hyperloglog]") {
	Allocator allocator;
	MemoryStream stream(allocator);
	SerializationOptions options;
	options.serialization_compatibility = SerializationCompatibility::FromString("v1.0.0");

	// Add 100M values to a NEW HyperLogLog
	HyperLogLog original_log;
	for (size_t i = 0; i < 100000000; i++) {
		original_log.InsertElement(Hash(i));

		switch (i + 1) {
		case 1:
		case 10:
		case 100:
		case 1000:
		case 10000:
		case 100000:
		case 1000000:
		case 10000000:
		case 100000000:
			break; // We roundtrip the serialization every order of magnitude
		default:
			continue;
		}

		// Grab the count
		const auto original_count = original_log.Count();

		// Serialize it as an OLD HyperLogLog
		stream.Rewind();
		BinarySerializer::Serialize(original_log, stream, options);

		// Deserialize it, creating a NEW HyperLogLog from the OLD one
		stream.Rewind();
		auto deserialized_log = BinaryDeserializer::Deserialize<HyperLogLog>(stream);

		// Verify that the deserialized count is equal
		const auto deserialized_count = deserialized_log->Count();
		REQUIRE(original_count == deserialized_count);
	}
}

TEST_CASE("CountZeros builtin equivalence", "[hyperloglog]") {
	// Preserve the original de Bruijn algorithms as independent reference paths.
	const auto reference_leading = [](uint64_t value) -> idx_t {
		if (!value) {
			return 64;
		}
		static constexpr uint64_t index[] = {0,  47, 1,  56, 48, 27, 2,  60, 57, 49, 41, 37, 28, 16, 3,  61,
		                                     54, 58, 35, 52, 50, 42, 21, 44, 38, 32, 29, 23, 17, 11, 4,  62,
		                                     46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43, 31, 22, 10, 45,
		                                     25, 39, 14, 33, 19, 30, 9,  24, 13, 18, 8,  12, 7,  6,  5,  63};
		value |= value >> 1;
		value |= value >> 2;
		value |= value >> 4;
		value |= value >> 8;
		value |= value >> 16;
		value |= value >> 32;
		return 63 - index[(value * 0X03F79D71B4CB0A89ULL) >> 58];
	};
	const auto reference_trailing = [](uint64_t value) -> idx_t {
		if (!value) {
			return 64;
		}
		static constexpr uint64_t index[] = {63, 0,  58, 1,  59, 47, 53, 2,  60, 39, 48, 27, 54, 33, 42, 3,
		                                     61, 51, 37, 40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4,
		                                     62, 57, 46, 52, 38, 26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21,
		                                     56, 45, 25, 31, 35, 16, 9,  12, 44, 24, 15, 8,  23, 7,  6,  5};
		return index[((value & -value) * 0x07EDD5E59A4E28C2ULL) >> 58];
	};
	const auto check = [&](uint64_t value) {
		CAPTURE(value);
		REQUIRE(CountZeros<uint64_t>::Leading(value) == reference_leading(value));
		REQUIRE(CountZeros<uint64_t>::Trailing(value) == reference_trailing(value));
	};

	check(0);
	check(~uint64_t(0));
	for (idx_t bit = 0; bit < 64; bit++) {
		check(uint64_t(1) << bit);
	}
	static constexpr idx_t positions[] = {0, 1, 5, 6, 57, 58, 62, 63};
	for (idx_t left = 0; left < 8; left++) {
		for (idx_t right = left + 1; right < 8; right++) {
			check((uint64_t(1) << positions[left]) | (uint64_t(1) << positions[right]));
		}
	}
	uint64_t state = 0xD021C0A17E202609ULL;
	for (idx_t i = 0; i < 1000000; i++) {
		state = state * 6364136223846793005ULL + 1442695040888963407ULL;
		check(state);
	}
}
