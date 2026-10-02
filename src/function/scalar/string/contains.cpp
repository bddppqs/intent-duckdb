#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/scalar/list_functions.hpp"
#include "duckdb/function/scalar/map_functions.hpp"
#include "duckdb/function/scalar/string_common.hpp"
#include "duckdb/function/scalar/string_functions.hpp"
#include "duckdb/function/scalar/struct_functions.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/tuning_defaults.hpp"
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define DUCKDB_CONTAINS_VECTOR_KERNEL 1
#endif
#if defined(__aarch64__) && !defined(__AARCH64EB__) && (defined(__GNUC__) || defined(__clang__))
#include <arm_neon.h>
#include <cstdlib>
#define DUCKDB_CONTAINS_NEON_KERNEL 1
#endif

namespace duckdb {

namespace {

struct ContainsAligned {
	template <class UNSIGNED>
	static bool MatchRemainder(const unsigned char *haystack, const unsigned char *needle, idx_t needle_size) {
		return true;
	}
};

struct ContainsUnaligned {
	template <class UNSIGNED>
	static bool MatchRemainder(const unsigned char *haystack, const unsigned char *needle, idx_t needle_size) {
		idx_t matches = 0;
		for (idx_t i = sizeof(UNSIGNED); i < needle_size; i++) {
			matches += haystack[i] == needle[i];
		}
		return matches == needle_size - sizeof(UNSIGNED);
	}
};

struct ContainsGeneric {
	template <class UNSIGNED>
	static bool MatchRemainder(const unsigned char *haystack, const unsigned char *needle, idx_t needle_size) {
		return memcmp(haystack + sizeof(UNSIGNED), needle + sizeof(UNSIGNED), needle_size - sizeof(UNSIGNED)) == 0;
	}
};

template <class UNSIGNED, class OP>
static idx_t Contains(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle,
                      idx_t needle_size, idx_t base_offset) {
	if (needle_size > haystack_size) {
		// needle is bigger than haystack: haystack cannot contain needle
		return DConstants::INVALID_INDEX;
	}
	haystack_size -= needle_size - 1;
	auto needle_entry = Load<UNSIGNED>(needle);
	for (idx_t offset = 0; offset < haystack_size; offset++) {
		// start off by performing a memchr to find the first character of the
		auto location =
		    static_cast<const unsigned char *>(memchr(haystack + offset, needle[0], haystack_size - offset));
		if (!location) {
			return DConstants::INVALID_INDEX;
		}
		// for this position we first compare the haystack with the needle
		offset = UnsafeNumericCast<idx_t>(location - haystack);
		auto haystack_entry = Load<UNSIGNED>(location);
		if (needle_entry == haystack_entry) {
			if (OP::template MatchRemainder<UNSIGNED>(location, needle, needle_size)) {
				return base_offset + offset;
			}
		}
	}
	return DConstants::INVALID_INDEX;
}

struct ContainsOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return FindStrInStr(left, right) != DConstants::INVALID_INDEX;
	}
};

#ifdef DUCKDB_CONTAINS_VECTOR_KERNEL
//! True when this CPU has AVX2; decided once at run time, so no compile flag and no ISA baseline changes
const bool contains_vector_supported = __builtin_cpu_supports("avx2");

//! The vector body is entered only for needles of at least two bytes in a haystack that holds a full 32-byte
//! window of candidate start positions, so both loads below stay inside the haystack.
inline bool ContainsUseVector(idx_t haystack_size, idx_t needle_size) {
	return contains_vector_supported && needle_size > 1 && haystack_size >= 32 + needle_size - 1;
}

//! First/last-needle-byte pair scan: broadcast needle[0] and needle[needle_size - 1], compare two 32-byte
//! windows offset by needle_size - 1, AND the masks, and verify each candidate with memcmp over the interior.
__attribute__((target("avx2"))) idx_t ContainsVector(const unsigned char *haystack, idx_t haystack_size,
                                                     const unsigned char *needle, idx_t needle_size) {
	const __m256i first = _mm256_set1_epi8(static_cast<char>(needle[0]));
	const idx_t last_offset = needle_size - 1;
	const __m256i last = _mm256_set1_epi8(static_cast<char>(needle[last_offset]));
	const idx_t limit = haystack_size - last_offset; // candidate start positions; >= 32 by ContainsUseVector
	idx_t offset = 0;
	while (offset < limit) {
		// the tail takes one final window that OVERLAPS the previous one instead of a byte loop: every position
		// below `offset` has already been excluded, so the lowest set bit of this mask is still the first match
		const idx_t at = (offset + 32 <= limit) ? offset : limit - 32;
		const __m256i block_first = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(haystack + at));
		const __m256i block_last = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(haystack + at + last_offset));
		auto mask = static_cast<uint32_t>(_mm256_movemask_epi8(
		    _mm256_and_si256(_mm256_cmpeq_epi8(block_first, first), _mm256_cmpeq_epi8(block_last, last))));
		while (mask) {
			const idx_t bit = static_cast<idx_t>(__builtin_ctz(mask));
			if (needle_size == 2 || memcmp(haystack + at + bit + 1, needle + 1, needle_size - 2) == 0) {
				return at + bit;
			}
			mask &= mask - 1;
		}
		offset = at + 32;
	}
	return DConstants::INVALID_INDEX;
}
#endif

#ifdef DUCKDB_CONTAINS_NEON_KERNEL
//! The arm64 twin of the AVX2 kernel above (Advanced SIMD is baseline on aarch64, so no run-time dispatch). With
//! kContainsNeonKernel false the memchr path below runs.
const bool contains_neon_enabled = kContainsNeonKernel;

//! The same entry rule as ContainsUseVector, for a 16-byte window.
inline bool ContainsUseNeon(idx_t haystack_size, idx_t needle_size) {
	return contains_neon_enabled && needle_size > 1 && haystack_size >= 16 + needle_size - 1;
}

//! ContainsVector with 16-byte windows: the two compare masks are ANDed and narrowed (shrn #4) to a 64-bit mask of
//! one nibble per position, kept to one bit per nibble so that ctz / 4 is the position. A candidate's interior is
//! compared inline rather than with memcmp: a call in the loop makes GCC keep both broadcast needle bytes on the
//! stack and reload them in every window (v8-v15 are callee-saved in their low 64 bits only).
idx_t ContainsNeon(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle, idx_t needle_size) {
	const uint8x16_t first = vdupq_n_u8(needle[0]);
	const idx_t last_offset = needle_size - 1;
	const uint8x16_t last = vdupq_n_u8(needle[last_offset]);
	const idx_t limit = haystack_size - last_offset; // candidate start positions; >= 16 by ContainsUseNeon
	idx_t offset = 0;
	while (offset < limit) {
		// the overlapping final window of ContainsVector: positions below `offset` are already excluded
		const idx_t at = (offset + 16 <= limit) ? offset : limit - 16;
		const uint8x16_t block_first = vld1q_u8(haystack + at);
		const uint8x16_t block_last = vld1q_u8(haystack + at + last_offset);
		const uint8x16_t both = vandq_u8(vceqq_u8(block_first, first), vceqq_u8(block_last, last));
		uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(both), 4)), 0);
		mask &= 0x8888888888888888ULL;
		while (mask) {
			const idx_t pos = static_cast<idx_t>(__builtin_ctzll(mask)) >> 2;
			const unsigned char *candidate = haystack + at + pos;
			idx_t i = 1;
			while (i < last_offset && candidate[i] == needle[i]) {
				i++;
			}
			if (i == last_offset) {
				return at + pos;
			}
			mask &= mask - 1;
		}
		offset = at + 16;
	}
	return DConstants::INVALID_INDEX;
}
#endif

} // namespace

idx_t FindStrInStr(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle, idx_t needle_size) {
	D_ASSERT(needle_size > 0);
#ifdef DUCKDB_CONTAINS_VECTOR_KERNEL
	if (ContainsUseVector(haystack_size, needle_size)) {
		return ContainsVector(haystack, haystack_size, needle, needle_size);
	}
#endif
#ifdef DUCKDB_CONTAINS_NEON_KERNEL
	if (ContainsUseNeon(haystack_size, needle_size)) {
		return ContainsNeon(haystack, haystack_size, needle, needle_size);
	}
#endif
	// start off by performing a memchr to find the first character of the
	auto location = memchr(haystack, needle[0], haystack_size);
	if (location == nullptr) {
		return DConstants::INVALID_INDEX;
	}
	idx_t base_offset = UnsafeNumericCast<idx_t>(const_uchar_ptr_cast(location) - haystack);
	haystack_size -= base_offset;
	haystack = const_uchar_ptr_cast(location);
	// switch algorithm depending on needle size
	switch (needle_size) {
	case 1:
		return base_offset;
	case 2:
		return Contains<uint16_t, ContainsAligned>(haystack, haystack_size, needle, 2, base_offset);
	case 3:
		return Contains<uint16_t, ContainsUnaligned>(haystack, haystack_size, needle, 3, base_offset);
	case 4:
		return Contains<uint32_t, ContainsAligned>(haystack, haystack_size, needle, 4, base_offset);
	case 5:
		return Contains<uint32_t, ContainsUnaligned>(haystack, haystack_size, needle, 5, base_offset);
	case 6:
		return Contains<uint32_t, ContainsUnaligned>(haystack, haystack_size, needle, 6, base_offset);
	case 7:
		return Contains<uint32_t, ContainsUnaligned>(haystack, haystack_size, needle, 7, base_offset);
	case 8:
		return Contains<uint64_t, ContainsAligned>(haystack, haystack_size, needle, 8, base_offset);
	default:
		return Contains<uint64_t, ContainsGeneric>(haystack, haystack_size, needle, needle_size, base_offset);
	}
}

idx_t FindStrInStr(const string_t &haystack_s, const string_t &needle_s) {
	auto haystack = const_uchar_ptr_cast(haystack_s.GetData());
	auto haystack_size = haystack_s.GetSize();
	auto needle = const_uchar_ptr_cast(needle_s.GetData());
	auto needle_size = needle_s.GetSize();
	if (needle_size == 0) {
		// empty needle: always true
		return 0;
	}
	return FindStrInStr(haystack, haystack_size, needle, needle_size);
}

ScalarFunction GetStringContains() {
	ScalarFunction string_fun("contains", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                          ScalarFunction::BinaryFunction<string_t, string_t, bool, ContainsOperator>);
	string_fun.SetCollationHandling(FunctionCollationHandling::PUSH_COMBINABLE_COLLATIONS);
	return string_fun;
}

ScalarFunctionSet ContainsFun::GetFunctions() {
	auto string_fun = GetStringContains();
	auto list_fun = ListContainsFun::GetFunction();
	auto map_fun = MapContainsFun::GetFunction();
	auto struct_fun = StructContainsFun::GetFunction();
	ScalarFunctionSet set("contains");
	set.AddFunction(string_fun);
	set.AddFunction(list_fun);
	set.AddFunction(map_fun);
	set.AddFunction(struct_fun);
	return set;
}

} // namespace duckdb
