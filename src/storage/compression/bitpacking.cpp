#include "duckdb/common/bitpacking.hpp"

#include "duckdb/common/limits.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/operator/add.hpp"
#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb/common/operator/multiply.hpp"
#include "duckdb/common/operator/subtract.hpp"
#include "duckdb/common/tuning_defaults.hpp"
#include "duckdb/function/compression/compression.hpp"
#include "duckdb/function/compression_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/compression/bitpacking.hpp"
#include "duckdb/storage/single_file_block_manager.hpp"
#include "duckdb/storage/table/column_data_checkpointer.hpp"
#include "duckdb/storage/table/column_segment.hpp"
#include "duckdb/storage/table/scan_state.hpp"

#include <functional>
#include <type_traits>

namespace duckdb {

constexpr const idx_t BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;
static constexpr const idx_t BITPACKING_METADATA_GROUP_SIZE = STANDARD_VECTOR_SIZE > 512 ? STANDARD_VECTOR_SIZE : 2048;

typedef struct {
	BitpackingMode mode;
	uint32_t offset;
} bitpacking_metadata_t;

typedef uint32_t bitpacking_metadata_encoded_t;

static bitpacking_metadata_encoded_t EncodeMeta(bitpacking_metadata_t metadata) {
	D_ASSERT(metadata.offset <= 0x00FFFFFF); // max uint24_t
	bitpacking_metadata_encoded_t encoded_value = metadata.offset;
	encoded_value |= UnsafeNumericCast<bitpacking_metadata_encoded_t>((uint8_t)metadata.mode << 24);
	return encoded_value;
}
static bitpacking_metadata_t DecodeMeta(bitpacking_metadata_encoded_t *metadata_encoded) {
	bitpacking_metadata_t metadata;
	metadata.mode = static_cast<BitpackingMode>((*metadata_encoded >> 24) & 0xFF);
	metadata.offset = *metadata_encoded & 0x00FFFFFF;
	return metadata;
}

//===--------------------------------------------------------------------===//
// FOR_SCALED (kScaledFrameOfReference)
//===--------------------------------------------------------------------===//
//! A FOR group whose valid values' offsets from the minimum share a greatest common divisor d > 1 stores offset / d,
//! and a scan computes offset * d + minimum. Integer types up to 64 bits; the arithmetic is unsigned 64-bit, wrapping
//! to T.
static bool ScaledFrameOfReferenceAllowed(BlockManager &block_manager) {
	if (!kScaledFrameOfReference) {
		return false;
	}
	auto single_file = dynamic_cast<SingleFileBlockManager *>(&block_manager);
	return single_file && single_file->ScaledFrameOfReferenceFile();
}

template <class T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
static uint64_t ScaledOffset(T value, T minimum) {
	using T_U = typename MakeUnsigned<T>::type;
	return static_cast<T_U>(static_cast<T_U>(value) - static_cast<T_U>(minimum));
}

template <class T, typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
static uint64_t ScaledOffset(T value, T minimum) {
	return 0;
}

//! The greatest common divisor of the valid values' offsets from the minimum (0 when every offset is 0)
template <class T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
static uint64_t ScaledFrameOfReferenceDivisor(const T *values, const bool *validity, idx_t count, T minimum) {
	uint64_t divisor = 0;
	for (idx_t i = 0; i < count && divisor != 1; i++) {
		if (!validity[i]) {
			continue;
		}
		uint64_t offset = ScaledOffset<T>(values[i], minimum);
		if (divisor != 0 && offset % divisor == 0) {
			continue;
		}
		while (offset != 0) {
			uint64_t remainder = divisor % offset;
			divisor = offset;
			offset = remainder;
		}
	}
	return divisor;
}

template <class T, typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
static uint64_t ScaledFrameOfReferenceDivisor(const T *values, const bool *validity, idx_t count, T minimum) {
	return 1;
}

//! Replaces each value with its offset from the minimum divided by the divisor (0 for a NULL)
template <class T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
static void DivideFrameOfReference(T *values, const bool *validity, idx_t count, T minimum, uint64_t divisor) {
	for (idx_t i = 0; i < count; i++) {
		values[i] = validity[i] ? static_cast<T>(ScaledOffset<T>(values[i], minimum) / divisor) : T(0);
	}
}

template <class T, typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
static void DivideFrameOfReference(T *values, const bool *validity, idx_t count, T minimum, uint64_t divisor) {
	throw InternalException("FOR_SCALED is not written for this type");
}

template <class T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
static void ApplyScaledFrameOfReference(T *dst, T frame_of_reference, T divisor, idx_t size) {
	using T_U = typename MakeUnsigned<T>::type;
	const uint64_t scale = static_cast<T_U>(divisor);
	const uint64_t base = static_cast<T_U>(frame_of_reference);
	for (idx_t i = 0; i < size; i++) {
		dst[i] = static_cast<T>(static_cast<uint64_t>(static_cast<T_U>(dst[i])) * scale + base);
	}
}

template <class T, typename std::enable_if<!std::is_integral<T>::value, int>::type = 0>
static void ApplyScaledFrameOfReference(T *dst, T frame_of_reference, T divisor, idx_t size) {
	throw InternalException("Invalid bitpacking mode");
}

//===--------------------------------------------------------------------===//
// PATCHED_FOR (kPatchedForBitpacking)
//===--------------------------------------------------------------------===//
//! A group whose valid values' offsets from the minimum, divided by their common divisor d (FOR_SCALED's, 1 without
//! one), mostly fall in a window [lo, lo + 2^b) narrower than FOR's width stores: base = minimum + lo * d, b and d (T
//! each), the exception count and a pad (u16 each), each exception's position in the group (u16) and raw value (T),
//! then the in-window offsets minus lo packed at width b (0 for an exception or a NULL). A scan computes
//! packed * d + base and writes the exceptions back. At most 1 % of a group's values are exceptions.
static constexpr idx_t BITPACKING_PATCHED_FOR_MAX_EXCEPTIONS = BITPACKING_METADATA_GROUP_SIZE / 100;

static bool PatchedForAllowed(BlockManager &block_manager) {
	if (!kPatchedForBitpacking) {
		return false;
	}
	auto single_file = dynamic_cast<SingleFileBlockManager *>(&block_manager);
	return single_file && single_file->PatchedForFile();
}

//! The bytes of a PATCHED_FOR group: base, width and divisor, the exception count and pad, the exceptions, the packing
template <class T>
static idx_t PatchedForBytes(idx_t count, bitpacking_width_t width, idx_t exceptions) {
	return 3 * sizeof(T) + 2 * sizeof(uint16_t) + exceptions * (sizeof(uint16_t) + sizeof(T)) +
	       BitpackingPrimitives::GetRequiredSize(count, width);
}

//! The bit width of an unsigned offset, rounded as BitpackingPrimitives rounds a T width (a width within sizeof(T) bits
//! of the type's is the type's)
template <class T>
static bitpacking_width_t PatchedForWidth(uint64_t value) {
	bitpacking_width_t width = 0;
	while (value) {
		width++;
		value >>= 1;
	}
	const bitpacking_width_t bits_of_type = sizeof(T) * 8;
	return width + sizeof(T) > bits_of_type ? bits_of_type : width;
}

struct EmptyBitpackingWriter {
	template <class T>
	static void WriteConstant(T constant, idx_t count, void *data_ptr, bool all_invalid) {
	}
	template <class T, class T_S = typename MakeSigned<T>::type>
	static void WriteConstantDelta(T_S constant, T frame_of_reference, idx_t count, T *values, bool *validity,
	                               void *data_ptr) {
	}
	template <class T, class T_S = typename MakeSigned<T>::type>
	static void WriteDeltaFor(T *values, bool *validity, bitpacking_width_t width, T frame_of_reference,
	                          T_S delta_offset, T *original_values, idx_t count, void *data_ptr) {
	}
	template <class T>
	static void WriteFor(T *values, bool *validity, bitpacking_width_t width, T frame_of_reference, idx_t count,
	                     void *data_ptr) {
	}
	template <class T>
	static void WriteForScaled(T *values, bool *validity, bitpacking_width_t width, T frame_of_reference, T divisor,
	                           idx_t count, void *data_ptr) {
	}
	template <class T>
	static void WritePatchedFor(T *values, bitpacking_width_t width, T base, T divisor, idx_t count,
	                            const uint16_t *positions, const T *exceptions, idx_t exception_count, void *data_ptr) {
	}
};

template <class T, class T_S = typename MakeSigned<T>::type>
struct BitpackingState {
public:
	BitpackingState() : compression_buffer_idx(0), total_size(0), data_ptr(nullptr) {
		compression_buffer_internal[0] = T(0);
		compression_buffer = &compression_buffer_internal[1];
		Reset();
	}

	// Extra val for delta encoding
	T compression_buffer_internal[BITPACKING_METADATA_GROUP_SIZE + 1];
	T *compression_buffer;
	T_S delta_buffer[BITPACKING_METADATA_GROUP_SIZE];
	bool compression_buffer_validity[BITPACKING_METADATA_GROUP_SIZE];
	idx_t compression_buffer_idx;
	idx_t total_size;

	// Used to pass CompressionState ptr through the Bitpacking writer
	void *data_ptr;

	// Stats on current compression buffer
	T minimum;
	T maximum;
	T min_max_diff;
	T_S minimum_delta;
	T_S maximum_delta;
	T_S min_max_delta_diff;
	T_S delta_offset;
	bool all_valid;
	bool all_invalid;

	bool has_valid;
	bool has_invalid;

	bool can_do_delta;
	bool can_do_for;

	// Used to force a specific mode, useful in testing
	BitpackingMode mode = BitpackingMode::AUTO;
	// FOR may be written as FOR_SCALED (ScaledFrameOfReferenceAllowed)
	bool allow_scaled = false;
	// a non-constant group may be written as PATCHED_FOR (PatchedForAllowed)
	bool allow_patched_for = false;
	// PATCHED_FOR's working set: the exceptions of the group it writes
	uint16_t patched_positions[BITPACKING_PATCHED_FOR_MAX_EXCEPTIONS];
	T patched_exceptions[BITPACKING_PATCHED_FOR_MAX_EXCEPTIONS];

public:
	void Reset() {
		minimum = NumericLimits<T>::Maximum();
		minimum_delta = NumericLimits<T_S>::Maximum();
		maximum = NumericLimits<T>::Minimum();
		maximum_delta = NumericLimits<T_S>::Minimum();
		delta_offset = 0;
		all_valid = true;
		all_invalid = true;
		has_valid = false;
		has_invalid = false;
		can_do_delta = false;
		can_do_for = false;
		compression_buffer_idx = 0;
		min_max_diff = 0;
		min_max_delta_diff = 0;
	}

	void CalculateFORStats() {
		can_do_for = TrySubtractOperator::Operation(maximum, minimum, min_max_diff);
	}

	void CalculateDeltaStats() {
		// TODO: currently we dont support delta compression of values above NumericLimits<T_S>::Maximum(),
		// 		 we could support this with some clever substract trickery?
		if (maximum > static_cast<T>(NumericLimits<T_S>::Maximum())) {
			return;
		}

		// Don't delta encoding 1 value makes no sense
		if (compression_buffer_idx < 2) {
			return;
		}

		// TODO: handle NULLS here?
		// Currently we cannot handle nulls because we would need an additional step of patching for this.
		// we could for example copy the last value on a null insert. This would help a bit, but not be optimal for
		// large deltas since theres suddenly a zero then. Ideally we would insert a value that leads to a delta within
		// the current domain of deltas however we dont know that domain here yet
		if (!all_valid) {
			return;
		}

		// Note: since we dont allow any values over NumericLimits<T_S>::Maximum(), all subtractions for unsigned types
		// are guaranteed not to overflow
		bool can_do_all = true;
		if (NumericLimits<T>::IsSigned()) {
			T_S bogus;
			can_do_all = TrySubtractOperator::Operation(static_cast<T_S>(minimum), static_cast<T_S>(maximum), bogus) &&
			             TrySubtractOperator::Operation(static_cast<T_S>(maximum), static_cast<T_S>(minimum), bogus);
		}

		// Calculate delta's
		// compression_buffer pointer points one element ahead of the internal buffer making the use of signed index
		// integer (-1) possible
		D_ASSERT(compression_buffer_idx <= NumericLimits<int64_t>::Maximum());
		if (can_do_all) {
			for (int64_t i = 0; i < static_cast<int64_t>(compression_buffer_idx); i++) {
				delta_buffer[i] = static_cast<T_S>(compression_buffer[i]) - static_cast<T_S>(compression_buffer[i - 1]);
			}
		} else {
			for (int64_t i = 0; i < static_cast<int64_t>(compression_buffer_idx); i++) {
				auto success =
				    TrySubtractOperator::Operation(static_cast<T_S>(compression_buffer[i]),
				                                   static_cast<T_S>(compression_buffer[i - 1]), delta_buffer[i]);
				if (!success) {
					return;
				}
			}
		}

		can_do_delta = true;

		for (idx_t i = 1; i < compression_buffer_idx; i++) {
			maximum_delta = MaxValue<T_S>(maximum_delta, delta_buffer[i]);
			minimum_delta = MinValue<T_S>(minimum_delta, delta_buffer[i]);
		}

		// Since we can set the first value arbitrarily, we want to pick one from the current domain, note that
		// we will store the original first value - this offset as the  delta_offset to be able to decode this again.
		delta_buffer[0] = minimum_delta;

		can_do_delta = can_do_delta && TrySubtractOperator::Operation(maximum_delta, minimum_delta, min_max_delta_diff);
		can_do_delta = can_do_delta && TrySubtractOperator::Operation(static_cast<T_S>(compression_buffer[0]),
		                                                              minimum_delta, delta_offset);
	}

	template <class T_INNER>
	void SubtractFrameOfReference(T_INNER *buffer, T_INNER frame_of_reference) {
		static_assert(NumericLimits<T_INNER>::IsIntegral(), "Integral type required.");

		using T_U = typename MakeUnsigned<T_INNER>::type;

		for (idx_t i = 0; i < compression_buffer_idx; i++) {
			reinterpret_cast<T_U *>(buffer)[i] -= static_cast<T_U>(frame_of_reference);
		}
	}

	//! Writes the group as PATCHED_FOR when its bytes are fewer than stock_bytes, the stock mode's (or when the mode is
	//! forced and a layout exists): the window and width minimising the group's bytes over every exception set of at
	//! most BITPACKING_PATCHED_FOR_MAX_EXCEPTIONS values, which a window leaves as some of the least and some of the
	//! greatest offsets. The size the analyze reports stays the stock mode's (stock_total), so the choice between
	//! bit-packing and the other compressions is the stock one. Must run before the group's values are changed in place
	template <class OP, class T_INNER = T, typename std::enable_if<std::is_integral<T_INNER>::value, int>::type = 0>
	bool TryPatchedFor(idx_t stock_bytes, idx_t stock_total) {
		if (!allow_patched_for || (mode != BitpackingMode::AUTO && mode != BitpackingMode::PATCHED_FOR)) {
			return false;
		}
		using T_U = typename MakeUnsigned<T>::type;
		static constexpr idx_t CAP = BITPACKING_PATCHED_FOR_MAX_EXCEPTIONS;
		const idx_t count = compression_buffer_idx;
		const uint64_t divisor =
		    ScaledFrameOfReferenceDivisor<T>(compression_buffer, compression_buffer_validity, count, minimum);
		if (divisor == 0) {
			// every valid value is the minimum
			return false;
		}
		// the CAP + 1 least offsets ascending and the CAP + 1 greatest descending, in one pass
		uint64_t least[CAP + 1];
		uint64_t greatest[CAP + 1];
		idx_t valid = 0;
		idx_t k = 0;
		for (idx_t i = 0; i < count; i++) {
			if (!compression_buffer_validity[i]) {
				continue;
			}
			const uint64_t offset = ScaledOffset<T>(compression_buffer[i], minimum) / divisor;
			valid++;
			if (k < CAP + 1) {
				idx_t l = k, g = k;
				for (; l > 0 && least[l - 1] > offset; l--) {
					least[l] = least[l - 1];
				}
				least[l] = offset;
				for (; g > 0 && greatest[g - 1] < offset; g--) {
					greatest[g] = greatest[g - 1];
				}
				greatest[g] = offset;
				k++;
				continue;
			}
			if (offset < least[CAP]) {
				idx_t l = CAP;
				for (; l > 0 && least[l - 1] > offset; l--) {
					least[l] = least[l - 1];
				}
				least[l] = offset;
			}
			if (offset > greatest[CAP]) {
				idx_t g = CAP;
				for (; g > 0 && greatest[g - 1] < offset; g--) {
					greatest[g] = greatest[g - 1];
				}
				greatest[g] = offset;
			}
		}
		if (valid <= 1) {
			return false;
		}
		const auto for_width = PatchedForWidth<T>(greatest[0] - least[0]);
		idx_t best_bytes = NumericLimits<idx_t>::Maximum();
		uint64_t best_low = 0;
		bitpacking_width_t best_width = 0;
		for (idx_t below = 0; below < k; below++) {
			for (idx_t above = 0; below + above < k; above++) {
				const idx_t exceptions = below + above;
				if (exceptions == 0 || exceptions > CAP || exceptions >= valid || greatest[above] < least[below]) {
					continue;
				}
				const auto width = PatchedForWidth<T>(greatest[above] - least[below]);
				if (width >= for_width) {
					continue;
				}
				const auto bytes = PatchedForBytes<T>(count, width, exceptions);
				if (bytes < best_bytes) {
					best_bytes = bytes;
					best_low = least[below];
					best_width = width;
				}
			}
		}
		if (best_bytes == NumericLimits<idx_t>::Maximum()) {
			return false;
		}
		// the window [best_low, best_low + 2^best_width) holds at least the counted inliers: count the exceptions
		const uint64_t span_mask = best_width >= 64 ? ~uint64_t(0) : (uint64_t(1) << best_width) - 1;
		idx_t exception_count = 0;
		for (idx_t i = 0; i < count; i++) {
			if (!compression_buffer_validity[i]) {
				continue;
			}
			const uint64_t offset = ScaledOffset<T>(compression_buffer[i], minimum) / divisor;
			if (offset < best_low || ((offset - best_low) & ~span_mask) != 0) {
				exception_count++;
			}
		}
		if (exception_count == 0 || exception_count > CAP) {
			return false;
		}
		const auto bytes = PatchedForBytes<T>(count, best_width, exception_count);
		if (mode != BitpackingMode::PATCHED_FOR && bytes >= stock_bytes) {
			return false;
		}
		// the packed values (offset - lo, or 0) in place, the exceptions' positions and raw values aside
		idx_t e = 0;
		for (idx_t i = 0; i < count; i++) {
			if (!compression_buffer_validity[i]) {
				compression_buffer[i] = T(0);
				continue;
			}
			const uint64_t offset = ScaledOffset<T>(compression_buffer[i], minimum) / divisor;
			if (offset < best_low || ((offset - best_low) & ~span_mask) != 0) {
				patched_positions[e] = UnsafeNumericCast<uint16_t>(i);
				patched_exceptions[e] = compression_buffer[i];
				e++;
				compression_buffer[i] = T(0);
			} else {
				compression_buffer[i] = static_cast<T>(static_cast<T_U>(offset - best_low));
			}
		}
		const auto base_offset = static_cast<T_U>(best_low * divisor);
		const T base = static_cast<T>(static_cast<T_U>(static_cast<T_U>(minimum) + base_offset));
		OP::WritePatchedFor(compression_buffer, best_width, base, static_cast<T>(divisor), count, patched_positions,
		                    patched_exceptions, exception_count, data_ptr);
		total_size += stock_total;
		return true;
	}

	template <class OP, class T_INNER = T, typename std::enable_if<!std::is_integral<T_INNER>::value, int>::type = 0>
	bool TryPatchedFor(idx_t stock_bytes, idx_t stock_total) {
		return false;
	}

	template <class OP>
	bool Flush() {
		if (compression_buffer_idx == 0) {
			return true;
		}

		if ((all_invalid || maximum == minimum) && (mode == BitpackingMode::AUTO || mode == BitpackingMode::CONSTANT)) {
			OP::WriteConstant(maximum, compression_buffer_idx, data_ptr, all_invalid);
			total_size += sizeof(T) + sizeof(bitpacking_metadata_encoded_t);
			return true;
		}

		CalculateFORStats();
		CalculateDeltaStats();

		// FOR_SCALED replaces FOR when its packed values and the divisor take fewer bytes
		uint64_t for_divisor = 1;
		bitpacking_width_t scaled_width = 0;
		if (allow_scaled && mode == BitpackingMode::AUTO && can_do_for) {
			for_divisor = ScaledFrameOfReferenceDivisor<T>(compression_buffer, compression_buffer_validity,
			                                               compression_buffer_idx, minimum);
			if (for_divisor > 1) {
				scaled_width = BitpackingPrimitives::MinimumBitWidth<T, false>(
				    static_cast<T>(ScaledOffset<T>(maximum, minimum) / for_divisor));
				auto for_width = BitpackingPrimitives::MinimumBitWidth<T, false>(min_max_diff);
				if (BitpackingPrimitives::GetRequiredSize(compression_buffer_idx, scaled_width) + sizeof(T) >=
				    BitpackingPrimitives::GetRequiredSize(compression_buffer_idx, for_width)) {
					for_divisor = 1;
				}
			} else {
				for_divisor = 1;
			}
		}

		if (can_do_delta) {
			if (maximum_delta == minimum_delta && mode != BitpackingMode::FOR && mode != BitpackingMode::DELTA_FOR) {
				// FOR needs to be T (considering hugeint is bigger than idx_t)
				T frame_of_reference = compression_buffer[0];

				OP::WriteConstantDelta(maximum_delta, static_cast<T>(frame_of_reference), compression_buffer_idx,
				                       compression_buffer, compression_buffer_validity, data_ptr);
				total_size += sizeof(T) + sizeof(T) + sizeof(bitpacking_metadata_encoded_t);
				return true;
			}

			// Check if delta has benefit
			auto delta_required_bitwidth =
			    BitpackingPrimitives::MinimumBitWidth<T, false>(static_cast<T>(min_max_delta_diff));
			auto regular_required_bitwidth = BitpackingPrimitives::MinimumBitWidth(min_max_diff);

			//! `min_max_diff` is uninitialized if `can_do_for` isn't true
			bool prefer_for = can_do_for && delta_required_bitwidth >= regular_required_bitwidth;
			if (for_divisor > 1) {
				prefer_for = delta_required_bitwidth >= scaled_width;
			}

			if (!prefer_for && mode != BitpackingMode::FOR) {
				auto delta_bytes =
				    BitpackingPrimitives::GetRequiredSize(compression_buffer_idx, delta_required_bitwidth);
				if (TryPatchedFor<OP>(3 * sizeof(T) + delta_bytes,
				                      2 * sizeof(T) + AlignValue(sizeof(bitpacking_width_t)) + delta_bytes)) {
					return true;
				}
				SubtractFrameOfReference(delta_buffer, minimum_delta);

				OP::WriteDeltaFor(reinterpret_cast<T *>(delta_buffer), compression_buffer_validity,
				                  delta_required_bitwidth, static_cast<T>(minimum_delta), delta_offset,
				                  compression_buffer, compression_buffer_idx, data_ptr);

				// FOR (frame of reference).
				total_size += sizeof(T);
				// Aligned bitpacking width.
				total_size += AlignValue(sizeof(bitpacking_width_t));
				// Delta offset.
				total_size += sizeof(T);
				// Compressed data size.
				total_size += BitpackingPrimitives::GetRequiredSize(compression_buffer_idx, delta_required_bitwidth);

				return true;
			}
		}

		if (can_do_for && for_divisor > 1) {
			auto scaled_bytes = BitpackingPrimitives::GetRequiredSize(compression_buffer_idx, scaled_width);
			if (TryPatchedFor<OP>(3 * sizeof(T) + scaled_bytes,
			                      2 * sizeof(T) + AlignValue(sizeof(bitpacking_width_t)) + scaled_bytes)) {
				return true;
			}
			DivideFrameOfReference(compression_buffer, compression_buffer_validity, compression_buffer_idx, minimum,
			                       for_divisor);
			OP::WriteForScaled(compression_buffer, compression_buffer_validity, scaled_width, minimum,
			                   static_cast<T>(for_divisor), compression_buffer_idx, data_ptr);

			total_size += BitpackingPrimitives::GetRequiredSize(compression_buffer_idx, scaled_width);
			total_size += sizeof(T); // FOR value
			total_size += AlignValue(sizeof(bitpacking_width_t));
			total_size += sizeof(T); // divisor

			return true;
		}

		if (can_do_for) {
			auto width = BitpackingPrimitives::MinimumBitWidth<T, false>(min_max_diff);
			auto for_bytes = BitpackingPrimitives::GetRequiredSize(compression_buffer_idx, width);
			if (TryPatchedFor<OP>(2 * sizeof(T) + for_bytes,
			                      sizeof(T) + AlignValue(sizeof(bitpacking_width_t)) + for_bytes)) {
				return true;
			}
			SubtractFrameOfReference(compression_buffer, minimum);
			OP::WriteFor(compression_buffer, compression_buffer_validity, width, minimum, compression_buffer_idx,
			             data_ptr);

			total_size += BitpackingPrimitives::GetRequiredSize(compression_buffer_idx, width);
			total_size += sizeof(T); // FOR value
			total_size += AlignValue(sizeof(bitpacking_width_t));

			return true;
		}

		return false;
	}

	template <class OP = EmptyBitpackingWriter>
	bool Update(T value, bool is_valid) {
		compression_buffer_validity[compression_buffer_idx] = is_valid;
		has_valid = has_valid || is_valid;
		has_invalid = has_invalid || !is_valid;
		all_valid = all_valid && is_valid;
		all_invalid = all_invalid && !is_valid;

		if (is_valid) {
			compression_buffer[compression_buffer_idx] = value;
			minimum = MinValue<T>(minimum, value);
			maximum = MaxValue<T>(maximum, value);
		}

		compression_buffer_idx++;

		if (compression_buffer_idx == BITPACKING_METADATA_GROUP_SIZE) {
			bool success = Flush<OP>();
			Reset();
			return success;
		}
		return true;
	}
};

//===--------------------------------------------------------------------===//
// Analyze
//===--------------------------------------------------------------------===//
template <class T>
struct BitpackingAnalyzeState : public AnalyzeState {
	explicit BitpackingAnalyzeState(const CompressionInfo &info) : AnalyzeState(info) {};
	BitpackingState<T> state;
};

template <class T>
unique_ptr<AnalyzeState> BitpackingInitAnalyze(ColumnData &col_data, PhysicalType type) {
	CompressionInfo info(col_data.GetBlockManager());
	auto state = make_uniq<BitpackingAnalyzeState<T>>(info);
	state->state.mode = Settings::Get<ForceBitpackingModeSetting>(col_data.GetDatabase());
	state->state.allow_scaled = ScaledFrameOfReferenceAllowed(col_data.GetBlockManager());
	state->state.allow_patched_for = PatchedForAllowed(col_data.GetBlockManager());

	return std::move(state);
}

template <class T>
bool BitpackingAnalyze(AnalyzeState &state, Vector &input, idx_t count) {
	// We use BITPACKING_METADATA_GROUP_SIZE tuples, which can exceed the block size.
	// In that case, we disable bitpacking.
	// we are conservative here by multiplying by 2
	auto type_size = GetTypeIdSize(input.GetType().InternalType());
	if (type_size * BITPACKING_METADATA_GROUP_SIZE * 2 > state.info.GetBlockSize()) {
		return false;
	}

	auto &analyze_state = state.Cast<BitpackingAnalyzeState<T>>();
	UnifiedVectorFormat vdata;
	input.ToUnifiedFormat(count, vdata);

	auto data = UnifiedVectorFormat::GetData<T>(vdata);
	for (idx_t i = 0; i < count; i++) {
		auto idx = vdata.sel->get_index(i);
		if (!analyze_state.state.template Update<EmptyBitpackingWriter>(data[idx], vdata.validity.RowIsValid(idx))) {
			return false;
		}
	}
	return true;
}

template <class T>
idx_t BitpackingFinalAnalyze(AnalyzeState &state) {
	auto &bitpacking_state = state.Cast<BitpackingAnalyzeState<T>>();
	auto flush_result = bitpacking_state.state.template Flush<EmptyBitpackingWriter>();
	if (!flush_result) {
		return DConstants::INVALID_INDEX;
	}
	return bitpacking_state.state.total_size;
}

//===--------------------------------------------------------------------===//
// Compress
//===--------------------------------------------------------------------===//
template <class T, bool WRITE_STATISTICS, class T_S = typename MakeSigned<T>::type>
struct BitpackingCompressionState : public CompressionState {
public:
	explicit BitpackingCompressionState(ColumnDataCheckpointData &checkpoint_data, const CompressionInfo &info)
	    : CompressionState(info), checkpoint_data(checkpoint_data),
	      function(checkpoint_data.GetCompressionFunction(CompressionType::COMPRESSION_BITPACKING)) {
		CreateEmptySegment();

		state.data_ptr = reinterpret_cast<void *>(this);
		state.mode = Settings::Get<ForceBitpackingModeSetting>(checkpoint_data.GetDatabase());
		state.allow_scaled = ScaledFrameOfReferenceAllowed(info.GetBlockManager());
		state.allow_patched_for = PatchedForAllowed(info.GetBlockManager());
	}

	ColumnDataCheckpointData &checkpoint_data;
	const CompressionFunction &function;
	unique_ptr<ColumnSegment> current_segment;
	BufferHandle handle;

	// Ptr to next free spot in segment;
	data_ptr_t data_ptr;
	// Ptr to next free spot for storing bitwidths and frame-of-references (growing downwards).
	data_ptr_t metadata_ptr;

	BitpackingState<T> state;

public:
	struct BitpackingWriter {
		static void WriteConstant(T constant, idx_t count, void *data_ptr, bool all_invalid) {
			auto state = reinterpret_cast<BitpackingCompressionState<T, WRITE_STATISTICS> *>(data_ptr);

			ReserveSpace(state, sizeof(T));
			WriteMetaData(state, BitpackingMode::CONSTANT);
			WriteData(state->data_ptr, constant);

			UpdateStats(state, count);
		}

		static void WriteConstantDelta(T_S constant, T frame_of_reference, idx_t count, T *values, bool *validity,
		                               void *data_ptr) {
			auto state = reinterpret_cast<BitpackingCompressionState<T, WRITE_STATISTICS> *>(data_ptr);

			ReserveSpace(state, 2 * sizeof(T));
			WriteMetaData(state, BitpackingMode::CONSTANT_DELTA);
			WriteData(state->data_ptr, frame_of_reference);
			WriteData(state->data_ptr, constant);

			UpdateStats(state, count);
		}
		static void WriteDeltaFor(T *values, bool *validity, bitpacking_width_t width, T frame_of_reference,
		                          T_S delta_offset, T *original_values, idx_t count, void *data_ptr) {
			auto state = reinterpret_cast<BitpackingCompressionState<T, WRITE_STATISTICS> *>(data_ptr);

			auto bp_size = BitpackingPrimitives::GetRequiredSize(count, width);
			ReserveSpace(state, bp_size + 3 * sizeof(T));

			WriteMetaData(state, BitpackingMode::DELTA_FOR);
			WriteData(state->data_ptr, frame_of_reference);
			WriteData(state->data_ptr, static_cast<T>(width));
			WriteData(state->data_ptr, delta_offset);

			BitpackingPrimitives::PackBuffer<T, false>(state->data_ptr, values, count, width);
			state->data_ptr += bp_size;

			UpdateStats(state, count);
		}

		static void WriteFor(T *values, bool *validity, bitpacking_width_t width, T frame_of_reference, idx_t count,
		                     void *data_ptr) {
			auto state = reinterpret_cast<BitpackingCompressionState<T, WRITE_STATISTICS> *>(data_ptr);

			auto bp_size = BitpackingPrimitives::GetRequiredSize(count, width);
			ReserveSpace(state, bp_size + 2 * sizeof(T));

			WriteMetaData(state, BitpackingMode::FOR);
			WriteData(state->data_ptr, frame_of_reference);
			WriteData(state->data_ptr, (T)width);

			BitpackingPrimitives::PackBuffer<T, false>(state->data_ptr, values, count, width);
			state->data_ptr += bp_size;

			UpdateStats(state, count);
		}

		static void WriteForScaled(T *values, bool *validity, bitpacking_width_t width, T frame_of_reference, T divisor,
		                           idx_t count, void *data_ptr) {
			auto state = reinterpret_cast<BitpackingCompressionState<T, WRITE_STATISTICS> *>(data_ptr);

			auto bp_size = BitpackingPrimitives::GetRequiredSize(count, width);
			ReserveSpace(state, bp_size + 3 * sizeof(T));

			WriteMetaData(state, BitpackingMode::FOR_SCALED);
			WriteData(state->data_ptr, frame_of_reference);
			WriteData(state->data_ptr, (T)width);
			WriteData(state->data_ptr, divisor);

			BitpackingPrimitives::PackBuffer<T, false>(state->data_ptr, values, count, width);
			state->data_ptr += bp_size;

			UpdateStats(state, count);
		}

		static void WritePatchedFor(T *values, bitpacking_width_t width, T base, T divisor, idx_t count,
		                            const uint16_t *positions, const T *exceptions, idx_t exception_count,
		                            void *data_ptr) {
			auto state = reinterpret_cast<BitpackingCompressionState<T, WRITE_STATISTICS> *>(data_ptr);

			auto bp_size = BitpackingPrimitives::GetRequiredSize(count, width);
			ReserveSpace(state, PatchedForBytes<T>(count, width, exception_count));

			WriteMetaData(state, BitpackingMode::PATCHED_FOR);
			WriteData(state->data_ptr, base);
			WriteData(state->data_ptr, (T)width);
			WriteData(state->data_ptr, divisor);
			WriteData(state->data_ptr, UnsafeNumericCast<uint16_t>(exception_count));
			WriteData(state->data_ptr, uint16_t(0));
			for (idx_t i = 0; i < exception_count; i++) {
				Store<uint16_t>(positions[i], state->data_ptr);
				state->data_ptr += sizeof(uint16_t);
			}
			for (idx_t i = 0; i < exception_count; i++) {
				Store<T>(exceptions[i], state->data_ptr);
				state->data_ptr += sizeof(T);
			}

			BitpackingPrimitives::PackBuffer<T, false>(state->data_ptr, values, count, width);
			state->data_ptr += bp_size;

			UpdateStats(state, count);
		}

		template <class T_OUT>
		static void WriteData(data_ptr_t &ptr, T_OUT val) {
			*reinterpret_cast<T_OUT *>(ptr) = val;
			ptr += sizeof(T_OUT);
		}

		static void WriteMetaData(BitpackingCompressionState<T, WRITE_STATISTICS> *state, BitpackingMode mode) {
			bitpacking_metadata_t metadata {mode, (uint32_t)(state->data_ptr - state->handle.Ptr())};
			state->metadata_ptr -= sizeof(bitpacking_metadata_encoded_t);
			Store<bitpacking_metadata_encoded_t>(EncodeMeta(metadata), state->metadata_ptr);
		}

		static void ReserveSpace(BitpackingCompressionState<T, WRITE_STATISTICS> *state, idx_t data_bytes) {
			idx_t meta_bytes = sizeof(bitpacking_metadata_encoded_t);
			state->FlushAndCreateSegmentIfFull(data_bytes, meta_bytes);
			D_ASSERT(state->CanStore(data_bytes, meta_bytes));
		}

		static void UpdateStats(BitpackingCompressionState<T, WRITE_STATISTICS> *state, idx_t count) {
			state->current_segment->count += count;

			if (WRITE_STATISTICS) {
				if (state->state.has_valid) {
					state->current_segment->stats.statistics.SetHasNoNullFast();
				}
				if (state->state.has_invalid) {
					state->current_segment->stats.statistics.SetHasNullFast();
				}

				if (!state->state.all_invalid) {
					state->current_segment->stats.statistics.template UpdateNumericStats<T>(state->state.maximum);
					state->current_segment->stats.statistics.template UpdateNumericStats<T>(state->state.minimum);
				}
			}
		}
	};

	bool CanStore(idx_t data_bytes, idx_t meta_bytes) {
		auto required_data_bytes = AlignValue<idx_t>(UnsafeNumericCast<idx_t>((data_ptr + data_bytes) - data_ptr));
		auto required_meta_bytes = info.GetBlockSize() - UnsafeNumericCast<idx_t>(metadata_ptr - data_ptr) + meta_bytes;

		return required_data_bytes + required_meta_bytes <=
		       info.GetBlockSize() - BitpackingPrimitives::BITPACKING_HEADER_SIZE;
	}

	void CreateEmptySegment() {
		auto &db = checkpoint_data.GetDatabase();
		auto &type = checkpoint_data.GetType();

		auto compressed_segment =
		    ColumnSegment::CreateTransientSegment(db, function, type, info.GetBlockSize(), info.GetBlockManager());
		current_segment = std::move(compressed_segment);

		auto &buffer_manager = BufferManager::GetBufferManager(db);
		handle = buffer_manager.Pin(current_segment->block);

		data_ptr = handle.Ptr() + BitpackingPrimitives::BITPACKING_HEADER_SIZE;
		metadata_ptr = handle.Ptr() + info.GetBlockSize();
	}

	void Append(UnifiedVectorFormat &vdata, idx_t count) {
		auto data = UnifiedVectorFormat::GetData<T>(vdata);

		for (idx_t i = 0; i < count; i++) {
			idx_t idx = vdata.sel->get_index(i);
			state.template Update<BitpackingCompressionState<T, WRITE_STATISTICS, T_S>::BitpackingWriter>(
			    data[idx], vdata.validity.RowIsValid(idx));
		}
	}

	void FlushAndCreateSegmentIfFull(idx_t required_data_bytes, idx_t required_meta_bytes) {
		if (!CanStore(required_data_bytes, required_meta_bytes)) {
			FlushSegment();
			CreateEmptySegment();
		}
	}

	void FlushSegment() {
		auto &state = checkpoint_data.GetCheckpointState();
		auto base_ptr = handle.Ptr();

		// Compact the segment by moving the metadata next to the data.

		idx_t unaligned_offset = NumericCast<idx_t>(data_ptr - base_ptr);
		idx_t metadata_offset = AlignValue(unaligned_offset);
		idx_t metadata_size = NumericCast<idx_t>(base_ptr + info.GetBlockSize() - metadata_ptr);
		idx_t total_segment_size = metadata_offset + metadata_size;

		// Asserting things are still sane here
		if (!CanStore(0, 0)) {
			throw InternalException("Error in bitpacking size calculation");
		}

		if (unaligned_offset != metadata_offset) {
			// zero initialize any padding bits
			memset(base_ptr + unaligned_offset, 0, metadata_offset - unaligned_offset);
		}
		memmove(base_ptr + metadata_offset, metadata_ptr, metadata_size);

		// Store the offset of the metadata of the first group (which is at the highest address).
		Store<idx_t>(metadata_offset + metadata_size, base_ptr);

		state.FlushSegment(std::move(current_segment), std::move(handle), total_segment_size);
	}

	void Finalize() {
		state.template Flush<BitpackingCompressionState<T, WRITE_STATISTICS, T_S>::BitpackingWriter>();
		FlushSegment();
		current_segment.reset();
	}
};

template <class T, bool WRITE_STATISTICS>
unique_ptr<CompressionState> BitpackingInitCompression(ColumnDataCheckpointData &checkpoint_data,
                                                       unique_ptr<AnalyzeState> state) {
	return make_uniq<BitpackingCompressionState<T, WRITE_STATISTICS>>(checkpoint_data, state->info);
}

template <class T, bool WRITE_STATISTICS>
void BitpackingCompress(CompressionState &state_p, Vector &scan_vector, idx_t count) {
	auto &state = state_p.Cast<BitpackingCompressionState<T, WRITE_STATISTICS>>();
	UnifiedVectorFormat vdata;
	scan_vector.ToUnifiedFormat(count, vdata);
	state.Append(vdata, count);
}

template <class T, bool WRITE_STATISTICS>
void BitpackingFinalizeCompress(CompressionState &state_p) {
	auto &state = state_p.Cast<BitpackingCompressionState<T, WRITE_STATISTICS>>();
	state.Finalize();
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
template <class T>
static void ApplyFrameOfReference(T *dst, T frame_of_reference, idx_t size) {
	using T_U = typename MakeUnsigned<T>::type;
	if (!frame_of_reference) {
		return;
	}

	for (idx_t i = 0; i < size; i++) {
		reinterpret_cast<T_U *>(dst)[i] += static_cast<T_U>(frame_of_reference);
	}
}

// Based on https://github.com/lemire/FastPFor (Apache License 2.0)
template <class T>
static T DeltaDecode(T *data, T previous_value, const size_t size) {
	D_ASSERT(size >= 1);

	data[0] += previous_value;

	const size_t UnrollQty = 4;
	const size_t sz0 = (size / UnrollQty) * UnrollQty; // equal to 0, if size < UnrollQty
	size_t i = 1;
	if (sz0 >= UnrollQty) {
		T a = data[0];
		for (; i < sz0 - UnrollQty; i += UnrollQty) {
			a = data[i] += a;
			a = data[i + 1] += a;
			a = data[i + 2] += a;
			a = data[i + 3] += a;
		}
	}
	for (; i != size; ++i) {
		data[i] += data[i - 1];
	}

	return data[size - 1];
}

template <class T, class T_S = typename MakeSigned<T>::type>
struct BitpackingScanState : public SegmentScanState {
public:
	explicit BitpackingScanState(const QueryContext &context, ColumnSegment &segment) : current_segment(segment) {
		auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
		handle = buffer_manager.Pin(context, segment.block);
		auto data_ptr = handle.Ptr();

		// load offset to bitpacking widths pointer
		auto bitpacking_metadata_offset = Load<idx_t>(data_ptr + segment.GetBlockOffset());
		bitpacking_metadata_ptr =
		    data_ptr + segment.GetBlockOffset() + bitpacking_metadata_offset - sizeof(bitpacking_metadata_encoded_t);
		if (bitpacking_metadata_ptr >= handle.Ptr() + current_segment.GetBlockSize()) {
			throw InternalException("Bitpacking offset is out of range at block \"%llu\" - corrupt database file",
			                        segment.block->BlockId());
		}

		// load the first group
		LoadNextGroup();
	}

	BufferHandle handle;
	ColumnSegment &current_segment;

	T decompression_buffer[BITPACKING_METADATA_GROUP_SIZE];

	bitpacking_metadata_t current_group;
	bitpacking_width_t current_width;
	T current_frame_of_reference;
	T current_constant;
	T current_delta_offset;
	T current_divisor;
	// PATCHED_FOR: the group's exceptions (positions ascending) and the first one a scan has not passed
	idx_t current_exception_count = 0;
	idx_t current_exception_index = 0;
	uint16_t current_exception_positions[BITPACKING_PATCHED_FOR_MAX_EXCEPTIONS];
	T current_exception_values[BITPACKING_PATCHED_FOR_MAX_EXCEPTIONS];

	idx_t current_group_offset = 0;
	data_ptr_t current_group_ptr;
	data_ptr_t bitpacking_metadata_ptr;

public:
	//! Loads the metadata for the current metadata group. This will set bitpacking_metadata_ptr to the next group.
	//! It also loads any metadata at the start of a compressed buffer (e.g. the width, for, or constant value)
	//! depending on the bitpacking mode of that group.
	void LoadNextGroup() {
		D_ASSERT(bitpacking_metadata_ptr > handle.Ptr() &&
		         (bitpacking_metadata_ptr < handle.Ptr() + current_segment.GetBlockSize()));
		current_group_offset = 0;
		current_group = DecodeMeta(reinterpret_cast<bitpacking_metadata_encoded_t *>(bitpacking_metadata_ptr));

		bitpacking_metadata_ptr -= sizeof(bitpacking_metadata_encoded_t);
		current_group_ptr = GetPtr(current_group);

		// Read first value
		switch (current_group.mode) {
		case BitpackingMode::CONSTANT:
			current_constant = *reinterpret_cast<T *>(current_group_ptr);
			current_group_ptr += sizeof(T);
			break;
		case BitpackingMode::FOR:
		case BitpackingMode::FOR_SCALED:
		case BitpackingMode::PATCHED_FOR:
		case BitpackingMode::CONSTANT_DELTA:
		case BitpackingMode::DELTA_FOR:
			current_frame_of_reference = *reinterpret_cast<T *>(current_group_ptr);
			current_group_ptr += sizeof(T);
			break;
		default:
			throw InternalException("Invalid bitpacking mode");
		}

		// Read second value
		switch (current_group.mode) {
		case BitpackingMode::CONSTANT_DELTA:
			current_constant = *reinterpret_cast<T *>(current_group_ptr);
			current_group_ptr += sizeof(T);
			break;
		case BitpackingMode::FOR:
		case BitpackingMode::FOR_SCALED:
		case BitpackingMode::PATCHED_FOR:
		case BitpackingMode::DELTA_FOR:
			current_width = (bitpacking_width_t)(*reinterpret_cast<T *>(current_group_ptr));
			current_group_ptr += MaxValue(sizeof(T), sizeof(bitpacking_width_t));
			break;
		case BitpackingMode::CONSTANT:
			break;
		default:
			throw InternalException("Invalid bitpacking mode");
		}

		// Read third value
		if (current_group.mode == BitpackingMode::DELTA_FOR) {
			current_delta_offset = *reinterpret_cast<T *>(current_group_ptr);
			current_group_ptr += sizeof(T);
		} else if (current_group.mode == BitpackingMode::FOR_SCALED ||
		           current_group.mode == BitpackingMode::PATCHED_FOR) {
			current_divisor = *reinterpret_cast<T *>(current_group_ptr);
			current_group_ptr += sizeof(T);
		}

		// PATCHED_FOR: the exceptions, before the packing
		current_exception_count = 0;
		current_exception_index = 0;
		if (current_group.mode == BitpackingMode::PATCHED_FOR) {
			current_exception_count = Load<uint16_t>(current_group_ptr);
			current_group_ptr += 2 * sizeof(uint16_t);
			if (current_exception_count > BITPACKING_PATCHED_FOR_MAX_EXCEPTIONS) {
				throw InternalException(
				    "Bitpacking group holds %llu exceptions at block \"%llu\" - corrupt database file",
				    current_exception_count, current_segment.block->BlockId());
			}
			for (idx_t i = 0; i < current_exception_count; i++) {
				current_exception_positions[i] = Load<uint16_t>(current_group_ptr);
				current_group_ptr += sizeof(uint16_t);
			}
			for (idx_t i = 0; i < current_exception_count; i++) {
				current_exception_values[i] = Load<T>(current_group_ptr);
				current_group_ptr += sizeof(T);
			}
		}
	}

	//! PATCHED_FOR: writes the exceptions at group offsets [start, start + count) into dst (dst[0] = offset start).
	//! Offsets only move forward within a group, so the first exception not yet passed is kept
	void ApplyExceptions(T *dst, idx_t start, idx_t count) {
		while (current_exception_index < current_exception_count &&
		       current_exception_positions[current_exception_index] < start) {
			current_exception_index++;
		}
		while (current_exception_index < current_exception_count &&
		       current_exception_positions[current_exception_index] < start + count) {
			dst[current_exception_positions[current_exception_index] - start] =
			    current_exception_values[current_exception_index];
			current_exception_index++;
		}
	}

	void Skip(ColumnSegment &segment, idx_t skip_count) {
		bool skip_sign_extend = true;

		idx_t skipped = 0;
		idx_t initial_group_offset = current_group_offset;

		// This skips straight to the correct metadata group
		idx_t meta_groups_to_skip = (skip_count + current_group_offset) / BITPACKING_METADATA_GROUP_SIZE;
		if (meta_groups_to_skip) {
			// bitpacking_metadata_ptr points to the next metadata: this means we need to advance the pointer by n-1
			bitpacking_metadata_ptr -= (meta_groups_to_skip - 1) * sizeof(bitpacking_metadata_encoded_t);
			LoadNextGroup();
			// The first (partial) group we skipped
			skipped += BITPACKING_METADATA_GROUP_SIZE - initial_group_offset;
			// The remaining groups that were skipped
			skipped += (meta_groups_to_skip - 1) * BITPACKING_METADATA_GROUP_SIZE;
		}

		// Assert we can are in the correct metadata group
		idx_t remaining_to_skip = skip_count - skipped;
		D_ASSERT(current_group_offset + remaining_to_skip < BITPACKING_METADATA_GROUP_SIZE);

		if (current_group.mode == BitpackingMode::CONSTANT || current_group.mode == BitpackingMode::CONSTANT_DELTA ||
		    current_group.mode == BitpackingMode::FOR || current_group.mode == BitpackingMode::FOR_SCALED ||
		    current_group.mode == BitpackingMode::PATCHED_FOR) {
			// Skipping within a constant or constant delta is done by increasing the current_group_offset
			skipped += remaining_to_skip;
			current_group_offset += remaining_to_skip;
		} else {
			// For DELTA we actually need to decompress from the current_group_offset up until the row we want to skip
			// to this is because we need that delta to be able to continue scanning from here
			D_ASSERT(current_group.mode == BitpackingMode::DELTA_FOR);

			while (skipped < skip_count) {
				// Calculate compression group offset and pointer
				idx_t offset_in_compression_group =
				    current_group_offset % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;
				data_ptr_t current_position_ptr = current_group_ptr + current_group_offset * current_width / 8;
				data_ptr_t decompression_group_start_pointer =
				    current_position_ptr - offset_in_compression_group * current_width / 8;

				idx_t skipping_this_algorithm_group =
				    MinValue(remaining_to_skip,
				             BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE - offset_in_compression_group);

				BitpackingPrimitives::UnPackBlock<T>(data_ptr_cast(decompression_buffer),
				                                     decompression_group_start_pointer, current_width,
				                                     skip_sign_extend);

				T *decompression_ptr = decompression_buffer + offset_in_compression_group;
				ApplyFrameOfReference<T_S>(reinterpret_cast<T_S *>(decompression_ptr),
				                           static_cast<T_S>(current_frame_of_reference), skipping_this_algorithm_group);
				DeltaDecode<T_S>(reinterpret_cast<T_S *>(decompression_ptr), static_cast<T_S>(current_delta_offset),
				                 skipping_this_algorithm_group);
				current_delta_offset = decompression_ptr[skipping_this_algorithm_group - 1];

				skipped += skipping_this_algorithm_group;
				current_group_offset += skipping_this_algorithm_group;
				remaining_to_skip -= skipping_this_algorithm_group;
			}
		}

		D_ASSERT(skipped == skip_count);
	}

	data_ptr_t GetPtr(bitpacking_metadata_t group) {
		return handle.Ptr() + current_segment.GetBlockOffset() + group.offset;
	}
};

template <class T>
unique_ptr<SegmentScanState> BitpackingInitScan(const QueryContext &context, ColumnSegment &segment) {
	auto result = make_uniq<BitpackingScanState<T>>(context, segment);
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// Scan base data
//===--------------------------------------------------------------------===//
template <class T, class T_S = typename MakeSigned<T>::type, class T_U = typename MakeUnsigned<T>::type>
void BitpackingScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
                           idx_t result_offset) {
	auto &scan_state = state.scan_state->Cast<BitpackingScanState<T>>();

	T *result_data = FlatVector::GetData<T>(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);

	//! Because FOR offsets all our values to be 0 or above, we can always skip sign extension here
	bool skip_sign_extend = true;

	idx_t scanned = 0;
	while (scanned < scan_count) {
		D_ASSERT(scan_state.current_group_offset <= BITPACKING_METADATA_GROUP_SIZE);

		// Exhausted this metadata group, move pointers to next group and load metadata for next group.
		if (scan_state.current_group_offset == BITPACKING_METADATA_GROUP_SIZE) {
			scan_state.LoadNextGroup();
		}

		idx_t offset_in_compression_group =
		    scan_state.current_group_offset % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;

		if (scan_state.current_group.mode == BitpackingMode::CONSTANT) {
			idx_t remaining = scan_count - scanned;
			idx_t to_scan = MinValue(remaining, BITPACKING_METADATA_GROUP_SIZE - scan_state.current_group_offset);
			T *begin = result_data + result_offset + scanned;
			T *end = begin + remaining;
			std::fill(begin, end, scan_state.current_constant);
			scanned += to_scan;
			scan_state.current_group_offset += to_scan;
			continue;
		}
		if (scan_state.current_group.mode == BitpackingMode::CONSTANT_DELTA) {
			idx_t remaining = scan_count - scanned;
			idx_t to_scan = MinValue(remaining, BITPACKING_METADATA_GROUP_SIZE - scan_state.current_group_offset);
			T *target_ptr = result_data + result_offset + scanned;

			for (idx_t i = 0; i < to_scan; i++) {
				idx_t multiplier = scan_state.current_group_offset + i;
				// intended static casts to unsigned and back for defined wrapping of integers
				target_ptr[i] = static_cast<T>((static_cast<T_U>(scan_state.current_constant) * multiplier) +
				                               static_cast<T_U>(scan_state.current_frame_of_reference));
			}

			scanned += to_scan;
			scan_state.current_group_offset += to_scan;
			continue;
		}
		D_ASSERT(scan_state.current_group.mode == BitpackingMode::FOR ||
		         scan_state.current_group.mode == BitpackingMode::FOR_SCALED ||
		         scan_state.current_group.mode == BitpackingMode::PATCHED_FOR ||
		         scan_state.current_group.mode == BitpackingMode::DELTA_FOR);

		idx_t to_scan = MinValue<idx_t>(scan_count - scanned, BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE -
		                                                          offset_in_compression_group);
		// Calculate start of compression algorithm group
		data_ptr_t current_position_ptr =
		    scan_state.current_group_ptr + scan_state.current_group_offset * scan_state.current_width / 8;
		data_ptr_t decompression_group_start_pointer =
		    current_position_ptr - offset_in_compression_group * scan_state.current_width / 8;

		T *current_result_ptr = result_data + result_offset + scanned;

		if (to_scan == BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE && offset_in_compression_group == 0) {
			// Decompress directly into result vector
			BitpackingPrimitives::UnPackBlock<T>(data_ptr_cast(current_result_ptr), decompression_group_start_pointer,
			                                     scan_state.current_width, skip_sign_extend);
		} else {
			// Decompress compression algorithm to buffer
			BitpackingPrimitives::UnPackBlock<T>(data_ptr_cast(scan_state.decompression_buffer),
			                                     decompression_group_start_pointer, scan_state.current_width,
			                                     skip_sign_extend);

			memcpy(current_result_ptr, scan_state.decompression_buffer + offset_in_compression_group,
			       to_scan * sizeof(T));
		}

		if (scan_state.current_group.mode == BitpackingMode::DELTA_FOR) {
			ApplyFrameOfReference<T_S>(reinterpret_cast<T_S *>(current_result_ptr),
			                           static_cast<T_S>(scan_state.current_frame_of_reference), to_scan);
			DeltaDecode<T_S>(reinterpret_cast<T_S *>(current_result_ptr),
			                 static_cast<T_S>(scan_state.current_delta_offset), to_scan);
			scan_state.current_delta_offset = current_result_ptr[to_scan - 1];
		} else if (scan_state.current_group.mode == BitpackingMode::FOR_SCALED) {
			ApplyScaledFrameOfReference<T>(current_result_ptr, scan_state.current_frame_of_reference,
			                               scan_state.current_divisor, to_scan);
		} else if (scan_state.current_group.mode == BitpackingMode::PATCHED_FOR) {
			ApplyScaledFrameOfReference<T>(current_result_ptr, scan_state.current_frame_of_reference,
			                               scan_state.current_divisor, to_scan);
			scan_state.ApplyExceptions(current_result_ptr, scan_state.current_group_offset, to_scan);
		} else {
			ApplyFrameOfReference<T>(current_result_ptr, scan_state.current_frame_of_reference, to_scan);
		}

		scanned += to_scan;
		scan_state.current_group_offset += to_scan;
	}
}

template <class T>
void BitpackingScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result) {
	BitpackingScanPartial<T>(segment, state, scan_count, result, 0);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
template <class T>
void BitpackingFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result,
                        idx_t result_idx) {
	BitpackingScanState<T> scan_state(state.context, segment);
	scan_state.Skip(segment, NumericCast<idx_t>(row_id));

	D_ASSERT(scan_state.current_group_offset < BITPACKING_METADATA_GROUP_SIZE);

	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
	T *result_data = FlatVector::GetData<T>(result);
	T *current_result_ptr = result_data + result_idx;

	idx_t offset_in_compression_group =
	    scan_state.current_group_offset % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;

	data_ptr_t decompression_group_start_pointer =
	    scan_state.current_group_ptr +
	    (scan_state.current_group_offset - offset_in_compression_group) * scan_state.current_width / 8;

	//! Because FOR offsets all our values to be 0 or above, we can always skip sign extension here
	bool skip_sign_extend = true;

	if (scan_state.current_group.mode == BitpackingMode::CONSTANT) {
		*current_result_ptr = scan_state.current_constant;
		return;
	}

	if (scan_state.current_group.mode == BitpackingMode::CONSTANT_DELTA) {
		T multiplier;
		auto cast = TryCast::Operation<idx_t, T>(scan_state.current_group_offset, multiplier);
		(void)cast;
		D_ASSERT(cast);
#ifdef DEBUG
		// overflow check
		T result;
		bool multiply = TryMultiplyOperator::Operation(multiplier, scan_state.current_constant, result);
		bool add = TryAddOperator::Operation(result, scan_state.current_frame_of_reference, result);
		D_ASSERT(multiply && add);
#endif
		*current_result_ptr = (multiplier * scan_state.current_constant) + scan_state.current_frame_of_reference;
		return;
	}

	D_ASSERT(scan_state.current_group.mode == BitpackingMode::FOR ||
	         scan_state.current_group.mode == BitpackingMode::FOR_SCALED ||
	         scan_state.current_group.mode == BitpackingMode::PATCHED_FOR ||
	         scan_state.current_group.mode == BitpackingMode::DELTA_FOR);

	BitpackingPrimitives::UnPackBlock<T>(data_ptr_cast(scan_state.decompression_buffer),
	                                     decompression_group_start_pointer, scan_state.current_width, skip_sign_extend);

	*current_result_ptr = scan_state.decompression_buffer[offset_in_compression_group];
	if (scan_state.current_group.mode == BitpackingMode::FOR_SCALED) {
		ApplyScaledFrameOfReference<T>(current_result_ptr, scan_state.current_frame_of_reference,
		                               scan_state.current_divisor, 1);
		return;
	}
	if (scan_state.current_group.mode == BitpackingMode::PATCHED_FOR) {
		ApplyScaledFrameOfReference<T>(current_result_ptr, scan_state.current_frame_of_reference,
		                               scan_state.current_divisor, 1);
		scan_state.ApplyExceptions(current_result_ptr, scan_state.current_group_offset, 1);
		return;
	}
	*current_result_ptr += scan_state.current_frame_of_reference;

	if (scan_state.current_group.mode == BitpackingMode::DELTA_FOR) {
		*current_result_ptr += scan_state.current_delta_offset;
	}
}

template <class T>
void BitpackingSkip(ColumnSegment &segment, ColumnScanState &state, idx_t skip_count) {
	auto &scan_state = static_cast<BitpackingScanState<T> &>(*state.scan_state);
	scan_state.Skip(segment, skip_count);
}

//===--------------------------------------------------------------------===//
// GetSegmentInfo
//===--------------------------------------------------------------------===//
template <class T>
InsertionOrderPreservingMap<string> BitpackingGetSegmentInfo(QueryContext context, ColumnSegment &segment) {
	map<BitpackingMode, idx_t> counts;
	auto tuple_count = segment.count.load();
	BitpackingScanState<T> scan_state(context, segment);
	for (idx_t i = 0; i < tuple_count; i += BITPACKING_METADATA_GROUP_SIZE) {
		if (i) {
			scan_state.LoadNextGroup();
		}
		counts[scan_state.current_group.mode]++;
	}

	InsertionOrderPreservingMap<string> result;
	for (auto &it : counts) {
		auto &mode = it.first;
		auto &count = it.second;
		result[EnumUtil::ToString(mode)] = StringUtil::Format("%d", count);
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Get Function
//===--------------------------------------------------------------------===//
template <class T, bool WRITE_STATISTICS = true>
CompressionFunction GetBitpackingFunction(PhysicalType data_type) {
	auto bitpacking = CompressionFunction(
	    CompressionType::COMPRESSION_BITPACKING, data_type, BitpackingInitAnalyze<T>, BitpackingAnalyze<T>,
	    BitpackingFinalAnalyze<T>, BitpackingInitCompression<T, WRITE_STATISTICS>,
	    BitpackingCompress<T, WRITE_STATISTICS>, BitpackingFinalizeCompress<T, WRITE_STATISTICS>, BitpackingInitScan<T>,
	    BitpackingScan<T>, BitpackingScanPartial<T>, BitpackingFetchRow<T>, BitpackingSkip<T>);
	bitpacking.get_segment_info = BitpackingGetSegmentInfo<T>;
	return bitpacking;
}

CompressionFunction BitpackingFun::GetFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return GetBitpackingFunction<int8_t>(type);
	case PhysicalType::INT16:
		return GetBitpackingFunction<int16_t>(type);
	case PhysicalType::INT32:
		return GetBitpackingFunction<int32_t>(type);
	case PhysicalType::INT64:
		return GetBitpackingFunction<int64_t>(type);
	case PhysicalType::UINT8:
		return GetBitpackingFunction<uint8_t>(type);
	case PhysicalType::UINT16:
		return GetBitpackingFunction<uint16_t>(type);
	case PhysicalType::UINT32:
		return GetBitpackingFunction<uint32_t>(type);
	case PhysicalType::UINT64:
		return GetBitpackingFunction<uint64_t>(type);
	case PhysicalType::INT128:
		return GetBitpackingFunction<hugeint_t>(type);
	case PhysicalType::UINT128:
		return GetBitpackingFunction<uhugeint_t>(type);
	case PhysicalType::LIST:
		return GetBitpackingFunction<uint64_t, false>(type);
	default:
		throw InternalException("Unsupported type for Bitpacking");
	}
}

bool BitpackingFun::TypeIsSupported(const PhysicalType physical_type) {
	switch (physical_type) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
	case PhysicalType::LIST:
	case PhysicalType::INT128:
	case PhysicalType::UINT128:
		return true;
	default:
		return false;
	}
}

} // namespace duckdb
