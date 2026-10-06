//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/compression/dict_fsst/split_segment.hpp
//
// A split DICT_FSST segment keeps its header and dictionary in its own block and its local codes (the bit-packed index
// array) in a code block shared by one row group's column: a codes-only scan never reads the dictionary bytes.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/storage/data_pointer.hpp"
#include "duckdb/storage/storage_info.hpp"

namespace duckdb {
class BlockHandle;
class BlockManager;
struct BlockIdVisitor;
class ColumnSegment;
class Deserializer;
class Serializer;
struct PrefetchState;
namespace dict_global {
class PersistedTranslations;
}

namespace dict_fsst {

//! Where a split segment's local codes are, and the header fields a codes-only scan reads without the segment's block
struct SplitCodes {
	block_id_t block = INVALID_BLOCK;
	uint32_t offset = 0;
	uint32_t size = 0;
	uint32_t dict_count = 0;
	uint8_t mode = 0;
	uint8_t indices_width = 0;

	bool IsSplit() const {
		return block != INVALID_BLOCK;
	}
};

//! The serialized segment state of a split segment (DataPointer::segment_state); `blocks` holds the code block
struct SplitSegmentState : public ColumnSegmentState {
	SplitSegmentState() = default;
	explicit SplitSegmentState(const SplitCodes &codes_p);

	SplitCodes codes;

	void Serialize(Serializer &serializer) const override;
	static unique_ptr<ColumnSegmentState> Deserialize(Deserializer &deserializer);
};

//! Whether a DICT_FSST writer splits its segments: the file is a block-compressed file at the split-dictionary version
bool SplitSegmentsEnabled(BlockManager &block_manager);
//! The segment's split codes, or null (not split)
optional_ptr<const SplitCodes> SegmentSplit(ColumnSegment &segment);
//! The writer records a segment's split codes (before the segment is flushed)
void SetSegmentSplit(ColumnSegment &segment, const SplitCodes &codes);
//! The handle of the segment's code block (registered once per segment)
shared_ptr<BlockHandle> SplitCodeHandle(ColumnSegment &segment);

//! The segment's column translations and entry index, linked when first read inside a statement; null when unlinked
shared_ptr<dict_global::PersistedTranslations> SegmentTranslationLink(ColumnSegment &segment, idx_t &entry);
void LinkSegmentTranslation(ColumnSegment &segment, shared_ptr<dict_global::PersistedTranslations> translations,
                            idx_t entry);

//! DICT_FSST's segment-state callbacks: the serialized state of a split segment (null otherwise), its deserialization,
//! the code block among the segment's blocks, and the prefetch of the blocks a scan on this thread reads
unique_ptr<ColumnSegmentState> DictFSSTSerializeState(ColumnSegment &segment);
unique_ptr<ColumnSegmentState> DictFSSTDeserializeState(Deserializer &deserializer);
void DictFSSTVisitBlockIds(const ColumnSegment &segment, BlockIdVisitor &visitor);
void DictFSSTInitPrefetch(ColumnSegment &segment, PrefetchState &prefetch_state);

} // namespace dict_fsst
} // namespace duckdb
