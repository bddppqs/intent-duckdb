//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/single_file_block_manager.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/storage/block_manager.hpp"
#include "duckdb/storage/block.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/common/map.hpp"
#include "duckdb/storage/storage_info.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/common/encryption_functions.hpp"

#include <condition_variable>

namespace duckdb {

class DatabaseInstance;
struct MetadataHandle;
struct BlockWriteback;
enum class FreeBlockType { NEWLY_USED_BLOCK, CHECKPOINTED_BLOCK };

struct EncryptionOptions {
	//! indicates whether the db is encrypted
	bool encryption_enabled = false;
	//! Whether Additional Authenticated Data is used
	bool additional_authenticated_data = false;
	//! derived encryption key id
	string derived_key_id;
	// //! Cipher used for encryption
	// EncryptionTypes::CipherType cipher = EncryptionTypes::CipherType::INVALID;
	//! key derivation function (kdf) used
	EncryptionTypes::KeyDerivationFunction kdf = EncryptionTypes::KeyDerivationFunction::SHA256;
	//! Key Length
	uint32_t key_length = MainHeader::DEFAULT_ENCRYPTION_KEY_LENGTH;
	//! User key pointer (to StorageOptions)
	shared_ptr<string> user_key;
	//! Version of duckdb-encryption
	EncryptionTypes::EncryptionVersion encryption_version = EncryptionTypes::NONE;
};

struct StorageManagerOptions {
	bool read_only = false;
	bool use_direct_io = false;
	DebugInitialize debug_initialize = DebugInitialize::NO_INITIALIZE;
	optional_idx block_alloc_size;
	optional_idx storage_version;
	optional_idx version_number;
	optional_idx block_header_size;
	//! Unique database identifier and optional encryption salt.
	data_t db_identifier[MainHeader::DB_IDENTIFIER_LEN];
	EncryptionOptions encryption_options;
};

//! SingleFileBlockManager is an implementation for a BlockManager which manages blocks in a single file
class SingleFileBlockManager : public BlockManager {
	//! The location in the file where the block writing starts
	static constexpr uint64_t BLOCK_START = Storage::FILE_HEADER_SIZE * 3;

public:
	SingleFileBlockManager(AttachedDatabase &db_p, const string &path_p, const StorageManagerOptions &options_p);
	~SingleFileBlockManager() override;

	FileOpenFlags GetFileFlags(bool create_new) const;
	//! Creates a new database.
	void CreateNewDatabase(QueryContext context);
	//! Loads an existing database. We pass the provided block allocation size as a parameter
	//! to detect inconsistencies with the file header.
	void LoadExistingDatabase(QueryContext context);

	//! Creates a new Block using the specified block_id and returns a pointer
	unique_ptr<Block> ConvertBlock(block_id_t block_id, FileBuffer &source_buffer) override;
	unique_ptr<Block> CreateBlock(block_id_t block_id, FileBuffer *source_buffer) override;
	//! Return the next free block id
	block_id_t GetFreeBlockId() override;
	//! Return the next free block id
	block_id_t GetFreeBlockIdForCheckpoint() override;
	//! Check the next free block id - but do not assign or allocate it
	block_id_t PeekFreeBlockId() override;
	//! Returns whether or not a specified block is the root block
	bool IsRootBlock(MetaBlockPointer root) override;
	//! Mark a block as included in a checkpoint
	void MarkBlockAsCheckpointed(block_id_t block_id) override;
	//! Mark a block as used (no longer re-writeable)
	void MarkBlockAsUsed(block_id_t block_id) override;
	//! Mark a block as modified (re-writeable after a checkpoint)
	void MarkBlockAsModified(block_id_t block_id) override;
	//! Increase the reference count of a block. The block should hold at least one reference
	void IncreaseBlockReferenceCount(block_id_t block_id) override;
	//! UnregisterBlock, only accepts non-temporary block ids
	void UnregisterBlock(block_id_t id) override;
	//! Return the meta block id
	idx_t GetMetaBlock() override;
	//! Read the content of the block from disk
	void Read(QueryContext context, Block &block) override;

	//! Read individual blocks
	void ReadBlock(Block &block, bool skip_block_header = false) const;
	void ReadBlock(data_ptr_t internal_buffer, uint64_t block_size, bool skip_block_header = false) const;
	//! Read the content of a range of blocks into a buffer
	void ReadBlocks(FileBuffer &buffer, block_id_t start_block, idx_t block_count) override;
	//! Write the block to disk. Use Write with client context instead.
	void Write(FileBuffer &buffer, block_id_t block_id) override;
	//! Write the block to disk.
	void Write(QueryContext context, FileBuffer &buffer, block_id_t block_id) override;
	//! Write the header to disk, this is the final step of the checkpointing process
	void WriteHeader(QueryContext context, DatabaseHeader header) override;
	//! Sync changes to the underlying file
	void FileSync() override;
	//! Truncate the underlying database file after a checkpoint
	void Truncate() override;

	bool InMemory() override {
		return false;
	}
	//! Returns the number of total blocks
	idx_t TotalBlocks() override;
	//! Returns the number of free blocks
	idx_t FreeBlocks() override;
	//! Whether or not the attached database is a remote file
	bool IsRemote() override;
	//! Whether or not to prefetch
	bool Prefetch() override;
	//! Read-ahead: ask the operating system to start reading the given blocks that are not loaded, without waiting for
	//! them: each block's slot, or in a compressed file its extent, in pieces the kernel reads in full. A no-op for
	//! direct IO, a remote file or a handle that cannot take the hint
	void ReadAhead(const vector<shared_ptr<BlockHandle>> &handles);

	//! Return the checkpoint iteration of the file.
	uint64_t GetCheckpointIteration() const {
		return iteration_count;
	}
	//! Return the version number of the file.
	uint64_t GetVersionNumber() const;
	//! Return the database identifier.
	data_ptr_t GetDBIdentifier() {
		return options.db_identifier;
	}
	//! Whether blocks are stored compressed, one variable-length extent per block (storage version BLOCK_COMPRESSION_VERSION_NUMBER)
	bool BlockCompression() const {
		return block_compression;
	}

private:
	//! Loads the free list of the file.
	void LoadFreeList(QueryContext context);

	//! Initializes the database header. We pass the provided block allocation size as a parameter
	//!	to detect inconsistencies with the file header.
	void Initialize(const DatabaseHeader &header, const optional_idx block_alloc_size);

	void CheckChecksum(FileBuffer &block, uint64_t location, uint64_t delta, bool skip_block_header = false) const;
	void CheckChecksum(data_ptr_t start_ptr, uint64_t delta, bool skip_block_header = false) const;

	void ReadAndChecksum(QueryContext context, FileBuffer &handle, uint64_t location,
	                     bool skip_block_header = false) const;
	void ChecksumAndWrite(QueryContext context, FileBuffer &handle, uint64_t location,
	                      bool skip_block_header = false) const;

	idx_t GetBlockLocation(block_id_t block_id) const;

	// Encrypt, Store, Decrypt the canary
	static void StoreEncryptedCanary(AttachedDatabase &db, MainHeader &main_header, const string &key_id);
	static void StoreDBIdentifier(MainHeader &main_header, const data_ptr_t db_identifier);
	void StoreEncryptionMetadata(MainHeader &main_header) const;
	template <typename T>
	static void WriteEncryptionData(MemoryStream &stream, const T &val);

	//! Check and adding Encryption Keys
	void CheckAndAddEncryptionKey(MainHeader &main_header, string &user_key);
	void CheckAndAddEncryptionKey(MainHeader &main_header);

	//! Return the blocks to which we will write the free list and modified blocks
	vector<MetadataHandle> GetFreeListBlocks();
	void TrimFreeBlocks(const set<block_id_t> &blocks);
	void TrimFreeBlockRange(block_id_t start, block_id_t end);

	void IncreaseBlockReferenceCountInternal(block_id_t block_id);

	//! Block compression: the file position of a block is its extent (offset, stored length), not its id
	struct BlockExtent {
		//! The file offset of the extent (page-aligned); 0 = the block has no extent
		uint64_t offset = 0;
		//! The bytes stored after the block header: a zstd frame, or the raw payload when it equals the block size
		uint32_t length = 0;
	};
	void ReadCompressedBlock(QueryContext context, data_ptr_t internal_buffer, block_id_t block_id);
	void ReadExtent(QueryContext context, data_ptr_t internal_buffer, block_id_t block_id, const BlockExtent &extent);
	void WriteCompressedBlock(QueryContext context, FileBuffer &buffer, block_id_t block_id);
	//! Extent allocation: the best-fitting free range, else at the end of the file
	uint64_t AllocateExtent(idx_t bytes);
	uint64_t AllocateExtentLocked(idx_t size);
	//! Free space bookkeeping (extent_lock held): replace it, or remove [offset, offset + size) from it
	void SetFreeSpaceLocked(const vector<pair<uint64_t, idx_t>> &ranges);
	void TakeFreeSpaceLocked(uint64_t offset, idx_t size);
	//! After a commit: the free space is every byte no live extent and no committed extent map holds; a free tail is cut
	void RebuildFreeExtents();
	//! Write the extent map of blocks [0, block_count) as one raw extent; its position goes into the database header
	void WriteExtentMap(QueryContext context, idx_t block_count, const set<block_id_t> &free_blocks,
	                    optional_idx at_offset = optional_idx());
	//! After a commit: an extent map that ends the file moves into the lowest free range that holds it (map, sync,
	//! in-place header), so the rebuild can cut the free tail before it
	void RelocateExtentMapLow(QueryContext context, const set<block_id_t> &free_blocks);
	void LoadExtentMap(QueryContext context, idx_t map_offset, idx_t map_entries);
	//! After a commit: move the live extents at the end of the file into the free space before them, then make the new
	//! placement durable (extent map, sync, the active header rewritten in place, sync); the next rebuild cuts the tail
	void CompactAfterCommit(QueryContext context, const set<block_id_t> &free_blocks);
	//! Rewrite the active database header in place with the current extent map position (same iteration), then sync
	void WriteActiveHeaderInPlace(QueryContext context);
	//! Leave the extent IO section (see extent_io)
	void EndExtentIO();

	//! Early writeback: start the device writes of written blocks in the background (no-op where unsupported)
	void NotifyBlockWritten(idx_t bytes);

	//! Verify the block usage count
	void VerifyBlocks(const unordered_map<block_id_t, idx_t> &block_usage_count) override;

	void AddStorageVersionTag();

	block_id_t GetFreeBlockIdInternal(FreeBlockType type);
	//! Adds a free block to the free_list, returns true if it was added to the regular free_list
	bool AddFreeBlock(unique_lock<mutex> &lock, block_id_t block_id);

private:
	AttachedDatabase &db;
	//! The active DatabaseHeader, either 0 (h1) or 1 (h2)
	uint8_t active_header;
	//! The path where the file is stored
	string path;
	//! The file handle
	unique_ptr<FileHandle> handle;
	//! The buffer used to read/write to the headers
	FileBuffer header_buffer;
	//! The list of free blocks that can be written to currently
	set<block_id_t> free_list;
	//! The list of blocks that have been freed, but cannot yet be re-used because they are still in-use
	set<block_id_t> free_blocks_in_use;
	//! The list of blocks that are in-use, but haven't been written as part of a checkpoint yet
	set<block_id_t> newly_used_blocks;
	//! The list of multi-use blocks (i.e. blocks that have >1 reference in the file)
	//! When a multi-use block is marked as modified, the reference count is decreased by 1 instead of directly
	//! Appending the block to the modified_blocks list
	unordered_map<block_id_t, uint32_t> multi_use_blocks;
	//! The list of blocks that are no longer in-use, but cannot be re-used until the next checkpoint
	unordered_set<block_id_t> modified_blocks;
	//! The current meta block id
	idx_t meta_block;
	//! The current maximum block id, this id will be given away first after the free_list runs out
	block_id_t max_block;
	//! The block id where the free list can be found
	idx_t free_list_id;
	//! The current header iteration count.
	uint64_t iteration_count;
	//! The storage manager options
	StorageManagerOptions options;
	//! Lock for performing various operations in the single file block manager
	mutex single_file_block_lock;
	//! Whether blocks are stored compressed (a file created at storage version BLOCK_COMPRESSION_VERSION_NUMBER)
	bool block_compression = false;
	//! Lock for the extent map and the extent allocator
	mutex extent_lock;
	//! The extent of each block id
	vector<BlockExtent> extents;
	//! The next free (page-aligned) file offset of the append-only extent allocator
	uint64_t next_extent_offset = BLOCK_START;
	//! The position of the extent map written with the last database header
	idx_t extent_map_offset = 0;
	idx_t extent_map_entries = 0;
	//! The active database header as last written or loaded (the base of an in-place rewrite)
	DatabaseHeader durable_header;
	//! Serializes the header writers: WriteHeader and FileSync's extent map commit
	mutex header_lock;
	//! Block reads and writes in flight (between taking an extent and finishing its IO), and whether a compaction runs;
	//! a compaction waits for the IO in flight and holds back new IO until it is done
	idx_t extent_io = 0;
	bool extent_compacting = false;
	std::condition_variable extent_cv;
	//! The free space: ranges no committed state and no live block references, reusable now (by offset and by size)
	map<uint64_t, idx_t> free_by_offset;
	multimap<idx_t, uint64_t> free_by_size;
	//! The extents the last compaction moved
	idx_t compact_moved = 0;
	//! Early writeback state, created at the first block write of a writable on-disk file
	unique_ptr<BlockWriteback> writeback;
	bool writeback_checked = false;
	mutex writeback_lock;
};
} // namespace duckdb
