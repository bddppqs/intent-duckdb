#include "duckdb/storage/single_file_block_manager.hpp"

#include "duckdb/catalog/duck_catalog.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/checksum.hpp"
#include "duckdb/common/encryption_functions.hpp"
#include "duckdb/common/encryption_key_manager.hpp"
#include "duckdb/common/encryption_state.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/storage/buffer/block_handle.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/tuning_defaults.hpp"
#include "duckdb/common/enums/checkpoint_abort.hpp"
#include "duckdb/common/enums/storage_block_prefetch.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/block_allocator.hpp"
#include "duckdb/storage/metadata/metadata_reader.hpp"
#include "duckdb/storage/metadata/metadata_writer.hpp"
#include "duckdb/storage/storage_info.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "zstd.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <thread>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

namespace duckdb {

const char MainHeader::MAGIC_BYTES[] = "DUCK";
const char MainHeader::CANARY[] = "DUCKKEY";
static constexpr idx_t ENCRYPTION_METADATA_LEN = 8;

//===--------------------------------------------------------------------===//
// Block compression (storage version BLOCK_COMPRESSION_VERSION_NUMBER)
//===--------------------------------------------------------------------===//
// A file created at this storage version stores every block as one variable-length extent: the block header (the
// checksum of the uncompressed payload), then the payload as one zstd frame, or raw when the frame does not save at
// least one page. Extents are page-aligned and appended at the end of the file; an extent map (block id -> offset,
// stored length) is written as one raw extent before each database header, which records its position. The buffer
// pool, the block ids and every segment format are unchanged: a block is decompressed when it is read from the file.
//! A value outside upstream DuckDB's sequential storage-version range, so other readers refuse the file.
static constexpr uint64_t BLOCK_COMPRESSION_VERSION_NUMBER = 0x40000001;
//! The release successor of BLOCK_COMPRESSION_VERSION_NUMBER, written whenever a new file stores its blocks compressed:
//! the block-compressed file whose DICT_FSST segments may keep their local codes in code blocks of their own and whose
//! tables may store column translations, whose bit-packing groups may be FOR_SCALED and whose VARCHAR statistics carry
//! the minimum non-empty value; a reader without them refuses it by its version, and this reader opens both
static constexpr uint64_t RELEASE_STORAGE_VERSION_NUMBER = 0x40000002;
//! The release successor of RELEASE_STORAGE_VERSION_NUMBER, written whenever a new file stores its blocks compressed:
//! the release storage version whose tables may also store, after their row-group pointers, an index of those pointers
//! and each row group's column statistics column by column (kPersistedRowGroupIndex); a reader without them refuses
//! it by its version, and this reader opens all three block-compressed versions
static constexpr uint64_t PERSISTED_ROW_GROUP_INDEX_VERSION_NUMBER = 0x40000003;
//! Every block-compressed file version shares the layout
static bool IsBlockCompressedVersion(uint64_t version_number) {
	return version_number == BLOCK_COMPRESSION_VERSION_NUMBER || version_number == RELEASE_STORAGE_VERSION_NUMBER ||
	       version_number == PERSISTED_ROW_GROUP_INDEX_VERSION_NUMBER;
}
//! The release storage version or its successor: the file properties of the release version hold at both
static bool IsReleaseStorageVersion(uint64_t version_number) {
	return version_number == RELEASE_STORAGE_VERSION_NUMBER ||
	       version_number == PERSISTED_ROW_GROUP_INDEX_VERSION_NUMBER;
}
static constexpr idx_t BLOCK_EXTENT_ALIGNMENT = 4096;
//! The automatic block level (zstd_block_compression_level = 0): the high level with at least this many threads
static constexpr int32_t BLOCK_COMPRESSION_HIGH_LEVEL_THREADS = 64;
static constexpr int BLOCK_COMPRESSION_HIGH_LEVEL = 9;
static constexpr int BLOCK_COMPRESSION_LOW_LEVEL = 3;
static constexpr idx_t EXTENT_MAP_HEADER_SIZE = 32;
static constexpr idx_t EXTENT_MAP_ENTRY_SIZE = sizeof(uint64_t) + sizeof(uint32_t);
static constexpr char EXTENT_MAP_MAGIC[] = "DUCKXMAP";

namespace {
//! Per-thread zstd contexts and a staging buffer for one compressed block
struct BlockCompressionState {
	duckdb_zstd::ZSTD_CCtx *cctx = nullptr;
	duckdb_zstd::ZSTD_DCtx *dctx = nullptr;
	unsafe_unique_array<data_t> staging;
	idx_t staging_size = 0;

	~BlockCompressionState() {
		duckdb_zstd::ZSTD_freeCCtx(cctx);
		duckdb_zstd::ZSTD_freeDCtx(dctx);
	}
	data_ptr_t Staging(idx_t size) {
		if (size > staging_size) {
			staging = make_unsafe_uniq_array_uninitialized<data_t>(size);
			staging_size = size;
		}
		return staging.get();
	}
};

BlockCompressionState &GetBlockCompressionState() {
	thread_local BlockCompressionState state;
	return state;
}
} // namespace

// Early writeback
//===--------------------------------------------------------------------===//
// Block writes land in the page cache. Left alone, the device writes them only when the checkpoint fsyncs (or when the
// kernel's dirty-page ageing or thresholds kick in), so a bulk load computes first and writes afterwards. A background
// thread asks the kernel to start writing back what has been written (sync_file_range(SYNC_FILE_RANGE_WRITE): it
// initiates writeback and does not wait for it) once a batch has accumulated, so the device writes overlap the
// producers' work; the writers never block on it. Durability is unchanged: the checkpoint's fsyncs before and after
// the database header remain the durability points.
struct BlockWriteback {
	static constexpr idx_t BATCH_BYTES = 32ULL << 20;

	explicit BlockWriteback(int fd_p) : fd(fd_p), thread([this]() { Run(); }) {
	}
	//! Stop and join the thread; the descriptor is closed by the destructor
	void Stop() {
		{
			lock_guard<mutex> guard(lock);
			stop = true;
		}
		cv.notify_all();
		if (thread.joinable()) {
			thread.join();
		}
	}
	~BlockWriteback() {
		Stop();
#ifdef __linux__
		close(fd);
#endif
	}
	void Add(idx_t bytes) {
		lock_guard<mutex> guard(lock);
		auto was_idle = pending == 0;
		pending += bytes;
		if (was_idle || pending >= BATCH_BYTES) {
			cv.notify_one();
		}
	}

private:
	void Run() {
		unique_lock<mutex> guard(lock);
		while (!stop) {
			if (pending == 0) {
				cv.wait(guard);
				continue;
			}
			if (pending < BATCH_BYTES) {
				// let a batch accumulate, without holding back a small tail for long
				cv.wait_for(guard, std::chrono::milliseconds(20), [&]() { return stop || pending >= BATCH_BYTES; });
				if (stop) {
					break;
				}
			}
			pending = 0;
			guard.unlock();
#ifdef __linux__
			// the whole file: the kernel submits the pages that are dirty and not yet under writeback
			sync_file_range(fd, 0, 0, SYNC_FILE_RANGE_WRITE);
#endif
			guard.lock();
		}
	}

	int fd;
	mutex lock;
	std::condition_variable cv;
	idx_t pending = 0;
	bool stop = false;
	//! declared last: started once the members above exist
	std::thread thread;
};

void SerializeVersionNumber(WriteStream &ser, const string &version_str) {
	data_t version[MainHeader::MAX_VERSION_SIZE];
	memset(version, 0, MainHeader::MAX_VERSION_SIZE);
	memcpy(version, version_str.c_str(), MinValue<idx_t>(version_str.size(), MainHeader::MAX_VERSION_SIZE));
	ser.WriteData(version, MainHeader::MAX_VERSION_SIZE);
}

void SerializeDBIdentifier(WriteStream &ser, data_ptr_t db_identifier_p) {
	data_t db_identifier[MainHeader::DB_IDENTIFIER_LEN];
	memset(db_identifier, 0, MainHeader::DB_IDENTIFIER_LEN);
	memcpy(db_identifier, db_identifier_p, MainHeader::DB_IDENTIFIER_LEN);
	ser.WriteData(db_identifier, MainHeader::DB_IDENTIFIER_LEN);
}

void SerializeEncryptionMetadata(WriteStream &ser, data_ptr_t metadata_p, const bool encrypted) {
	// Zero-initialize.
	data_t metadata[MainHeader::ENCRYPTION_METADATA_LEN];
	memset(metadata, 0, MainHeader::ENCRYPTION_METADATA_LEN);

	// Write metadata, if encrypted.
	if (encrypted) {
		memcpy(metadata, metadata_p, MainHeader::ENCRYPTION_METADATA_LEN);
	}
	ser.WriteData(metadata, MainHeader::ENCRYPTION_METADATA_LEN);
}

void SerializeIV(WriteStream &ser, data_ptr_t metadata_p, const bool encrypted) {
	// Used for Canary encryption
	// Zero-initialize.
	data_t iv[MainHeader::AES_NONCE_LEN];
	memset(iv, 0, MainHeader::AES_NONCE_LEN);

	// Write metadata, if encrypted.
	if (encrypted) {
		memcpy(iv, metadata_p, MainHeader::AES_NONCE_LEN);
	}
	ser.WriteData(iv, MainHeader::AES_NONCE_LEN);
}

void SerializeTag(WriteStream &ser, data_ptr_t metadata_p, const bool encrypted) {
	// Used for Canary encryption
	// Zero-initialize.
	data_t tag[MainHeader::AES_TAG_LEN];
	memset(tag, 0, MainHeader::AES_TAG_LEN);

	// Write metadata, if encrypted.
	if (encrypted) {
		memcpy(tag, metadata_p, MainHeader::AES_TAG_LEN);
	}
	ser.WriteData(tag, MainHeader::AES_TAG_LEN);
}

void DeserializeVersionNumber(ReadStream &stream, data_t *dest) {
	memset(dest, 0, MainHeader::MAX_VERSION_SIZE);
	stream.ReadData(dest, MainHeader::MAX_VERSION_SIZE);
}

void DeserializeEncryptionData(ReadStream &stream, data_t *dest, idx_t size) {
	memset(dest, 0, size);
	stream.ReadData(dest, size);
}

void GenerateDBIdentifier(uint8_t *db_identifier) {
	memset(db_identifier, 0, MainHeader::DB_IDENTIFIER_LEN);
	RandomEngine engine;
	engine.RandomData(db_identifier, MainHeader::DB_IDENTIFIER_LEN);
}

void EncryptCanary(MainHeader &main_header, const shared_ptr<EncryptionState> &encryption_state,
                   const_data_ptr_t derived_key) {
	EncryptionCanary canary;
	EncryptionNonce nonce(EncryptionTypes::CipherType::GCM, encryption_state->metadata->GetVersion());
	memset(nonce.data(), 0, nonce.size());
	EncryptionTag tag;

	switch (encryption_state->metadata->GetVersion()) {
	case EncryptionTypes::V0_0:
		D_ASSERT(nonce.total_size() == MainHeader::AES_NONCE_LEN_DEPRECATED);
		encryption_state->InitializeEncryption(nonce, derived_key);
		encryption_state->Process(reinterpret_cast<const_data_ptr_t>(MainHeader::CANARY), canary.size(), canary.data(),
		                          canary.size());
		break;
	case EncryptionTypes::V0_1:
		// for GCM, total nonce size should be always equal to 12 bytes
		D_ASSERT(nonce.total_size() == MainHeader::AES_NONCE_LEN);
		encryption_state->GenerateRandomData(nonce.data(), nonce.size());
		main_header.SetCanaryIV(nonce.data());
		encryption_state->InitializeEncryption(nonce, derived_key);
		encryption_state->Process(reinterpret_cast<const_data_ptr_t>(MainHeader::CANARY), canary.size(), canary.data(),
		                          canary.size());
		encryption_state->Finalize(canary.data(), canary.size(), tag.data(), MainHeader::AES_TAG_LEN);
		main_header.SetCanaryTag(tag.data());
		break;
	default:
		throw InvalidInputException("No valid encryption version found!");
	}

	main_header.SetEncryptedCanary(canary.data());
}

bool DecryptCanary(MainHeader &main_header, const shared_ptr<EncryptionState> &encryption_state,
                   data_ptr_t derived_key) {
	auto encryption_version = encryption_state->metadata->GetVersion();
	EncryptionNonce nonce(EncryptionTypes::CipherType::GCM, encryption_version);
	EncryptionTag tag;
	EncryptionCanary decrypted_canary;

	switch (encryption_version) {
	case EncryptionTypes::V0_0:
		D_ASSERT(nonce.total_size() == MainHeader::AES_NONCE_LEN_DEPRECATED);
		//! Decrypt the canary, Nonce is zeroed out
		encryption_state->InitializeDecryption(nonce, derived_key);
		encryption_state->Process(main_header.GetEncryptedCanary(), decrypted_canary.size(), decrypted_canary.data(),
		                          decrypted_canary.size());
		break;
	case EncryptionTypes::V0_1:
		D_ASSERT(nonce.total_size() == MainHeader::AES_NONCE_LEN);
		// get the IV and the Tag
		memcpy(nonce.data(), main_header.GetIV(), nonce.total_size());
		memcpy(tag.data(), main_header.GetTag(), tag.size());

		//! Decrypt the canary
		encryption_state->InitializeDecryption(nonce, derived_key);
		encryption_state->Process(main_header.GetEncryptedCanary(), decrypted_canary.size(), decrypted_canary.data(),
		                          decrypted_canary.size());
		try {
			encryption_state->Finalize(decrypted_canary.data(), decrypted_canary.size(), tag.data(), tag.size());
		} catch (const std::exception &e) {
			throw InvalidInputException("Wrong encryption key used to open the database file");
		}
		break;
	default:
		throw InvalidInputException("No valid encryption version found!");
	}

	//! compare to check whether the decrypted canary is correct
	if (memcmp(decrypted_canary.data(), MainHeader::CANARY, MainHeader::CANARY_BYTE_SIZE) != 0) {
		return false;
	}

	return true;
}

void MainHeader::Write(WriteStream &ser) {
	ser.WriteData(const_data_ptr_cast(MAGIC_BYTES), MAGIC_BYTE_SIZE);
	ser.Write<uint64_t>(version_number);
	for (idx_t i = 0; i < FLAG_COUNT; i++) {
		ser.Write<uint64_t>(flags[i]);
	}

	SerializeVersionNumber(ser, DuckDB::LibraryVersion());
	SerializeVersionNumber(ser, DuckDB::SourceID());

	// We always serialize, and write zeros, if not set.
	auto encryption_enabled = IsEncrypted();
	SerializeEncryptionMetadata(ser, encryption_metadata, encryption_enabled);
	SerializeDBIdentifier(ser, db_identifier);
	SerializeEncryptionMetadata(ser, encrypted_canary, encryption_enabled);
	SerializeIV(ser, canary_iv, encryption_enabled);
	SerializeTag(ser, canary_tag, encryption_enabled);
}

void MainHeader::CheckMagicBytes(QueryContext context, FileHandle &handle) {
	data_t magic_bytes[MAGIC_BYTE_SIZE];
	if (handle.GetFileSize() < MainHeader::MAGIC_BYTE_SIZE + MainHeader::MAGIC_BYTE_OFFSET) {
		throw IOException("The file \"%s\" exists, but it is not a valid DuckDB database file!", handle.path);
	}
	handle.Read(context, magic_bytes, MainHeader::MAGIC_BYTE_SIZE, MainHeader::MAGIC_BYTE_OFFSET);
	if (memcmp(magic_bytes, MainHeader::MAGIC_BYTES, MainHeader::MAGIC_BYTE_SIZE) != 0) {
		throw IOException("The file \"%s\" exists, but it is not a valid DuckDB database file!", handle.path);
	}
}

MainHeader MainHeader::Read(ReadStream &source) {
	data_t magic_bytes[MAGIC_BYTE_SIZE];

	MainHeader header;
	source.ReadData(magic_bytes, MainHeader::MAGIC_BYTE_SIZE);
	if (memcmp(magic_bytes, MainHeader::MAGIC_BYTES, MainHeader::MAGIC_BYTE_SIZE) != 0) {
		throw IOException("The file is not a valid DuckDB database file!");
	}

	header.version_number = source.Read<uint64_t>();

	// Check the version number to determine if we can read this file.
	if ((header.version_number < VERSION_NUMBER_LOWER || header.version_number > VERSION_NUMBER_UPPER) &&
	    !IsBlockCompressedVersion(header.version_number)) {
		auto version = GetDuckDBVersions(header.version_number);
		string version_text;
		if (!version.empty()) {
			// Known version.
			version_text = "DuckDB version " + string(version);
		} else {
			version_text = string("an ") +
			               (VERSION_NUMBER_UPPER > header.version_number ? "older development" : "newer") +
			               string(" version of DuckDB");
		}
		throw IOException(
		    "Trying to read a database file with version number %lld, but we can only read versions between %lld and "
		    "%lld.\n"
		    "The database file was created with %s.\n\n"
		    "Newer DuckDB version might introduce backward incompatible changes (possibly guarded by compatibility "
		    "settings).\n"
		    "See the storage page for migration strategy and more information: https://duckdb.org/internals/storage",
		    header.version_number, VERSION_NUMBER_LOWER, VERSION_NUMBER_UPPER, version_text);
	}

	// Read the flags.
	for (idx_t i = 0; i < FLAG_COUNT; i++) {
		header.flags[i] = source.Read<uint64_t>();
	}
	DeserializeVersionNumber(source, header.library_git_desc);
	DeserializeVersionNumber(source, header.library_git_hash);

	// We always deserialize, and read zeros, if not set.
	DeserializeEncryptionData(source, header.encryption_metadata, MainHeader::ENCRYPTION_METADATA_LEN);
	DeserializeEncryptionData(source, header.db_identifier, MainHeader::DB_IDENTIFIER_LEN);
	DeserializeEncryptionData(source, header.encrypted_canary, MainHeader::CANARY_BYTE_SIZE);
	DeserializeEncryptionData(source, header.canary_iv, MainHeader::AES_NONCE_LEN);
	DeserializeEncryptionData(source, header.canary_tag, MainHeader::AES_TAG_LEN);

	return header;
}

void DatabaseHeader::Write(WriteStream &ser) {
	ser.Write<uint64_t>(iteration);
	ser.Write<idx_t>(meta_block);
	ser.Write<idx_t>(free_list);
	ser.Write<uint64_t>(block_count);
	ser.Write<idx_t>(block_alloc_size);
	ser.Write<idx_t>(vector_size);
	ser.Write<idx_t>(serialization_compatibility);
}

DatabaseHeader DatabaseHeader::Read(const MainHeader &main_header, ReadStream &source) {
	DatabaseHeader header;
	header.iteration = source.Read<uint64_t>();
	header.meta_block = source.Read<idx_t>();
	header.free_list = source.Read<idx_t>();
	header.block_count = source.Read<uint64_t>();
	header.block_alloc_size = source.Read<idx_t>();

	// backwards compatibility
	if (!header.block_alloc_size) {
		header.block_alloc_size = DEFAULT_BLOCK_ALLOC_SIZE;
	}

	header.vector_size = source.Read<idx_t>();
	if (!header.vector_size) {
		// backwards compatibility
		header.vector_size = DEFAULT_STANDARD_VECTOR_SIZE;
	}
	if (header.vector_size != STANDARD_VECTOR_SIZE) {
		throw IOException("Cannot read database file: DuckDB's compiled vector size is %llu bytes, but the file has a "
		                  "vector size of %llu bytes.",
		                  STANDARD_VECTOR_SIZE, header.vector_size);
	}

	// Default to 1 for version 64, else read from file.
	header.serialization_compatibility = main_header.version_number == 64 ? 1 : source.Read<idx_t>();

	return header;
}

template <class T>
void SerializeHeaderStructure(T header, data_ptr_t ptr) {
	MemoryStream ser(ptr, Storage::FILE_HEADER_SIZE);
	header.Write(ser);
}

MainHeader DeserializeMainHeader(data_ptr_t ptr) {
	MemoryStream source(ptr, Storage::FILE_HEADER_SIZE);
	return MainHeader::Read(source);
}

DatabaseHeader DeserializeDatabaseHeader(const MainHeader &main_header, data_ptr_t ptr) {
	MemoryStream source(ptr, Storage::FILE_HEADER_SIZE);
	return DatabaseHeader::Read(main_header, source);
}

//! The extent map position (offset, entries) a block-compressed database header stores after its other fields
pair<idx_t, idx_t> DeserializeExtentMapPosition(const MainHeader &main_header, data_ptr_t ptr) {
	MemoryStream source(ptr, Storage::FILE_HEADER_SIZE);
	DatabaseHeader::Read(main_header, source);
	if (!IsBlockCompressedVersion(main_header.version_number)) {
		return make_pair(idx_t(0), idx_t(0));
	}
	auto map_offset = source.Read<idx_t>();
	auto map_entries = source.Read<idx_t>();
	return make_pair(map_offset, map_entries);
}

SingleFileBlockManager::SingleFileBlockManager(AttachedDatabase &db_p, const string &path_p,
                                               const StorageManagerOptions &options)
    : BlockManager(BufferManager::GetBufferManager(db_p), options.block_alloc_size, options.block_header_size),
      db(db_p), path(path_p), header_buffer(BlockAllocator::Get(db_p), FileBufferType::MANAGED_BUFFER,
                                            Storage::FILE_HEADER_SIZE - options.block_header_size.GetIndex(),
                                            options.block_header_size.GetIndex()),
      iteration_count(0), options(options) {
}

bool SingleFileBlockManager::SplitDictionarySegments() const {
	return block_compression && options.version_number.IsValid() &&
	       IsReleaseStorageVersion(options.version_number.GetIndex());
}

bool SingleFileBlockManager::WritesStringMinNonEmpty() const {
	return block_compression && options.version_number.IsValid() &&
	       IsReleaseStorageVersion(options.version_number.GetIndex());
}

bool SingleFileBlockManager::PersistedRowGroupIndex() const {
	return kPersistedRowGroupIndex && block_compression && options.version_number.IsValid() &&
	       options.version_number.GetIndex() == PERSISTED_ROW_GROUP_INDEX_VERSION_NUMBER;
}

SingleFileBlockManager::~SingleFileBlockManager() {
	// flip the flag to not perform UnregisterBlock on the block manager that is being destructed
	this->in_destruction = true;
	if (writeback) {
		// stop the writeback thread, close the database handle, then the writeback descriptor: closing a descriptor of
		// the file releases the process's POSIX locks on it, so it must not close while the database handle is open
		writeback->Stop();
		handle.reset();
		writeback.reset();
	}
}

void SingleFileBlockManager::NotifyBlockWritten(idx_t bytes) {
#ifdef __linux__
	BlockWriteback *target;
	{
		lock_guard<mutex> guard(writeback_lock);
		if (!writeback_checked) {
			// once, at the first block write: a writable file on disk with buffered IO only
			writeback_checked = true;
			if (!options.read_only && !options.use_direct_io && handle && handle->OnDiskFile()) {
				auto fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
				if (fd >= 0) {
					writeback = make_uniq<BlockWriteback>(fd);
				}
			}
		}
		target = writeback.get();
	}
	if (target) {
		target->Add(bytes);
	}
#endif
}

FileOpenFlags SingleFileBlockManager::GetFileFlags(bool create_new) const {
	FileOpenFlags result;
	if (options.read_only) {
		D_ASSERT(!create_new);
		result = FileFlags::FILE_FLAGS_READ | FileFlags::FILE_FLAGS_NULL_IF_NOT_EXISTS | FileLockType::READ_LOCK;
	} else {
		result = FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_READ | FileLockType::WRITE_LOCK;
		if (create_new) {
			result |= FileFlags::FILE_FLAGS_FILE_CREATE;
		}
	}
	if (options.use_direct_io) {
		result |= FileFlags::FILE_FLAGS_DIRECT_IO;
	}
	// database files can be read from in parallel
	result |= FileFlags::FILE_FLAGS_PARALLEL_ACCESS;
	result |= FileFlags::FILE_FLAGS_MULTI_CLIENT_ACCESS;
	return result;
}

void SingleFileBlockManager::AddStorageVersionTag() {
	db.tags["storage_version"] = GetStorageVersionName(options.storage_version.GetIndex(), true);
}

uint64_t SingleFileBlockManager::GetVersionNumber() const {
	auto storage_version = options.storage_version.GetIndex();
	if (storage_version < 4) {
		return VERSION_NUMBER;
	}
	// Look up the matching version number.
	auto version_name = GetStorageVersionName(storage_version, false);
	return GetStorageVersion(version_name.c_str()).GetIndex();
}

MainHeader ConstructMainHeader(idx_t version_number) {
	MainHeader header;
	header.version_number = version_number;
	memset(header.flags, 0, sizeof(uint64_t) * MainHeader::FLAG_COUNT);
	return header;
}

void SingleFileBlockManager::StoreEncryptedCanary(AttachedDatabase &db, MainHeader &main_header, const string &key_id) {
	const_data_ptr_t key = EncryptionEngine::GetKeyFromCache(db.GetDatabase(), key_id);
	// Encrypt canary with the derived key
	shared_ptr<EncryptionState> encryption_state;
	auto encryption_version = static_cast<EncryptionTypes::EncryptionVersion>(main_header.GetEncryptionVersion());
	if (encryption_version > EncryptionTypes::V0_0 && encryption_version != EncryptionTypes::NONE) {
		// From Encryption Version 1+, always encrypt canary with GCM
		auto metadata = make_uniq<EncryptionStateMetadata>(
		    EncryptionTypes::GCM, MainHeader::DEFAULT_ENCRYPTION_KEY_LENGTH, encryption_version);
		encryption_state =
		    db.GetDatabase().GetEncryptionUtil(db.IsReadOnly())->CreateEncryptionState(std::move(metadata));
	} else {
		auto metadata = make_uniq<EncryptionStateMetadata>(
		    static_cast<EncryptionTypes::CipherType>(main_header.GetEncryptionCipher()),
		    MainHeader::DEFAULT_ENCRYPTION_KEY_LENGTH, encryption_version);
		encryption_state =
		    db.GetDatabase().GetEncryptionUtil(db.IsReadOnly())->CreateEncryptionState(std::move(metadata));
	}

	EncryptCanary(main_header, encryption_state, key);
}

void SingleFileBlockManager::StoreDBIdentifier(MainHeader &main_header, data_ptr_t db_identifier) {
	main_header.SetDBIdentifier(db_identifier);
}

template <typename T>
void SingleFileBlockManager::WriteEncryptionData(MemoryStream &stream, const T &val) {
	stream.WriteData(reinterpret_cast<const_data_ptr_t>(&val), sizeof(val));
}

void SingleFileBlockManager::StoreEncryptionMetadata(MainHeader &main_header) const {
	// The first byte is the key derivation function (kdf).
	// The second byte is for the usage of AAD.
	// The third byte is for the cipher.
	// The subsequent byte is empty.
	// The last 4 bytes are the key length.
	auto metadata_stream = make_uniq<MemoryStream>(ENCRYPTION_METADATA_LEN);

	WriteEncryptionData<uint8_t>(*metadata_stream, options.encryption_options.kdf);
	WriteEncryptionData<uint8_t>(*metadata_stream, options.encryption_options.additional_authenticated_data);
	WriteEncryptionData<uint8_t>(*metadata_stream, db.GetStorageManager().GetCipher());
	WriteEncryptionData<uint8_t>(*metadata_stream, options.encryption_options.encryption_version);
	WriteEncryptionData<uint32_t>(*metadata_stream, options.encryption_options.key_length);

	main_header.SetEncryptionMetadata(metadata_stream->GetData());
}

void SingleFileBlockManager::CheckAndAddEncryptionKey(MainHeader &main_header, string &user_key) {
	//! Get the database identifier.
	uint8_t db_identifier[MainHeader::DB_IDENTIFIER_LEN];
	memset(db_identifier, 0, MainHeader::DB_IDENTIFIER_LEN);
	memcpy(db_identifier, main_header.GetDBIdentifier(), MainHeader::DB_IDENTIFIER_LEN);

	//! Check if the correct key is used to decrypt the database
	// Derive the encryption key and add it to cache
	data_t derived_key[MainHeader::DEFAULT_ENCRYPTION_KEY_LENGTH];
	EncryptionKeyManager::DeriveKey(user_key, db_identifier, derived_key);

	shared_ptr<EncryptionState> encryption_state;
	auto encryption_version = static_cast<EncryptionTypes::EncryptionVersion>(main_header.GetEncryptionVersion());
	if (encryption_version > EncryptionTypes::V0_0 && encryption_version != EncryptionTypes::NONE) {
		// From Encryption Version 1+, always encrypt canary with GCM
		auto metadata = make_uniq<EncryptionStateMetadata>(
		    EncryptionTypes::GCM, MainHeader::DEFAULT_ENCRYPTION_KEY_LENGTH, encryption_version);
		encryption_state =
		    db.GetDatabase().GetEncryptionUtil(db.IsReadOnly())->CreateEncryptionState(std::move(metadata));
	} else {
		auto metadata = make_uniq<EncryptionStateMetadata>(
		    static_cast<EncryptionTypes::CipherType>(main_header.GetEncryptionCipher()),
		    MainHeader::DEFAULT_ENCRYPTION_KEY_LENGTH, encryption_version);
		encryption_state =
		    db.GetDatabase().GetEncryptionUtil(db.IsReadOnly())->CreateEncryptionState(std::move(metadata));
	}

	if (!DecryptCanary(main_header, encryption_state, derived_key)) {
		throw InvalidInputException("Wrong encryption key used to open the database file");
	}

	options.encryption_options.derived_key_id = EncryptionEngine::AddKeyToCache(db.GetDatabase(), derived_key);
	auto &catalog = db.GetCatalog().Cast<DuckCatalog>();
	catalog.SetEncryptionKeyId(options.encryption_options.derived_key_id);
	catalog.SetIsEncrypted();

	std::fill(user_key.begin(), user_key.end(), 0);
	user_key.clear();
}

void SingleFileBlockManager::CheckAndAddEncryptionKey(MainHeader &main_header) {
	return CheckAndAddEncryptionKey(main_header, *options.encryption_options.user_key);
}

void SingleFileBlockManager::CreateNewDatabase(QueryContext context) {
	auto flags = GetFileFlags(true);

	auto encryption_enabled = options.encryption_options.encryption_enabled;
	if (encryption_enabled) {
		// Check if we can read/write the encrypted database
		db.GetDatabase().GetEncryptionUtil(options.read_only);
	}

	// open the RDBMS handle
	auto &fs = FileSystem::Get(db);
	handle = fs.OpenFile(path, flags);
	// the file is read block by block at block offsets: no sequential read-ahead past a requested block
	LocalFileSystem::RandomAccessHint(*handle);
	header_buffer.Clear();

	options.version_number = GetVersionNumber();
	if (kBlockCompression && options.version_number.GetIndex() >= 68 && !encryption_enabled && !options.use_direct_io) {
		// a new file at the latest storage version stores its blocks compressed
		block_compression = true;
		options.version_number = PERSISTED_ROW_GROUP_INDEX_VERSION_NUMBER;
		scaled_frame_of_reference = true;
		next_extent_offset = BLOCK_START;
	}
	db.GetStorageManager().SetStorageVersion(options.storage_version.GetIndex());
	AddStorageVersionTag();

	MainHeader main_header = ConstructMainHeader(options.version_number.GetIndex());

	// Derive the encryption key and add it to the cache.
	// Not used for plain databases.
	data_t derived_key[MainHeader::DEFAULT_ENCRYPTION_KEY_LENGTH];

	// We need the unique database identifier, if the storage version is new enough.
	// If encryption is enabled, we also use it as the salt.
	memset(options.db_identifier, 0, MainHeader::DB_IDENTIFIER_LEN);
	if (encryption_enabled || options.version_number.GetIndex() >= 67) {
		GenerateDBIdentifier(options.db_identifier);
	}

	if (encryption_enabled) {
		// The key is given via ATTACH.
		EncryptionKeyManager::DeriveKey(*options.encryption_options.user_key, options.db_identifier, derived_key);
		options.encryption_options.user_key = nullptr;

		// if no encryption cipher is specified, use GCM
		if (db.GetStorageManager().GetCipher() == EncryptionTypes::INVALID) {
			db.GetStorageManager().SetCipher(EncryptionTypes::GCM);
		}

		// Set the encrypted DB bit to 1.
		main_header.SetEncrypted();

		if (options.encryption_options.encryption_version == EncryptionTypes::NONE) {
			throw InvalidConfigurationException("No Encryption type set");
		}

		main_header.SetEncryptionVersion(options.encryption_options.encryption_version);

		// The derived key is wiped in AddKeyToCache.
		options.encryption_options.derived_key_id = EncryptionEngine::AddKeyToCache(db.GetDatabase(), derived_key);
		auto &catalog = db.GetCatalog().Cast<DuckCatalog>();
		catalog.SetEncryptionKeyId(options.encryption_options.derived_key_id);
		catalog.SetIsEncrypted();
	}

	// Store all metadata in the main header.
	if (encryption_enabled) {
		StoreEncryptionMetadata(main_header);
	}
	// Always store the database identifier.
	StoreDBIdentifier(main_header, options.db_identifier);
	if (encryption_enabled) {
		StoreEncryptedCanary(db, main_header, options.encryption_options.derived_key_id);
	}

	// Write the main database header.
	SerializeHeaderStructure<MainHeader>(main_header, header_buffer.buffer);
	ChecksumAndWrite(context, header_buffer, 0, true);

	// write the database headers
	// initialize meta_block and free_list to INVALID_BLOCK because the database file does not contain any actual
	// content yet
	DatabaseHeader h1;
	// header 1
	h1.iteration = 0;
	h1.meta_block = idx_t(INVALID_BLOCK);
	h1.free_list = idx_t(INVALID_BLOCK);
	h1.block_count = 0;
	// We create the SingleFileBlockManager with the desired block allocation size before calling CreateNewDatabase.
	h1.block_alloc_size = GetBlockAllocSize();
	h1.vector_size = STANDARD_VECTOR_SIZE;
	h1.serialization_compatibility = options.storage_version.GetIndex();
	if (block_compression) {
		// the extent map position after the header fields reads as none
		header_buffer.Clear();
	}
	SerializeHeaderStructure<DatabaseHeader>(h1, header_buffer.buffer);
	ChecksumAndWrite(context, header_buffer, Storage::FILE_HEADER_SIZE);

	// header 2
	DatabaseHeader h2;
	h2.iteration = 0;
	h2.meta_block = idx_t(INVALID_BLOCK);
	h2.free_list = idx_t(INVALID_BLOCK);
	h2.block_count = 0;
	// We create the SingleFileBlockManager with the desired block allocation size before calling CreateNewDatabase.
	h2.block_alloc_size = GetBlockAllocSize();
	h2.vector_size = STANDARD_VECTOR_SIZE;
	h2.serialization_compatibility = options.storage_version.GetIndex();
	if (block_compression) {
		header_buffer.Clear();
		durable_header = h2;
	}
	SerializeHeaderStructure<DatabaseHeader>(h2, header_buffer.buffer);
	ChecksumAndWrite(context, header_buffer, Storage::FILE_HEADER_SIZE * 2ULL);

	// ensure that writing to disk is completed before returning
	handle->Sync();
	// we start with h2 as active_header, this way our initial write will be in h1
	iteration_count = 0;
	active_header = 1;
	max_block = 0;
}

void SingleFileBlockManager::LoadExistingDatabase(QueryContext context) {
	auto flags = GetFileFlags(false);

	// open the RDBMS handle
	auto &fs = FileSystem::Get(db);
	handle = fs.OpenFile(path, flags);
	if (!handle) {
		// this can only happen in read-only mode - as that is when we set FILE_FLAGS_NULL_IF_NOT_EXISTS
		throw IOException("Cannot open database \"%s\" in read-only mode: database does not exist", path);
	}
	// the file is read block by block at block offsets: no sequential read-ahead past a requested block
	LocalFileSystem::RandomAccessHint(*handle);

	MainHeader::CheckMagicBytes(context, *handle);
	// otherwise, we check the metadata of the file
	ReadAndChecksum(context, header_buffer, 0, true);

	uint64_t delta = 0;
	if (GetBlockHeaderSize() > DEFAULT_BLOCK_HEADER_STORAGE_SIZE) {
		delta = GetBlockHeaderSize() - DEFAULT_BLOCK_HEADER_STORAGE_SIZE;
	}

	MainHeader main_header = DeserializeMainHeader(header_buffer.buffer - delta);
	memcpy(options.db_identifier, main_header.GetDBIdentifier(), MainHeader::DB_IDENTIFIER_LEN);

	if (!main_header.IsEncrypted() && options.encryption_options.encryption_enabled) {
		throw CatalogException("A key is explicitly specified, but database \"%s\" is not encrypted", path);
		// database is not encrypted, but is tried to be opened with a key
	}

	if (main_header.IsEncrypted()) {
		auto &storage_manager = db.GetStorageManager();
		if (options.encryption_options.encryption_enabled) {
			//! Encryption is set
			D_ASSERT(db.GetStorageManager().IsEncrypted());
			options.encryption_options.encryption_version =
			    static_cast<EncryptionTypes::EncryptionVersion>(main_header.GetEncryptionVersion());

			//! Check if our encryption module can write, if not, we throw
			db.GetDatabase().GetEncryptionUtil(options.read_only);

			//! Check if the given key upon attach is correct
			// Derive the encryption key and add it to cache
			CheckAndAddEncryptionKey(main_header);
			// delete user key ptr
			options.encryption_options.user_key = nullptr;
		} else {
			// if encrypted, but no encryption key given
			throw CatalogException("Cannot open encrypted database \"%s\" without a key", path);
		}

		// if a cipher was provided, check if it is the same as in the config
		auto stored_cipher = static_cast<EncryptionTypes::CipherType>(main_header.GetEncryptionCipher());
		auto config_cipher = storage_manager.GetCipher();
		if (config_cipher != EncryptionTypes::INVALID && config_cipher != stored_cipher) {
			throw CatalogException("Cannot open encrypted database \"%s\" with a different cipher (%s) than the one "
			                       "used to create it (%s)",
			                       path, EncryptionTypes::CipherToString(config_cipher),
			                       EncryptionTypes::CipherToString(stored_cipher));
		}

		// This avoids the cipher from being downgrades by an attacker
		// FIXME: we likely want to have a proper validation
		// of the cipher used instead of this trick to avoid downgrades
		if (stored_cipher != EncryptionTypes::GCM) {
			if (config_cipher == EncryptionTypes::INVALID) {
				throw CatalogException(
				    "Cannot open encrypted database \"%s\" without explicitly specifying the "
				    "encryption cipher for security reasons. Please make sure you understand the security implications "
				    "and re-attach the database specifying the desired cipher.",
				    path);
			}
		}

		// this is ugly, but the storage manager does not know the cipher type before
		storage_manager.SetCipher(stored_cipher);
		// encryption version can be overridden by the serialized encryption version
		storage_manager.SetEncryptionVersion(
		    static_cast<EncryptionTypes::EncryptionVersion>(main_header.GetEncryptionVersion()));
	}

	options.version_number = main_header.version_number;
	block_compression = IsBlockCompressedVersion(main_header.version_number);
	scaled_frame_of_reference = IsReleaseStorageVersion(main_header.version_number);
	if (block_compression && (main_header.IsEncrypted() || options.use_direct_io)) {
		throw IOException("Cannot open database \"%s\": compressed blocks (storage version %llu) are not supported "
		                  "together with encryption or direct IO",
		                  path, main_header.version_number);
	}

	// read the database headers from disk
	DatabaseHeader h1;
	ReadAndChecksum(context, header_buffer, Storage::FILE_HEADER_SIZE);
	h1 = DeserializeDatabaseHeader(main_header, header_buffer.buffer);
	auto h1_map = DeserializeExtentMapPosition(main_header, header_buffer.buffer);

	DatabaseHeader h2;
	ReadAndChecksum(context, header_buffer, Storage::FILE_HEADER_SIZE * 2ULL);
	h2 = DeserializeDatabaseHeader(main_header, header_buffer.buffer);
	auto h2_map = DeserializeExtentMapPosition(main_header, header_buffer.buffer);

	// check the header with the highest iteration count
	if (h1.iteration > h2.iteration) {
		// h1 is active header
		active_header = 0;
		Initialize(h1, GetOptionalBlockAllocSize());
	} else {
		// h2 is active header
		active_header = 1;
		Initialize(h2, GetOptionalBlockAllocSize());
	}
	AddStorageVersionTag();
	if (block_compression) {
		// the extent map must be loaded before any block (the free list's metadata first) is read
		durable_header = active_header == 0 ? h1 : h2;
		auto &active_map = active_header == 0 ? h1_map : h2_map;
		LoadExtentMap(context, active_map.first, active_map.second);
	}
	LoadFreeList(context);
}

void SingleFileBlockManager::CheckChecksum(data_ptr_t start_ptr, uint64_t delta, bool skip_block_header) const {
	uint64_t stored_checksum;
	uint64_t computed_checksum;

	if (skip_block_header && delta > 0) {
		//! Even with encryption enabled, the main header should be plaintext
		stored_checksum = Load<uint64_t>(start_ptr);
		computed_checksum = Checksum(start_ptr + DEFAULT_BLOCK_HEADER_STORAGE_SIZE, GetBlockSize() + delta);
	} else {
		//! We do have to decrypt other headers
		stored_checksum = Load<uint64_t>(start_ptr + delta);
		computed_checksum = Checksum(start_ptr + GetBlockHeaderSize(), GetBlockSize());
	}

	// verify the checksum
	if (stored_checksum != computed_checksum) {
		throw IOException("Corrupt database file: computed checksum %llu does not match stored checksum %llu in block "
		                  "at location %llu",
		                  computed_checksum, stored_checksum, start_ptr);
	}
}

void SingleFileBlockManager::CheckChecksum(FileBuffer &block, uint64_t location, uint64_t delta,
                                           bool skip_block_header) const {
	uint64_t stored_checksum;
	uint64_t computed_checksum;

	if (skip_block_header && delta > 0) {
		//! Even with encryption enabled, the main header should be plaintext
		stored_checksum = Load<uint64_t>(block.InternalBuffer());
		computed_checksum = Checksum(block.buffer - delta, block.Size() + delta);
	} else {
		//! We do have to decrypt other headers
		stored_checksum = Load<uint64_t>(block.InternalBuffer() + delta);
		computed_checksum = Checksum(block.buffer, block.Size());
	}

	// verify the checksum
	if (stored_checksum != computed_checksum) {
		throw IOException("Corrupt database file: computed checksum %llu does not match stored checksum %llu in block "
		                  "at location %llu",
		                  computed_checksum, stored_checksum, location);
	}
}

void SingleFileBlockManager::ReadAndChecksum(QueryContext context, FileBuffer &block, uint64_t location,
                                             bool skip_block_header) const {
	// read the buffer from disk
	block.Read(context, *handle, location);

	//! calculate delta header bytes (if any)
	uint64_t delta = GetBlockHeaderSize() - Storage::DEFAULT_BLOCK_HEADER_SIZE;

	if (options.encryption_options.encryption_enabled && !skip_block_header) {
		auto key_id = options.encryption_options.derived_key_id;
		EncryptionEngine::DecryptBlock(db, key_id, block.InternalBuffer(), block.Size(), delta);
	}

	CheckChecksum(block, location, delta, skip_block_header);
}

void SingleFileBlockManager::ChecksumAndWrite(QueryContext context, FileBuffer &block, uint64_t location,
                                              bool skip_block_header) const {
	auto delta = GetBlockHeaderSize() - Storage::DEFAULT_BLOCK_HEADER_SIZE;
	uint64_t checksum;

	if (skip_block_header && delta > 0) {
		//! This happens only for the main database header
		//! We do not encrypt the main database header
		memmove(block.InternalBuffer() + Storage::DEFAULT_BLOCK_HEADER_SIZE, block.buffer, block.Size());
		//! zero out the last bytes of the block
		memset(block.InternalBuffer() + block.Size() + Storage::DEFAULT_BLOCK_HEADER_SIZE, 0, delta);
		checksum = Checksum(block.buffer - delta, block.Size() + delta);
		delta = 0;
	} else {
		checksum = Checksum(block.buffer, block.Size());
	}

	Store<uint64_t>(checksum, block.InternalBuffer() + delta);

	// encrypt if required
	unique_ptr<FileBuffer> temp_buffer_manager;
	if (options.encryption_options.encryption_enabled && !skip_block_header) {
		auto key_id = options.encryption_options.derived_key_id;
		temp_buffer_manager =
		    make_uniq<FileBuffer>(BlockAllocator::Get(db), block.GetBufferType(), block.Size(), GetBlockHeaderSize());
		EncryptionEngine::EncryptBlock(db, key_id, block, *temp_buffer_manager, delta);
		temp_buffer_manager->Write(context, *handle, location);
	} else {
		block.Write(context, *handle, location);
	}
}

void SingleFileBlockManager::Initialize(const DatabaseHeader &header, const optional_idx block_alloc_size) {
	free_list_id = header.free_list;
	meta_block = header.meta_block;
	iteration_count = header.iteration;
	max_block = NumericCast<block_id_t>(header.block_count);
	if (options.storage_version.IsValid()) {
		// storage version specified explicity - use requested storage version
		auto requested_compat_version = options.storage_version.GetIndex();
		if (requested_compat_version < header.serialization_compatibility) {
			throw InvalidInputException(
			    "Error opening \"%s\": cannot initialize database with storage version %d - which is lower than what "
			    "the database itself uses (%d). The storage version of an existing database cannot be lowered.",
			    path, requested_compat_version, header.serialization_compatibility);
		}
	} else {
		// load storage version from header
		options.storage_version = header.serialization_compatibility;
	}
	if (header.serialization_compatibility > SerializationCompatibility::Latest().serialization_version) {
		throw InvalidInputException(
		    "Error opening \"%s\": file was written with a storage version greater than the latest version supported "
		    "by this DuckDB instance. Try opening the file with a newer version of DuckDB.",
		    path);
	}

	db.GetStorageManager().SetStorageVersion(options.storage_version.GetIndex());

	if (block_alloc_size.IsValid() && block_alloc_size.GetIndex() != header.block_alloc_size) {
		throw InvalidInputException(
		    "Error opening \"%s\": cannot initialize the same database with a different block size: provided block "
		    "size: %llu, file block size: %llu",
		    path, GetBlockAllocSize(), header.block_alloc_size);
	}

	SetBlockAllocSize(header.block_alloc_size);
}

void SingleFileBlockManager::LoadFreeList(QueryContext context) {
	MetaBlockPointer free_pointer(free_list_id, 0);
	if (!free_pointer.IsValid()) {
		// no free list
		return;
	}
	MetadataReader reader(GetMetadataManager(), free_pointer, nullptr, BlockReaderType::REGISTER_BLOCKS);
	auto free_list_count = reader.Read<uint64_t>(context);
	free_list.clear();
	for (idx_t i = 0; i < free_list_count; i++) {
		auto block = reader.Read<block_id_t>(context);
		free_list.insert(block);
	}
	auto multi_use_blocks_count = reader.Read<uint64_t>(context);
	multi_use_blocks.clear();
	for (idx_t i = 0; i < multi_use_blocks_count; i++) {
		auto block_id = reader.Read<block_id_t>(context);
		auto usage_count = reader.Read<uint32_t>(context);
		multi_use_blocks[block_id] = usage_count;
	}
	GetMetadataManager().Read(reader);
	GetMetadataManager().MarkBlocksAsModified();
}

bool SingleFileBlockManager::IsRootBlock(MetaBlockPointer root) {
	return root.block_pointer == meta_block;
}

block_id_t SingleFileBlockManager::GetFreeBlockIdInternal(FreeBlockType type) {
	lock_guard<mutex> lock(single_file_block_lock);
	block_id_t block_id;
	if (!free_list.empty()) {
		// The free list is not empty, so we take its first element.
		block_id = *free_list.begin();
		// erase the entry from the free list again
		free_list.erase(free_list.begin());
	} else {
		block_id = max_block++;
	}
	// add the entry to the list of newly used blocks
	if (type == FreeBlockType::NEWLY_USED_BLOCK) {
		newly_used_blocks.insert(block_id);
	}
	if (BlockIsRegistered(block_id)) {
		throw InternalException("Free block %d is already registered", block_id);
	}
	return block_id;
}

block_id_t SingleFileBlockManager::GetFreeBlockId() {
	return GetFreeBlockIdInternal(FreeBlockType::NEWLY_USED_BLOCK);
}

block_id_t SingleFileBlockManager::GetFreeBlockIdForCheckpoint() {
	return GetFreeBlockIdInternal(FreeBlockType::CHECKPOINTED_BLOCK);
}

block_id_t SingleFileBlockManager::PeekFreeBlockId() {
	lock_guard<mutex> lock(single_file_block_lock);
	if (!free_list.empty()) {
		return *free_list.begin();
	} else {
		return max_block;
	}
}

void SingleFileBlockManager::MarkBlockAsCheckpointed(block_id_t block_id) {
	lock_guard<mutex> lock(single_file_block_lock);
	D_ASSERT(block_id >= 0);
	newly_used_blocks.erase(block_id);
}

void SingleFileBlockManager::MarkBlockAsUsed(block_id_t block_id) {
	lock_guard<mutex> lock(single_file_block_lock);
	D_ASSERT(block_id >= 0);
	if (max_block <= block_id) {
		// the block is past the current max_block
		// in this case we need to increment  "max_block" to "block_id"
		// any blocks in the middle are added to the free list
		// i.e. if max_block = 0, and block_id = 3, we need to add blocks 1 and 2 to the free list
		while (max_block < block_id) {
			free_list.insert(max_block);
			max_block++;
		}
		max_block++;
	} else if (free_list.find(block_id) != free_list.end()) {
		// block is currently in the free list - erase
		free_list.erase(block_id);
	} else {
		// block is already in use - increase reference count
		IncreaseBlockReferenceCountInternal(block_id);
	}
}

void SingleFileBlockManager::MarkBlockAsModified(block_id_t block_id) {
	unique_lock<mutex> lock(single_file_block_lock);
	D_ASSERT(block_id >= 0);
	D_ASSERT(block_id < max_block);

	// check if the block is a multi-use block
	auto entry = multi_use_blocks.find(block_id);
	if (entry != multi_use_blocks.end()) {
		// it is! reduce the reference count of the block
		entry->second--;
		// check the reference count: is the block still a multi-use block?
		if (entry->second <= 1) {
			// no longer a multi-use block!
			multi_use_blocks.erase(entry);
		}
		return;
	}
	// Check for multi-free
	if (modified_blocks.find(block_id) != modified_blocks.end()) {
		throw InternalException("MarkBlockAsModified called with already modified block id %d", block_id);
	}
	if (free_list.find(block_id) != free_list.end()) {
		throw InternalException("MarkBlockAsModified called with already freed block id %d", block_id);
	}
	auto newly_used_entry = newly_used_blocks.find(block_id);
	if (newly_used_entry != newly_used_blocks.end()) {
		// this block was newly used - and now we are labeling it as no longer being required
		// we can directly add it back to the free list
		newly_used_blocks.erase(block_id);
		AddFreeBlock(lock, block_id);
	} else {
		// this block was used in storage, we cannot directly re-use it
		// add it to the modified blocks indicating it will be re-usable after the next checkpoint
		modified_blocks.insert(block_id);
	}
}

void SingleFileBlockManager::IncreaseBlockReferenceCountInternal(block_id_t block_id) {
	D_ASSERT(block_id >= 0);
	D_ASSERT(block_id < max_block);
	D_ASSERT(free_list.find(block_id) == free_list.end());
	auto entry = multi_use_blocks.find(block_id);
	if (entry != multi_use_blocks.end()) {
		entry->second++;
	} else {
		multi_use_blocks[block_id] = 2;
	}
}

void SingleFileBlockManager::VerifyBlocks(const unordered_map<block_id_t, idx_t> &block_usage_count) {
	// probably don't need this?
	lock_guard<mutex> lock(single_file_block_lock);
	// all blocks should be accounted for - either in the block_usage_count, or in the free list
	set<block_id_t> referenced_blocks;
	for (auto &block : block_usage_count) {
		if (block.first == INVALID_BLOCK) {
			continue;
		}
		if (block.first >= max_block) {
			throw InternalException("Block %lld is used, but it is bigger than the max block %d", block.first,
			                        max_block);
		}
		referenced_blocks.insert(block.first);
		if (block.second > 1) {
			// multi-use block
			auto entry = multi_use_blocks.find(block.first);
			if (entry == multi_use_blocks.end()) {
				throw InternalException("Block %lld was used %llu times, but not present in multi_use_blocks",
				                        block.first, block.second);
			}
			if (entry->second != block.second) {
				throw InternalException(
				    "Block %lld was used %llu times, but multi_use_blocks says it is used %llu times", block.first,
				    block.second, entry->second);
			}
		} else {
			D_ASSERT(block.second > 0);
			auto entry = free_list.find(block.first);
			if (entry != free_list.end()) {
				throw InternalException("Block %lld was used, but it is present in the free list", block.first);
			}
		}
	}
	for (auto &newly_used_block : newly_used_blocks) {
		referenced_blocks.insert(newly_used_block);
	}
	for (auto &free_block : free_list) {
		referenced_blocks.insert(free_block);
	}
	for (auto &free_block : free_blocks_in_use) {
		referenced_blocks.insert(free_block);
	}
	if (referenced_blocks.size() != NumericCast<idx_t>(max_block)) {
		// not all blocks are accounted for
		string missing_blocks;
		for (block_id_t i = 0; i < max_block; i++) {
			if (referenced_blocks.find(i) == referenced_blocks.end()) {
				if (!missing_blocks.empty()) {
					missing_blocks += ", ";
				}
				missing_blocks += to_string(i);
			}
		}
		string free_list_str;
		for (auto &block : free_list) {
			if (!free_list_str.empty()) {
				free_list_str += ", ";
			}
			free_list_str += to_string(block);
		}
		string block_usage_str;
		for (auto &entry : block_usage_count) {
			if (!block_usage_str.empty()) {
				block_usage_str += ", ";
			}
			block_usage_str += to_string(entry.first);
		}
		string multi_use_blocks_str;
		for (auto &entry : multi_use_blocks) {
			if (!multi_use_blocks_str.empty()) {
				multi_use_blocks_str += ", ";
			}
			multi_use_blocks_str += to_string(entry.first);
		}
		string newly_used_blocks_str;
		for (auto &block : newly_used_blocks) {
			if (!newly_used_blocks_str.empty()) {
				newly_used_blocks_str += ", ";
			}
			newly_used_blocks_str += to_string(block);
		}

		throw InternalException(
		    "Block verification failed - blocks \"%s\" were not found as being used OR marked as free\nMax block: "
		    "%d\nBlock usage: %s\nFree list: %s\nMulti-use blocks: %s\nNewly used blocks: %s",
		    missing_blocks, max_block, block_usage_str, free_list_str, multi_use_blocks_str, newly_used_blocks_str);
	}
}

void SingleFileBlockManager::IncreaseBlockReferenceCount(block_id_t block_id) {
	lock_guard<mutex> lock(single_file_block_lock);
	IncreaseBlockReferenceCountInternal(block_id);
}

idx_t SingleFileBlockManager::GetMetaBlock() {
	return meta_block;
}

idx_t SingleFileBlockManager::TotalBlocks() {
	lock_guard<mutex> lock(single_file_block_lock);
	return NumericCast<idx_t>(max_block);
}

idx_t SingleFileBlockManager::FreeBlocks() {
	lock_guard<mutex> lock(single_file_block_lock);
	return free_list.size();
}

bool SingleFileBlockManager::IsRemote() {
	return !handle->OnDiskFile();
}

bool SingleFileBlockManager::Prefetch() {
	switch (Settings::Get<StorageBlockPrefetchSetting>(db.GetDatabase())) {
	case StorageBlockPrefetch::NEVER:
		return false;
	case StorageBlockPrefetch::DEBUG_FORCE_ALWAYS:
	case StorageBlockPrefetch::ALWAYS_PREFETCH:
		return !InMemory();
	case StorageBlockPrefetch::REMOTE_ONLY:
		return IsRemote();
	default:
		throw InternalException("Unknown StorageBlockPrefetch type");
	}
}

unique_ptr<Block> SingleFileBlockManager::ConvertBlock(block_id_t block_id, FileBuffer &source_buffer) {
	D_ASSERT(source_buffer.AllocSize() == GetBlockAllocSize());
	// FIXME; maybe we should pass the block header size explicitly
	return make_uniq<Block>(source_buffer, block_id, GetBlockHeaderSize());
}

unique_ptr<Block> SingleFileBlockManager::CreateBlock(block_id_t block_id, FileBuffer *source_buffer) {
	// FIXME; maybe we should pass the block header size explicitly
	unique_ptr<Block> result;
	if (source_buffer) {
		result = ConvertBlock(block_id, *source_buffer);
	} else {
		result = make_uniq<Block>(BlockAllocator::Get(db), block_id, *this);
	}
	result->Initialize(options.debug_initialize);
	return result;
}

void SingleFileBlockManager::ReadAhead(const vector<shared_ptr<BlockHandle>> &handles) {
	if (!handle || options.use_direct_io || IsRemote()) {
		return;
	}
	vector<block_id_t> block_ids;
	for (auto &block_handle : handles) {
		if (!block_handle || block_handle->BlockId() >= MAXIMUM_BLOCK) {
			continue;
		}
		if (block_handle->GetMemory().GetState() == BlockState::BLOCK_LOADED) {
			// already in memory: nothing to read
			continue;
		}
		block_ids.push_back(block_handle->BlockId());
	}
	if (block_ids.empty()) {
		return;
	}
	std::sort(block_ids.begin(), block_ids.end());
	block_ids.erase(std::unique(block_ids.begin(), block_ids.end()), block_ids.end());
	// each block's bytes in the file, as Read reads them: in the fixed layout the block's slot; in a compressed file
	// its extent (the block header, then the stored length), looked up in the extent map
	vector<pair<idx_t, idx_t>> ranges;
	ranges.reserve(block_ids.size());
	if (block_compression) {
		lock_guard<mutex> guard(extent_lock);
		if (extent_compacting) {
			// the extents are moving: the reads after the compaction go to the new places unhinted
			return;
		}
		for (auto block_id : block_ids) {
			auto index = NumericCast<idx_t>(block_id);
			if (index >= extents.size() || extents[index].offset == 0) {
				continue;
			}
			ranges.emplace_back(extents[index].offset, GetBlockHeaderSize() + extents[index].length);
		}
	} else {
		for (auto block_id : block_ids) {
			ranges.emplace_back(GetBlockLocation(block_id), GetBlockAllocSize());
		}
	}
	// Linux caps one read-ahead request at the device's read-ahead window (128 KiB by default) from its start, so a
	// longer range would be read only in part: hint every range in pieces of at most that window
	static constexpr idx_t READ_AHEAD_PIECE = 128 * 1024;
	for (auto &range : ranges) {
		for (idx_t offset = 0; offset < range.second; offset += READ_AHEAD_PIECE) {
			if (!LocalFileSystem::ReadAheadHint(*handle, range.first + offset,
			                                    MinValue(READ_AHEAD_PIECE, range.second - offset))) {
				return;
			}
		}
	}
}

idx_t SingleFileBlockManager::GetBlockLocation(block_id_t block_id) const {
	return BLOCK_START + NumericCast<idx_t>(block_id) * GetBlockAllocSize();
}

void SingleFileBlockManager::ReadBlock(data_ptr_t internal_buffer, uint64_t block_size, bool skip_block_header) const {
	//! calculate delta header bytes (if any)
	uint64_t delta = GetBlockHeaderSize() - Storage::DEFAULT_BLOCK_HEADER_SIZE;

	if (options.encryption_options.encryption_enabled && !skip_block_header) {
		EncryptionEngine::DecryptBlock(db, options.encryption_options.derived_key_id, internal_buffer, block_size,
		                               delta);
	}

	CheckChecksum(internal_buffer, delta, skip_block_header);
}

void SingleFileBlockManager::ReadBlock(Block &block, bool skip_block_header) const {
	// read the buffer from disk
	auto location = GetBlockLocation(block.id);
	block.Read(QueryContext(), *handle, location);

	//! calculate delta header bytes (if any)
	uint64_t delta = GetBlockHeaderSize() - Storage::DEFAULT_BLOCK_HEADER_SIZE;

	if (options.encryption_options.encryption_enabled && !skip_block_header) {
		EncryptionEngine::DecryptBlock(db, options.encryption_options.derived_key_id, block.InternalBuffer(),
		                               block.Size(), delta);
	}

	CheckChecksum(block, location, delta, skip_block_header);
}

void SingleFileBlockManager::Read(QueryContext context, Block &block) {
	D_ASSERT(block.id >= 0);
	D_ASSERT(std::find(free_list.begin(), free_list.end(), block.id) == free_list.end());
	if (block_compression) {
		D_ASSERT(block.AllocSize() == GetBlockAllocSize());
		ReadCompressedBlock(context, block.InternalBuffer(), block.id);
		return;
	}
	ReadAndChecksum(context, block, GetBlockLocation(block.id));
}

void SingleFileBlockManager::ReadBlocks(FileBuffer &buffer, block_id_t start_block, idx_t block_count) {
	D_ASSERT(start_block >= 0);
	D_ASSERT(block_count >= 1);
	if (block_compression) {
		// consecutive ids are not contiguous in the file: one read (and decompression) per block
		for (idx_t i = 0; i < block_count; i++) {
			ReadCompressedBlock(QueryContext(), buffer.InternalBuffer() + i * GetBlockAllocSize(),
			                    start_block + NumericCast<block_id_t>(i));
		}
		return;
	}

	// read the buffer from disk
	auto location = GetBlockLocation(start_block);
	buffer.Read(QueryContext(), *handle, location);

	// for each of the blocks - verify the checksum
	auto ptr = buffer.InternalBuffer();
	for (idx_t i = 0; i < block_count; i++) {
		auto start_ptr = ptr + i * GetBlockAllocSize();
		ReadBlock(start_ptr, GetBlockSize());
	}
}

void SingleFileBlockManager::Write(FileBuffer &buffer, block_id_t block_id) {
	Write(QueryContext(), buffer, block_id);
}

void SingleFileBlockManager::Write(QueryContext context, FileBuffer &buffer, block_id_t block_id) {
	D_ASSERT(block_id >= 0);
	if (block_compression) {
		WriteCompressedBlock(context, buffer, block_id);
		return;
	}
	ChecksumAndWrite(context, buffer, BLOCK_START + NumericCast<idx_t>(block_id) * GetBlockAllocSize());
	NotifyBlockWritten(buffer.AllocSize());
}

uint64_t SingleFileBlockManager::AllocateExtent(idx_t bytes) {
	lock_guard<mutex> guard(extent_lock);
	return AllocateExtentLocked(AlignValue<idx_t>(bytes, BLOCK_EXTENT_ALIGNMENT));
}

uint64_t SingleFileBlockManager::AllocateExtentLocked(idx_t size) {
	// the smallest free range that fits, else the end of the file
	auto entry = free_by_size.lower_bound(size);
	if (entry == free_by_size.end()) {
		auto offset = next_extent_offset;
		next_extent_offset += size;
		return offset;
	}
	auto offset = entry->second;
	TakeFreeSpaceLocked(offset, size);
	return offset;
}

void SingleFileBlockManager::SetFreeSpaceLocked(const vector<pair<uint64_t, idx_t>> &ranges) {
	free_by_offset.clear();
	free_by_size.clear();
	for (auto &range : ranges) {
		if (range.second == 0) {
			continue;
		}
		free_by_offset[range.first] = range.second;
		free_by_size.insert(make_pair(range.second, range.first));
	}
}

void SingleFileBlockManager::TakeFreeSpaceLocked(uint64_t offset, idx_t size) {
	// [offset, offset + size) lies inside one free range: remove it, keeping what is left on either side
	auto entry = free_by_offset.upper_bound(offset);
	if (entry == free_by_offset.begin()) {
		throw InternalException("Extent allocation: %llu is not free", offset);
	}
	--entry;
	auto range_offset = entry->first;
	auto range_size = entry->second;
	if (offset + size > range_offset + range_size) {
		throw InternalException("Extent allocation: [%llu, +%llu) is not free", offset, size);
	}
	free_by_offset.erase(entry);
	auto sized = free_by_size.equal_range(range_size);
	for (auto it = sized.first; it != sized.second; ++it) {
		if (it->second == range_offset) {
			free_by_size.erase(it);
			break;
		}
	}
	if (offset > range_offset) {
		free_by_offset[range_offset] = offset - range_offset;
		free_by_size.insert(make_pair(offset - range_offset, range_offset));
	}
	auto tail = range_offset + range_size - (offset + size);
	if (tail > 0) {
		free_by_offset[offset + size] = tail;
		free_by_size.insert(make_pair(tail, offset + size));
	}
}

void SingleFileBlockManager::WriteCompressedBlock(QueryContext context, FileBuffer &buffer, block_id_t block_id) {
	auto header_size = GetBlockHeaderSize();
	auto payload_size = GetBlockSize();
	if (buffer.Size() != payload_size || buffer.GetHeaderSize() != header_size) {
		throw InternalException("Compressed block write of block %lld: buffer size %llu (header %llu) is not the "
		                        "block size %llu (header %llu)",
		                        block_id, buffer.Size(), buffer.GetHeaderSize(), payload_size, header_size);
	}

	// the block header holds the checksum of the uncompressed payload, as in the fixed layout
	Store<uint64_t>(Checksum(buffer.buffer, payload_size), buffer.InternalBuffer());

	auto &state = GetBlockCompressionState();
	if (!state.cctx) {
		state.cctx = duckdb_zstd::ZSTD_createCCtx();
		if (!state.cctx) {
			throw InternalException("Failed to create a zstd compression context");
		}
	}
	auto bound = duckdb_zstd::ZSTD_compressBound(payload_size);
	auto staging = state.Staging(header_size + bound);
	auto level = NumericCast<int>(Settings::Get<ZstdBlockCompressionLevelSetting>(db.GetDatabase()));
	if (level == 0) {
		// automatic: the higher level pays for its compression time only when enough threads write in parallel
		level = TaskScheduler::GetScheduler(db.GetDatabase()).NumberOfThreads() >= BLOCK_COMPRESSION_HIGH_LEVEL_THREADS
		            ? BLOCK_COMPRESSION_HIGH_LEVEL
		            : BLOCK_COMPRESSION_LOW_LEVEL;
	}
	auto frame_size = duckdb_zstd::ZSTD_compressCCtx(state.cctx, staging + header_size, bound, buffer.buffer,
	                                                 payload_size, level);

	data_ptr_t source;
	idx_t length;
	if (!duckdb_zstd::ZSTD_isError(frame_size) &&
	    AlignValue<idx_t>(header_size + frame_size, BLOCK_EXTENT_ALIGNMENT) <
	        AlignValue<idx_t>(header_size + payload_size, BLOCK_EXTENT_ALIGNMENT)) {
		// the frame saves at least one page: store it
		memcpy(staging, buffer.InternalBuffer(), header_size);
		source = staging;
		length = frame_size;
	} else {
		// store the payload raw (byte-identical to a block of the fixed layout)
		source = buffer.InternalBuffer();
		length = payload_size;
	}
	auto bytes = header_size + length;
	uint64_t offset;
	{
		unique_lock<mutex> guard(extent_lock);
		extent_cv.wait(guard, [&]() { return !extent_compacting; });
		offset = AllocateExtentLocked(AlignValue<idx_t>(bytes, BLOCK_EXTENT_ALIGNMENT));
		extent_io++;
	}
	try {
		handle->Write(context, source, bytes, offset);
	} catch (...) {
		EndExtentIO();
		throw;
	}
	{
		lock_guard<mutex> guard(extent_lock);
		auto index = NumericCast<idx_t>(block_id);
		if (index >= extents.size()) {
			extents.resize(MaxValue<idx_t>(index + 1, extents.size() * 2));
		}
		extents[index].offset = offset;
		extents[index].length = NumericCast<uint32_t>(length);
	}
	EndExtentIO();
	NotifyBlockWritten(bytes);
}

void SingleFileBlockManager::EndExtentIO() {
	lock_guard<mutex> guard(extent_lock);
	D_ASSERT(extent_io > 0);
	if (--extent_io == 0 && extent_compacting) {
		extent_cv.notify_all();
	}
}

void SingleFileBlockManager::ReadCompressedBlock(QueryContext context, data_ptr_t internal_buffer,
                                                 block_id_t block_id) {
	BlockExtent extent;
	{
		unique_lock<mutex> guard(extent_lock);
		extent_cv.wait(guard, [&]() { return !extent_compacting; });
		auto index = NumericCast<idx_t>(block_id);
		if (index < extents.size()) {
			extent = extents[index];
		}
		if (extent.offset == 0) {
			throw IOException("Corrupt database file: block %lld has no extent in \"%s\"", block_id, path);
		}
		extent_io++;
	}
	try {
		ReadExtent(context, internal_buffer, block_id, extent);
	} catch (...) {
		EndExtentIO();
		throw;
	}
	EndExtentIO();
	CheckChecksum(internal_buffer, 0, false);
}

void SingleFileBlockManager::ReadExtent(QueryContext context, data_ptr_t internal_buffer, block_id_t block_id,
                                        const BlockExtent &extent) {
	auto header_size = GetBlockHeaderSize();
	auto payload_size = GetBlockSize();
	if (extent.length == payload_size) {
		// raw extent: read straight into the buffer
		handle->Read(context, internal_buffer, header_size + payload_size, extent.offset);
	} else {
		auto &state = GetBlockCompressionState();
		if (!state.dctx) {
			state.dctx = duckdb_zstd::ZSTD_createDCtx();
			if (!state.dctx) {
				throw InternalException("Failed to create a zstd decompression context");
			}
		}
		auto staging = state.Staging(header_size + extent.length);
		handle->Read(context, staging, header_size + extent.length, extent.offset);
		memcpy(internal_buffer, staging, header_size);
		auto size = duckdb_zstd::ZSTD_decompressDCtx(state.dctx, internal_buffer + header_size, payload_size,
		                                             staging + header_size, extent.length);
		if (duckdb_zstd::ZSTD_isError(size) || size != payload_size) {
			throw IOException("Corrupt database file: block %lld at location %llu in \"%s\" failed to decompress",
			                  block_id, extent.offset, path);
		}
	}
}

void SingleFileBlockManager::WriteExtentMap(QueryContext context, idx_t block_count,
                                            const set<block_id_t> &free_blocks, optional_idx at_offset) {
	idx_t entries;
	unsafe_unique_array<data_t> map;
	idx_t map_size;
	{
		lock_guard<mutex> guard(extent_lock);
		entries = MinValue<idx_t>(block_count, extents.size());
		map_size = EXTENT_MAP_HEADER_SIZE + entries * EXTENT_MAP_ENTRY_SIZE;
		map = make_unsafe_uniq_array<data_t>(map_size);
		auto ptr = map.get() + EXTENT_MAP_HEADER_SIZE;
		for (idx_t i = 0; i < entries; i++) {
			// a free id has no extent in the committed state
			auto is_free = free_blocks.find(NumericCast<block_id_t>(i)) != free_blocks.end();
			Store<uint64_t>(is_free ? 0 : extents[i].offset, ptr);
			Store<uint32_t>(is_free ? 0 : extents[i].length, ptr + sizeof(uint64_t));
			ptr += EXTENT_MAP_ENTRY_SIZE;
		}
	}
	memcpy(map.get(), EXTENT_MAP_MAGIC, sizeof(uint64_t));
	Store<uint64_t>(entries, map.get() + 8);
	Store<uint64_t>(Checksum(map.get() + EXTENT_MAP_HEADER_SIZE, entries * EXTENT_MAP_ENTRY_SIZE), map.get() + 16);
	Store<uint64_t>(GetBlockAllocSize(), map.get() + 24);
	auto offset = at_offset.IsValid() ? at_offset.GetIndex() : AllocateExtent(map_size);
	handle->Write(context, map.get(), map_size, offset);
	extent_map_offset = offset;
	extent_map_entries = entries;
}

void SingleFileBlockManager::CompactAfterCommit(QueryContext context, const set<block_id_t> &free_blocks) {
	{
		unique_lock<mutex> guard(extent_lock);
		// hold back new block IO and wait for the IO in flight: no extent is taken, read or written while extents move
		extent_compacting = true;
		extent_cv.wait(guard, [&]() { return extent_io == 0; });
		set<block_id_t> free_ids;
		{
			lock_guard<mutex> free_guard(single_file_block_lock);
			free_ids = free_list;
		}
		struct LiveExtent {
			uint64_t offset;
			idx_t size;
			idx_t index;
		};
		vector<LiveExtent> live;
		for (idx_t i = 0; i < extents.size(); i++) {
			if (extents[i].offset == 0 || free_ids.count(NumericCast<block_id_t>(i))) {
				continue;
			}
			live.push_back({extents[i].offset,
			                AlignValue<idx_t>(GetBlockHeaderSize() + extents[i].length, BLOCK_EXTENT_ALIGNMENT), i});
		}
		std::sort(live.begin(), live.end(), [](const LiveExtent &a, const LiveExtent &b) { return a.offset < b.offset; });
		struct Hole {
			uint64_t offset;
			idx_t size;
		};
		// the holes: the free space the rebuild after this commit found (nothing durable references it)
		vector<Hole> holes;
		for (auto &range : free_by_offset) {
			holes.push_back({range.first, range.second});
		}
		compact_moved = 0;
		// from the last extent backwards: into the smallest hole before it that fits; a vacated extent stays as it is
		// (the durable map references it until the header rewrite below), and a hole found at or after the extent lies
		// after every extent still to come, so it leaves the index
		multimap<idx_t, idx_t> by_size;
		for (idx_t h = 0; h < holes.size(); h++) {
			by_size.insert(make_pair(holes[h].size, h));
		}
		unsafe_unique_array<data_t> buffer;
		idx_t buffer_size = 0;
		for (idx_t l = live.size(); l > 0; l--) {
			auto &extent = live[l - 1];
			auto entry = by_size.lower_bound(extent.size);
			while (entry != by_size.end() && holes[entry->second].offset >= extent.offset) {
				entry = by_size.erase(entry);
			}
			if (entry == by_size.end()) {
				continue;
			}
			auto hole_index = entry->second;
			by_size.erase(entry);
			auto &hole = holes[hole_index];
			auto bytes = GetBlockHeaderSize() + extents[extent.index].length;
			if (bytes > buffer_size) {
				buffer = make_unsafe_uniq_array_uninitialized<data_t>(bytes);
				buffer_size = bytes;
			}
			handle->Read(context, buffer.get(), bytes, extent.offset);
			handle->Write(context, buffer.get(), bytes, hole.offset);
			TakeFreeSpaceLocked(hole.offset, extent.size);
			NotifyBlockWritten(bytes);
			extents[extent.index].offset = hole.offset;
			extent.offset = hole.offset;
			hole.offset += extent.size;
			hole.size -= extent.size;
			if (hole.size > 0) {
				by_size.insert(make_pair(hole.size, hole_index));
			}
			compact_moved++;
		}
		extent_compacting = false;
		extent_cv.notify_all();
	}
	if (compact_moved == 0) {
		RelocateExtentMapLow(context, free_blocks);
		return;
	}
	// durable: the moved extents and a new extent map, then the active header pointing at it; only then are the vacated
	// extents and the previous map free (the rebuild), and the tail they leave is cut
	idx_t block_count;
	{
		lock_guard<mutex> guard(single_file_block_lock);
		block_count = NumericCast<idx_t>(max_block);
	}
	WriteExtentMap(context, block_count, free_blocks);
	handle->Sync();
	WriteActiveHeaderInPlace(context);
	RebuildFreeExtents();
	RelocateExtentMapLow(context, free_blocks);
}

void SingleFileBlockManager::RelocateExtentMapLow(QueryContext context, const set<block_id_t> &free_blocks) {
	idx_t block_count;
	{
		lock_guard<mutex> guard(single_file_block_lock);
		block_count = NumericCast<idx_t>(max_block);
	}
	uint64_t target;
	{
		lock_guard<mutex> guard(extent_lock);
		// room for block_count entries: WriteExtentMap writes at most that many
		auto map_size =
		    AlignValue<idx_t>(EXTENT_MAP_HEADER_SIZE + block_count * EXTENT_MAP_ENTRY_SIZE, BLOCK_EXTENT_ALIGNMENT);
		auto current_size =
		    AlignValue<idx_t>(EXTENT_MAP_HEADER_SIZE + extent_map_entries * EXTENT_MAP_ENTRY_SIZE, BLOCK_EXTENT_ALIGNMENT);
		if (extent_map_offset == 0 || extent_map_offset + current_size != next_extent_offset) {
			// the map does not end the file: nothing it pins
			return;
		}
		// the lowest free range that holds the map, before the map
		optional_idx found;
		for (auto &range : free_by_offset) {
			if (range.first >= extent_map_offset) {
				break;
			}
			if (range.second >= map_size) {
				found = range.first;
				break;
			}
		}
		if (!found.IsValid()) {
			return;
		}
		target = found.GetIndex();
		TakeFreeSpaceLocked(target, map_size);
	}
	WriteExtentMap(context, block_count, free_blocks, target);
	handle->Sync();
	WriteActiveHeaderInPlace(context);
	RebuildFreeExtents();
}

void SingleFileBlockManager::RebuildFreeExtents() {
	unique_lock<mutex> guard(extent_lock);
	extent_compacting = true;
	extent_cv.wait(guard, [&]() { return extent_io == 0; });
	set<block_id_t> free_ids;
	block_id_t block_limit;
	{
		lock_guard<mutex> free_guard(single_file_block_lock);
		free_ids = free_list;
		block_limit = max_block;
	}
	// held: every extent of a block id that is not free (live in the committed state, or in use), and the committed
	// extent map; everything else is free, including what this commit released
	vector<pair<uint64_t, idx_t>> held;
	for (idx_t i = 0; i < extents.size(); i++) {
		if (i >= NumericCast<idx_t>(block_limit) || free_ids.count(NumericCast<block_id_t>(i))) {
			// a free id holds no extent: its old one is free space now, and a later write gives it a new one
			extents[i] = BlockExtent();
			continue;
		}
		if (extents[i].offset == 0) {
			continue;
		}
		held.emplace_back(extents[i].offset,
		                  AlignValue<idx_t>(GetBlockHeaderSize() + extents[i].length, BLOCK_EXTENT_ALIGNMENT));
	}
	if (extent_map_offset != 0) {
		held.emplace_back(extent_map_offset, AlignValue<idx_t>(EXTENT_MAP_HEADER_SIZE + extent_map_entries *
		                                                                                   EXTENT_MAP_ENTRY_SIZE,
		                                                       BLOCK_EXTENT_ALIGNMENT));
	}
	std::sort(held.begin(), held.end());
	vector<pair<uint64_t, idx_t>> ranges;
	uint64_t position = BLOCK_START;
	for (auto &range : held) {
		if (range.first > position) {
			ranges.emplace_back(position, range.first - position);
		}
		position = MaxValue<uint64_t>(position, range.first + range.second);
	}
	// a free tail is cut from the file
	if (position < next_extent_offset) {
		handle->Truncate(NumericCast<int64_t>(position));
		next_extent_offset = position;
	}
	SetFreeSpaceLocked(ranges);
	extent_compacting = false;
	extent_cv.notify_all();
}

void SingleFileBlockManager::LoadExtentMap(QueryContext context, idx_t map_offset, idx_t map_entries) {
	lock_guard<mutex> guard(extent_lock);
	extents.clear();
	auto file_end = AlignValue<idx_t>(MaxValue<idx_t>(handle->GetFileSize(), BLOCK_START), BLOCK_EXTENT_ALIGNMENT);
	if (map_offset == 0) {
		// no checkpoint has written blocks yet
		next_extent_offset = file_end;
		return;
	}
	auto map_size = EXTENT_MAP_HEADER_SIZE + map_entries * EXTENT_MAP_ENTRY_SIZE;
	if (map_offset < BLOCK_START || map_offset + map_size > handle->GetFileSize()) {
		throw IOException("Corrupt database file: the extent map of \"%s\" lies outside the file", path);
	}
	auto map = make_unsafe_uniq_array_uninitialized<data_t>(map_size);
	handle->Read(context, map.get(), map_size, map_offset);
	if (memcmp(map.get(), EXTENT_MAP_MAGIC, sizeof(uint64_t)) != 0 || Load<uint64_t>(map.get() + 8) != map_entries ||
	    Load<uint64_t>(map.get() + 24) != GetBlockAllocSize() ||
	    Load<uint64_t>(map.get() + 16) !=
	        Checksum(map.get() + EXTENT_MAP_HEADER_SIZE, map_entries * EXTENT_MAP_ENTRY_SIZE)) {
		throw IOException("Corrupt database file: the extent map of \"%s\" does not match its header", path);
	}
	extents.resize(map_entries);
	auto ptr = map.get() + EXTENT_MAP_HEADER_SIZE;
	for (idx_t i = 0; i < map_entries; i++) {
		extents[i].offset = Load<uint64_t>(ptr);
		extents[i].length = Load<uint32_t>(ptr + sizeof(uint64_t));
		ptr += EXTENT_MAP_ENTRY_SIZE;
	}
	// the free space: every byte from the first block to the end of the file that neither a block of the loaded map
	// (whose free ids hold no extent) nor the map itself holds
	vector<pair<uint64_t, idx_t>> held;
	for (auto &extent : extents) {
		if (extent.offset != 0) {
			held.emplace_back(extent.offset,
			                  AlignValue<idx_t>(GetBlockHeaderSize() + extent.length, BLOCK_EXTENT_ALIGNMENT));
		}
	}
	held.emplace_back(map_offset, AlignValue<idx_t>(map_size, BLOCK_EXTENT_ALIGNMENT));
	std::sort(held.begin(), held.end());
	vector<pair<uint64_t, idx_t>> ranges;
	uint64_t position = BLOCK_START;
	for (auto &range : held) {
		if (range.first > position) {
			ranges.emplace_back(position, range.first - position);
		}
		position = MaxValue<uint64_t>(position, range.first + range.second);
	}
	if (position < file_end) {
		ranges.emplace_back(position, file_end - position);
	}
	SetFreeSpaceLocked(ranges);
	extent_map_offset = map_offset;
	extent_map_entries = map_entries;
	next_extent_offset = file_end;
}

void SingleFileBlockManager::Truncate() {
	BlockManager::Truncate();

	lock_guard<mutex> guard(single_file_block_lock);
	idx_t blocks_to_truncate = 0;
	// reverse iterate over the free-list
	for (auto entry = free_list.rbegin(); entry != free_list.rend(); entry++) {
		auto block_id = *entry;
		if (block_id + 1 != max_block) {
			break;
		}
		blocks_to_truncate++;
		max_block--;
	}
	if (blocks_to_truncate == 0) {
		// nothing to truncate
		return;
	}
	// truncate the file
	free_list.erase(free_list.lower_bound(max_block), free_list.end());
	if (block_compression) {
		// block positions are extents: only the ids are trimmed (the append-only extents stay)
		return;
	}
	handle->Truncate(NumericCast<int64_t>(BLOCK_START + NumericCast<idx_t>(max_block) * GetBlockAllocSize()));
}

vector<MetadataHandle> SingleFileBlockManager::GetFreeListBlocks() {
	vector<MetadataHandle> free_list_blocks;
	auto &metadata_manager = GetMetadataManager();

	// reserve all blocks that we are going to write the free list to
	// since these blocks are no longer free we cannot just include them in the free list!
	auto block_size = metadata_manager.GetMetadataBlockSize() - sizeof(idx_t);
	idx_t allocated_size = 0;
	while (true) {
		idx_t free_list_count;
		idx_t multi_use_blocks_count;
		{
			lock_guard<mutex> guard(single_file_block_lock);
			free_list_count =
			    free_list.size() + modified_blocks.size() + free_blocks_in_use.size() + newly_used_blocks.size();
			multi_use_blocks_count = multi_use_blocks.size();
		}
		auto free_list_size = sizeof(uint64_t) + sizeof(block_id_t) * free_list_count;
		auto multi_use_blocks_size =
		    sizeof(uint64_t) + (sizeof(block_id_t) + sizeof(uint32_t)) * multi_use_blocks_count;
		auto metadata_blocks =
		    sizeof(uint64_t) + (sizeof(block_id_t) + sizeof(idx_t)) * GetMetadataManager().BlockCount();
		auto total_size = free_list_size + multi_use_blocks_size + metadata_blocks;
		if (total_size < allocated_size) {
			break;
		}
		auto free_list_handle = GetMetadataManager().AllocateHandle();
		free_list_blocks.push_back(std::move(free_list_handle));
		allocated_size += block_size;
	}

	return free_list_blocks;
}

class FreeListBlockWriter : public MetadataWriter {
public:
	FreeListBlockWriter(MetadataManager &manager, vector<MetadataHandle> free_list_blocks_p)
	    : MetadataWriter(manager), free_list_blocks(std::move(free_list_blocks_p)), index(0) {
	}

	vector<MetadataHandle> free_list_blocks;
	idx_t index;

protected:
	MetadataHandle NextHandle() override {
		if (index >= free_list_blocks.size()) {
			throw InternalException("Free List Block Writer ran out of blocks, this means not enough blocks were "
			                        "allocated up front (%d total allocated)",
			                        free_list_blocks.size());
		}
		return std::move(free_list_blocks[index++]);
	}
};

bool SingleFileBlockManager::AddFreeBlock(unique_lock<mutex> &lock, block_id_t block_id) {
	if (!lock.owns_lock()) {
		throw InternalException("AddFreeBlock must be called while holding the lock");
	}
	shared_ptr<BlockHandle> block = TryGetBlock(block_id);
	if (!block) {
		// the block does not exist
		// regular free block
		free_list.insert(block_id);
		return true;
	}
	// the block exists - add to blocks in use
	free_blocks_in_use.insert(block_id);

	// release the lock while destroying the block since the block destructor can call UnregisterBlock
	lock.unlock();
	block.reset();
	lock.lock();
	return false;
}
void SingleFileBlockManager::WriteHeader(QueryContext context, DatabaseHeader header) {
	unique_lock<mutex> header_guard(header_lock, std::defer_lock);
	if (block_compression) {
		header_guard.lock();
	}
	auto free_list_blocks = GetFreeListBlocks();

	// now handle the free list
	auto &metadata_manager = GetMetadataManager();
	// add all modified blocks to the free list: they can now be written to again
	metadata_manager.MarkBlocksAsModified();

	unique_lock<mutex> lock(single_file_block_lock);
	// set the iteration count
	header.iteration = ++iteration_count;

	set<block_id_t> all_free_blocks = free_list;
	set<block_id_t> fully_freed_blocks;
	for (auto &block : modified_blocks) {
		all_free_blocks.insert(block);
		if (AddFreeBlock(lock, block)) {
			fully_freed_blocks.insert(block);
		}
	}
	auto written_multi_use_blocks = multi_use_blocks;
	// newly used blocks are still free blocks for this checkpoint - so add them to the free list that we write
	for (auto &newly_used_block : newly_used_blocks) {
		all_free_blocks.insert(newly_used_block);
		written_multi_use_blocks.erase(newly_used_block);
	}
	modified_blocks.clear();

	if (!free_list_blocks.empty()) {
		// there are blocks to write, either in the free_list or in the modified_blocks
		// we write these blocks specifically to the free_list_blocks
		// a normal MetadataWriter will fetch blocks to use from the free_list
		// but since we are WRITING the free_list, this behavior is sub-optimal
		FreeListBlockWriter writer(metadata_manager, std::move(free_list_blocks));

		auto ptr = writer.GetMetaBlockPointer();
		header.free_list = ptr.block_pointer;

		writer.Write<uint64_t>(all_free_blocks.size());
		for (auto &block_id : all_free_blocks) {
			writer.Write<block_id_t>(block_id);
		}
		writer.Write<uint64_t>(written_multi_use_blocks.size());
		for (auto &entry : written_multi_use_blocks) {
			writer.Write<block_id_t>(entry.first);
			writer.Write<uint32_t>(entry.second);
		}
		GetMetadataManager().Write(writer);
		writer.Flush();
	} else {
		// no blocks in the free list
		header.free_list = DConstants::INVALID_INDEX;
	}
	lock.unlock();
	metadata_manager.Flush();

	lock.lock();
	header.block_count = NumericCast<idx_t>(max_block);
	lock.unlock();

	if (block_compression) {
		// the extent map of every block the header can reference, written (and synced) before the header
		WriteExtentMap(context, header.block_count, all_free_blocks);
	}

	header.serialization_compatibility = options.storage_version.GetIndex();

	auto debug_checkpoint_abort = Settings::Get<DebugCheckpointAbortSetting>(db.GetDatabase());
	if (debug_checkpoint_abort == CheckpointAbort::DEBUG_ABORT_AFTER_FREE_LIST_WRITE) {
		throw FatalException("Checkpoint aborted after free list write because of PRAGMA checkpoint_abort flag");
	}

	// We need to fsync BEFORE we write the header to ensure that all the previous blocks are written as well
	handle->Sync();

	header_buffer.Clear();
	// if we are upgrading the database from version 64 -> version 65, we need to re-write the main header
	if (options.version_number.GetIndex() == 64 && options.storage_version.GetIndex() >= 4) {
		// rewrite the main header
		options.version_number = 65;
		MainHeader main_header = ConstructMainHeader(options.version_number.GetIndex());
		SerializeHeaderStructure<MainHeader>(main_header, header_buffer.buffer);
		// now write the header to the file
		ChecksumAndWrite(context, header_buffer, 0);
		header_buffer.Clear();
	}

	// set the header inside the buffer
	MemoryStream serializer(Allocator::Get(db));
	header.Write(serializer);
	if (block_compression) {
		// two fields after serialization_compatibility (block compression): the extent map position
		serializer.Write<idx_t>(extent_map_offset);
		serializer.Write<idx_t>(extent_map_entries);
	}
	memcpy(header_buffer.buffer, serializer.GetData(), serializer.GetPosition());
	// now write the header to the file, active_header determines whether we write to h1 or h2
	// note that if active_header is h1 we write to h2, and vice versa
	auto location = active_header == 1 ? Storage::FILE_HEADER_SIZE : Storage::FILE_HEADER_SIZE * 2;
	ChecksumAndWrite(context, header_buffer, location);
	// switch active header to the other header
	active_header = 1 - active_header;
	//! Ensure the header write ends up on disk
	handle->Sync();
	if (block_compression) {
		// committed: what this checkpoint released becomes reusable and a free tail is cut; then the extents at the end
		// of the file move into the free space before them (durable through an in-place header rewrite) and the tail
		// they leave is cut
		durable_header = header;
		RebuildFreeExtents();
		CompactAfterCommit(context, all_free_blocks);
	}
	// Release the free fully freed blocks to the filesystem.
	TrimFreeBlocks(fully_freed_blocks);
}

void SingleFileBlockManager::FileSync() {
	if (!block_compression) {
		handle->Sync();
		return;
	}
	// the WAL is about to reference blocks written since the last commit (optimistic writes): their extents must be
	// found at restart, so the extent map of the current blocks is written and the active header rewritten in place
	// to point at it (same iteration, so the WAL still matches the header)
	lock_guard<mutex> header_guard(header_lock);
	set<block_id_t> free_ids;
	idx_t block_count;
	{
		lock_guard<mutex> guard(single_file_block_lock);
		free_ids = free_list;
		block_count = NumericCast<idx_t>(max_block);
	}
	WriteExtentMap(QueryContext(), block_count, free_ids);
	handle->Sync();
	WriteActiveHeaderInPlace(QueryContext());
}

void SingleFileBlockManager::WriteActiveHeaderInPlace(QueryContext context) {
	header_buffer.Clear();
	MemoryStream serializer(Allocator::Get(db));
	durable_header.Write(serializer);
	serializer.Write<idx_t>(extent_map_offset);
	serializer.Write<idx_t>(extent_map_entries);
	memcpy(header_buffer.buffer, serializer.GetData(), serializer.GetPosition());
	auto location = active_header == 0 ? Storage::FILE_HEADER_SIZE : Storage::FILE_HEADER_SIZE * 2;
	ChecksumAndWrite(context, header_buffer, location);
	handle->Sync();
}

void SingleFileBlockManager::UnregisterBlock(block_id_t id) {
	// perform the actual unregistration
	BlockManager::UnregisterBlock(id);
	// check if it is part of the newly free list
	lock_guard<mutex> lock(single_file_block_lock);
	auto entry = free_blocks_in_use.find(id);
	if (entry != free_blocks_in_use.end()) {
		// it is! move it to the regular free list so the block can be re-used
		free_list.insert(id);
		free_blocks_in_use.erase(entry);
	}
}

void SingleFileBlockManager::TrimFreeBlockRange(block_id_t start, block_id_t end) {
	auto block_count = NumericCast<idx_t>(end + 1 - start);
	handle->Trim(BLOCK_START + (NumericCast<idx_t>(start) * GetBlockAllocSize()), block_count * GetBlockAllocSize());
}

void SingleFileBlockManager::TrimFreeBlocks(const set<block_id_t> &blocks) {
	if (!DBConfig::Get(db).options.trim_free_blocks || block_compression) {
		return;
	}
	lock_guard<mutex> lock(single_file_block_lock);
	for (auto itr = blocks.begin(); itr != blocks.end(); ++itr) {
		if (!free_list.count(*itr)) {
			continue;
		}
		block_id_t first = *itr;
		block_id_t last = first;
		// Find end of contiguous range.
		for (++itr; itr != blocks.end() && (*itr == last + 1) && free_list.count(*itr); ++itr) {
			last = *itr;
		}
		// We are now one too far.
		--itr;
		// Trim the range.
		TrimFreeBlockRange(first, last);
	}
}

} // namespace duckdb
