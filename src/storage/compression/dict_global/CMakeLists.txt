add_library_unity(duckdb_storage_compression_dict_global OBJECT
                  column_dictionary.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_storage_compression_dict_global>
    PARENT_SCOPE)
