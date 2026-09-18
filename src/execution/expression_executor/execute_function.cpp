#include "duckdb/common/type_visitor.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/function/scalar/regexp.hpp"
#include "duckdb/storage/object_cache.hpp"

namespace duckdb {

namespace {
class RegexpReplaceDictionaryMemo : public ObjectCacheEntry {
public:
	RegexpReplaceDictionaryMemo(buffer_ptr<VectorChildBuffer> dictionary_p, idx_t bytes_p)
	    : dictionary(std::move(dictionary_p)), bytes(bytes_p) {
	}
	static string ObjectType() {
		return "regexp_replace_dictionary_memo_v1";
	}
	string GetObjectType() override {
		return ObjectType();
	}
	optional_idx GetEstimatedCacheMemory() const override {
		return bytes;
	}
	buffer_ptr<VectorChildBuffer> dictionary;
	const idx_t bytes;
};

optional_idx DictionaryMemoWeight(VectorChildBuffer &dictionary, idx_t count, const string &key) {
	static constexpr idx_t MAX_ENTRY_BYTES = 8ULL * 1024 * 1024;
	idx_t bytes = 0;
	auto add = [&](idx_t part) {
		if (part > MAX_ENTRY_BYTES - bytes) {
			return false;
		}
		bytes += part;
		return true;
	};
	// Include both LRU/map key copies and conservative cache-node overhead. Prepay
	// the only lazy shared data: the existing mutex-protected cached_hashes vector.
	if (count > MAX_ENTRY_BYTES / (sizeof(string_t) + sizeof(hash_t)) ||
	    !add(sizeof(RegexpReplaceDictionaryMemo) + sizeof(VectorChildBuffer) + sizeof(VectorStringBuffer) +
	         2 * sizeof(VectorBuffer) + 1024) ||
	    !add(key.size()) || !add(key.size()) ||
	    !add(count * (sizeof(string_t) + sizeof(hash_t))) ||
	    !add(ValidityMask::ValidityMaskSize(count)) || !add(ValidityMask::ValidityMaskSize(count)) ||
	    !add(StringVector::GetStringBuffer(dictionary.data).GetStringAllocator().AllocationSize())) {
		return optional_idx();
	}
	return bytes;
}
} // namespace


ExecuteFunctionState::ExecuteFunctionState(const Expression &expr, ExpressionExecutorState &root)
    : ExpressionState(expr, root) {
	// Check if the expression is eligible for dictionary optimization
	if (!expr.IsConsistent() || expr.IsVolatile() || expr.CanThrow()) {
		return; // Needs to be consistent, non-volatile, and non-throwing
	}

	if (expr.return_type.InternalType() == PhysicalType::STRUCT) {
		return; // FIXME: get this working for STRUCT
	}

	// Set input_col_idx accordingly, marking the expression as eligible for dictionary optimization
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_FUNCTION: {
		auto &bound_function = expr.Cast<BoundFunctionExpression>();
		auto &children = bound_function.children;
		for (idx_t child_idx = 0; child_idx < children.size(); child_idx++) {
			auto &child = *children[child_idx];
			if (child.IsFoldable()) {
				continue; // Constant
			}
			if (input_col_idx.IsValid()) {
				input_col_idx.SetInvalid(); // Found more than 1 non-constant
				break;
			}
			if (child.return_type.InternalType() == PhysicalType::STRUCT) {
				break; // FIXME
			}
			input_col_idx = child_idx;
		}
		if (input_col_idx.IsValid() && input_col_idx.GetIndex() == 0) {
			TryGetRegexpReplaceDictionaryMemoKey(bound_function, dictionary_memo_key);
		}
		break;
	}
	default:
		break;
	}
}

ExecuteFunctionState::~ExecuteFunctionState() {
}

bool ExecuteFunctionState::TryExecuteDictionaryExpression(const BoundFunctionExpression &expr, DataChunk &args,
                                                          ExpressionState &state, Vector &result) {
	static constexpr idx_t MAX_DICTIONARY_SIZE_THRESHOLD = 20000;
	static constexpr double CHUNK_FILL_RATIO_THRESHOLD = 0.5;

	if (!input_col_idx.IsValid()) {
		return false; // This expression is not eligible for dictionary optimization
	}

	// Figure out if we can do the optimization
	const auto &unary_input = args.data[input_col_idx.GetIndex()];
	if (unary_input.GetVectorType() != VectorType::DICTIONARY_VECTOR) {
		return false; // Not a dictionary
	}

	const auto input_dictionary_size_opt = DictionaryVector::DictionarySize(unary_input);
	const auto &input_dictionary_id = DictionaryVector::DictionaryId(unary_input);
	if (!input_dictionary_size_opt.IsValid() || input_dictionary_id.empty()) {
		return false; // Not a dictionary that comes from storage
	}

	const auto input_dictionary_size = input_dictionary_size_opt.GetIndex();
	if (input_dictionary_size >= MAX_DICTIONARY_SIZE_THRESHOLD) {
		return false; // Dictionary is too large, bail
	}

	if (!output_dictionary || current_input_dictionary_id != input_dictionary_id) {
		// We haven't seen this dictionary before
		const auto chunk_fill_ratio = static_cast<double>(args.size()) / STANDARD_VECTOR_SIZE;
		if (input_dictionary_size > STANDARD_VECTOR_SIZE && chunk_fill_ratio <= CHUNK_FILL_RATIO_THRESHOLD) {
			// If the dictionary size is <= STANDARD_VECTOR_SIZE, we always do the optimization
			// If it's greater, we only do the optimization if the chunk is more than 50% full
			// This protects the optimization against selective filters
			return false;
		}

		string memo_key;
		if (!dictionary_memo_key.empty() && state.HasContext()) {
			memo_key = dictionary_memo_key + to_string(input_dictionary_id.size()) + ":" + input_dictionary_id;
			auto cached = ObjectCache::GetObjectCache(state.GetContext()).Get<RegexpReplaceDictionaryMemo>(memo_key);
			if (cached) {
				output_dictionary = cached->dictionary;
				current_input_dictionary_id = input_dictionary_id;
				result.Dictionary(output_dictionary, DictionaryVector::SelVector(unary_input));
				return true;
			}
		}

		// We can do dictionary optimization! Re-initialize
		output_dictionary = DictionaryVector::CreateReusableDictionary(result.GetType(), input_dictionary_size);
		current_input_dictionary_id = input_dictionary_id;

		// Set up the input chunk
		DataChunk input_chunk;
		input_chunk.InitializeEmpty(args.GetTypes());
		for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
			if (col_idx != input_col_idx.GetIndex()) {
				input_chunk.data[col_idx].Reference(args.data[col_idx]);
			}
		}

		// Loop over the dictionary, executing at most STANDARD_VECTOR_SIZE at a time
		for (idx_t offset = 0; offset < input_dictionary_size; offset += STANDARD_VECTOR_SIZE) {
			const auto count = MinValue<idx_t>(input_dictionary_size - offset, STANDARD_VECTOR_SIZE);

			// Offset the input dictionary
			Vector offset_input(DictionaryVector::Child(unary_input), offset, offset + count);
			input_chunk.data[input_col_idx.GetIndex()].Reference(offset_input);
			input_chunk.SetCardinality(count);

			// Execute, storing the result in an intermediate vector, and copying it to the output dictionary
			Vector output_intermediate(result.GetType());
			expr.function.GetFunctionCallback()(input_chunk, state, output_intermediate);
			VectorOperations::Copy(output_intermediate, output_dictionary->data, count, 0, offset);
		}
		if (!memo_key.empty()) {
			const auto bytes = DictionaryMemoWeight(*output_dictionary, input_dictionary_size, memo_key);
			if (bytes.IsValid()) {
				auto cached = ObjectCache::GetObjectCache(state.GetContext()).GetOrCreate<RegexpReplaceDictionaryMemo>(
				    memo_key, output_dictionary, bytes.GetIndex());
				if (cached) {
					output_dictionary = cached->dictionary;
				}
			}
		}
	}

	// Result references the dictionary
	result.Dictionary(output_dictionary, DictionaryVector::SelVector(unary_input));

	return true;
}

void ExecuteFunctionState::ResetDictionaryStates() {
	// Clear the cached dictionary information
	current_input_dictionary_id.clear();
	output_dictionary.reset();

	for (const auto &child_state : child_states) {
		child_state->ResetDictionaryStates();
	}
}

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundFunctionExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_uniq<ExecuteFunctionState>(expr, root);
	for (auto &child : expr.children) {
		result->AddChild(*child);
	}

	result->Finalize();
	if (expr.function.HasInitStateCallback()) {
		result->local_state = expr.function.GetInitStateCallback()(*result, expr, expr.bind_info.get());
	}
	return std::move(result);
}

static void VerifyNullHandling(const BoundFunctionExpression &expr, DataChunk &args, Vector &result) {
#ifdef DEBUG
	if (args.data.empty() || expr.function.GetNullHandling() != FunctionNullHandling::DEFAULT_NULL_HANDLING) {
		return;
	}

	// Combine all the argument validity masks into a flat validity mask
	idx_t count = args.size();
	ValidityMask combined_mask(count);
	for (auto &arg : args.data) {
		UnifiedVectorFormat arg_data;
		arg.ToUnifiedFormat(count, arg_data);

		for (idx_t i = 0; i < count; i++) {
			auto idx = arg_data.sel->get_index(i);
			if (!arg_data.validity.RowIsValid(idx)) {
				combined_mask.SetInvalid(i);
			}
		}
	}

	// Default is that if any of the arguments are NULL, the result is also NULL
	UnifiedVectorFormat result_data;
	result.ToUnifiedFormat(count, result_data);
	for (idx_t i = 0; i < count; i++) {
		if (!combined_mask.RowIsValid(i)) {
			auto idx = result_data.sel->get_index(i);
			D_ASSERT(!result_data.validity.RowIsValid(idx));
		}
	}
#endif
}

void ExpressionExecutor::Execute(const BoundFunctionExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	state->intermediate_chunk.Reset();
	auto &arguments = state->intermediate_chunk;
	if (!state->types.empty()) {
		for (idx_t i = 0; i < expr.children.size(); i++) {
			D_ASSERT(state->types[i] == expr.children[i]->return_type);
			Execute(*expr.children[i], state->child_states[i].get(), sel, count, arguments.data[i]);
#ifdef DEBUG
			if (expr.children[i]->return_type.id() == LogicalTypeId::VARCHAR) {
				arguments.data[i].UTFVerify(count);
			}
#endif
		}
	}
	arguments.SetCardinality(count);
	arguments.Verify();

	D_ASSERT(expr.function.HasFunctionCallback());
	auto &execute_function_state = state->Cast<ExecuteFunctionState>();
	if (!execute_function_state.TryExecuteDictionaryExpression(expr, arguments, *state, result)) {
		expr.function.GetFunctionCallback()(arguments, *state, result);
	}

	VerifyNullHandling(expr, arguments, result);
	D_ASSERT(result.GetType() == expr.return_type);
}

} // namespace duckdb
