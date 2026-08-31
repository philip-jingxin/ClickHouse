#include <Columns/ColumnArray.h>
#include <Columns/ColumnSparse.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/IColumn.h>
#include <Columns/IColumn_fwd.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/IFunction.h>
#include <Interpreters/Context_fwd.h>
#include <base/BFloat16.h>
#include <base/types.h>
#include <Common/FunctionDocumentation.h>
#include <Common/PODArray.h>
#include <Common/PODArray_fwd.h>
#include <Common/RandomSparseTransform.h>
#include <Common/assert_cast.h>

#include <memory>
#include <vector>

#include <city.h>
namespace DB
{
namespace ErrorCodes
{
extern const int ILLEGAL_TYPE_OF_ARGUMENT;
extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
extern const int ILLEGAL_COLUMN;
extern const int ARGUMENT_OUT_OF_BOUND;
extern const int BAD_ARGUMENTS;
}


namespace
{
using namespace RandomSparseTransform;

class FunctionRandomSparseTransform : public IFunction
{
public:
    static constexpr auto name = "RandomSparseTransform";
    static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionRandomSparseTransform>(); }
    String getName() const override { return name; }
    bool isVariadic() const override { return true; }
    size_t getNumberOfArguments() const override { return 0; }
    bool useDefaultImplementationForConstants() const override { return false; }
    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo &) const override { return false; }

    bool useDefaultImplementationForSparseColumns() const override { return false; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override
    {
        if (arguments.empty() || arguments.size() > 3)
            throw Exception(
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                "Function {} requires 1 to 3 arguments: vector [, seed] [, output_dims]",
                getName());

        const auto * array_type = checkAndGetDataType<DataTypeArray>(arguments[0].type.get());

        if (!array_type)
            throw Exception(
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "First argument of function {} must be an Array of floats, got {}",
                getName(),
                arguments[0].type->getName());

        WhichDataType which_nested(array_type->getNestedType());
        if (!which_nested.isFloat32() && !which_nested.isFloat64() && !which_nested.isBFloat16())
            throw Exception(
                ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "First argument of function {} must be Array(BFloat16|Float32|Float64), got Array({})",
                getName(),
                array_type->getNestedType()->getName());

        for (size_t i = 1; i < arguments.size(); ++i)
        {
            WhichDataType which(arguments[i].type);
            if (!which.isNativeUInt() && !which.isNativeInt())
                throw Exception(
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                    "The {} argument of function {} must be an integer, got {}",
                    i == 1 ? "'seed'" : "'output_dims'",
                    getName(),
                    arguments[i].type->getName());
        }

        return std::make_shared<DataTypeArray>(array_type->getNestedType());
    }

    ColumnPtr
    executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & /* result_type */, size_t input_rows_count) const override
    {
        UInt64 seed = 0;

        if (arguments.size() >= 2)
        {
            if (!isColumnConst(*arguments[1].column))
                throw Exception(ErrorCodes::ILLEGAL_COLUMN, "The 'seed' argument of function {} must be a constant", getName());
            seed = arguments[1].column->getUInt(0);
        }

        ColumnPtr arg0 = arguments[0].column->convertToFullColumnIfConst();

        WhichDataType which_type(checkAndGetDataType<DataTypeArray>(arguments[0].type.get())->getNestedType());

        if (const ColumnSparse * col_sparse = checkAndGetColumn<ColumnSparse>(arg0.get()))
        {
            ColumnPtr col_ptr = col_sparse->getValuesPtr();
            const ColumnArray * col_array = checkAndGetColumn<ColumnArray>(col_ptr.get());

            const ColumnArray::Offsets & offsets = col_array->getOffsets();
            const IColumn & col = col_array->getData();

            ColumnPtr result_ptr;
            if (which_type.isFloat64())
                result_ptr = run<Float64, Float64, Float64>(col, offsets, col_ptr->size(), seed);
            else if (which_type.isFloat32())
                result_ptr = run<Float32, Float32, Float32>(col, offsets, col_ptr->size(), seed);
            else
                result_ptr = run<BFloat16, Float32, BFloat16>(col, offsets, col_ptr->size(), seed);

            return ColumnSparse::create(result_ptr, col_sparse->getOffsetsPtr(), input_rows_count);
        }

        if (const ColumnArray * col_array = checkAndGetColumn<ColumnArray>(arg0.get()))
        {
            const IColumn & col = col_array->getData();
            const ColumnArray::Offsets & offsets = col_array->getOffsets();
            if (which_type.isFloat64())
                return run<Float64, Float64, Float64>(col, offsets, input_rows_count, seed);
            if (which_type.isFloat32())
                return run<Float32, Float32, Float32>(col, offsets, input_rows_count, seed);
            return run<BFloat16, Float32, BFloat16>(col, offsets, input_rows_count, seed);
        }

        throw Exception(
            ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
            "First argument of function {} must be an Array of floats, got {}",
            getName(),
            arguments[0].type->getName());
    }

private:
    template <typename In, typename Compute, typename Out>
    static ColumnPtr run(const IColumn & col, const ColumnArray::Offsets & offsets, size_t rows, UInt64 seed)
    {
        const auto & input = assert_cast<const ColumnVector<In> &>(col).getData();


        auto result_column = ColumnVector<Out>::create();
        auto result_offsets_column = ColumnArray::ColumnOffsets::create();

        auto & result = result_column->getData();
        auto & result_offsets = result_offsets_column->getData();
        result_offsets.resize(rows);

        size_t written = 0;
        size_t start = 0;

        PaddedPODArray<Compute> buffer;

        for (size_t row = 0; row < rows; ++row)
        {
            const size_t length = offsets[row] - start;
            const size_t working_dim = length;
            const size_t k = working_dim;

            const In * in = input.data() + start;
            result.resize(written + k);
            Out * out = result.data() + written;
            buffer.resize(working_dim);
            for (size_t i = 0; i < length; ++i)
                buffer[i] = static_cast<Compute>(in[i]);

            UInt64 state = CityHash_v1_0_2::CityHash64WithSeed(reinterpret_cast<const char *>(&row), sizeof(row), seed);
            sparseScalar(buffer.data(), state, working_dim);

            for (size_t i = 0; i < k; ++i)
                out[i] = static_cast<Out>(buffer[i]);

            written += k;


            result_offsets[row] = written;
            start = offsets[row];
        }


        return ColumnArray::create(std::move(result_column), std::move(result_offsets_column));
    }
};

}

REGISTER_FUNCTION(RandomSparseTransform)
{
    FunctionDocumentation::Description description = "some description";
    FunctionDocumentation::Syntax syntax = "some syntax";
    FunctionDocumentation::Arguments arguments = {{"arg1", " arg1", {"UInt*"}}};
    FunctionDocumentation::ReturnedValue returned_value = {};
    FunctionDocumentation::Examples examples = {};
    FunctionDocumentation::IntroducedIn introduced_in = {};
    FunctionDocumentation::Category category = FunctionDocumentation::Category::Array;
    FunctionDocumentation documentation = {description, syntax, arguments, {}, returned_value, examples, introduced_in, category};

    factory.registerFunction<FunctionRandomSparseTransform>(documentation);
}


}
