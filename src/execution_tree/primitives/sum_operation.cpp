// Copyright (c) 2018 Parsa Amini
// Copyright (c) 2018 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>
#include <phylanx/execution_tree/primitives/sum_operation.hpp>
#include <phylanx/ir/node_data.hpp>
#include <phylanx/util/matrix_iterators.hpp>

#include <hpx/include/lcos.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/util.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/iterator_facade.hpp>
#include <hpx/util/optional.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <blaze/Math.h>

///////////////////////////////////////////////////////////////////////////////
namespace phylanx { namespace execution_tree { namespace primitives
{
    ///////////////////////////////////////////////////////////////////////////
    primitive create_sum_operation(hpx::id_type const& locality,
        std::vector<primitive_argument_type>&& operands,
        std::string const& name, std::string const& codename)
    {
        static std::string type("sum_operation");
        return create_primitive_component(
            locality, type, std::move(operands), name, codename);
    }

    match_pattern_type const sum_operation::match_data =
    {
        hpx::util::make_tuple("sum_operation",
        std::vector<std::string>{"sum(_1)", "sum(_1, _2)", "sum(_1, _2, _3)"},
        &create_sum_operation, &create_primitive<sum_operation>)
    };

    ///////////////////////////////////////////////////////////////////////////
    sum_operation::sum_operation(
        std::vector<primitive_argument_type>&& operands,
        std::string const& name, std::string const& codename)
      : primitive_component_base(std::move(operands), name, codename)
    {}

    primitive_argument_type sum_operation::sum0d(arg_type&& arg,
        hpx::util::optional<std::int64_t> axis, bool keep_dims) const
    {
        if (axis && axis.value() != 0 && axis.value() != -1)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "sum_operation::sum0d",
                execution_tree::generate_error_message(
                    "the sum_operation primitive requires operand axis to be "
                    "either 0 or -1 for scalar values.",
                    name_, codename_));
        }

        return primitive_argument_type{arg.scalar()};
    }

    primitive_argument_type sum_operation::sum1d(arg_type&& arg,
        hpx::util::optional<std::int64_t> axis, bool keep_dims) const
    {
        if (axis && axis.value() != 0 && axis.value() != -1)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "sum_operation::sum1d",
                execution_tree::generate_error_message(
                    "the sum_operation primitive requires operand axis to be "
                    "either 0 or -1 for vectors.",
                    name_, codename_));
        }

        auto v = arg.vector();
        double result = std::accumulate(v.begin(), v.end(), 0.);

        if (keep_dims)
        {
            return primitive_argument_type{
                blaze::DynamicVector<val_type>{result}};
        }
        return primitive_argument_type{result};
    }

    primitive_argument_type sum_operation::sum2d(arg_type&& arg,
        hpx::util::optional<std::int64_t> axis, bool keep_dims) const
    {
        if (axis)
        {
            switch (axis.value())
            {
            case -2: HPX_FALLTHROUGH;
            case 0:
                return sum2d_axis0(std::move(arg));
            case -1: HPX_FALLTHROUGH;
            case 1:
                return sum2d_axis1(std::move(arg));
            default:
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "sum_operation::sum1d",
                    execution_tree::generate_error_message(
                        "the sum_operation primitive requires operand axis to be "
                        "between -2 and 1 for matrices.",
                        name_, codename_));
            }
        }
        return sum2d_flat(std::move(arg), keep_dims);
    }

    primitive_argument_type sum_operation::sum2d_flat(
        arg_type&& arg, bool keep_dims) const
    {
        auto m = arg.matrix();
        double result = 0.;
        for (std::size_t i = 0; i < m.columns(); ++i)
        {
            auto col = blaze::column(m, i);
            result += std::accumulate(col.begin(), col.end(), 0.);
        }

        if (keep_dims)
        {
            return primitive_argument_type{
                blaze::DynamicMatrix<val_type>{{result}}};
        }
        return primitive_argument_type{result};
    }

    primitive_argument_type sum_operation::sum2d_axis0(arg_type&& arg) const
    {
        auto m = arg.matrix();
        blaze::DynamicVector<double> result(m.columns());
        for (std::size_t i = 0; i < m.columns(); ++i)
        {
            auto col = blaze::column(m, i);
            result[i] = std::accumulate(col.begin(), col.end(), 0.);
        }

        return primitive_argument_type{result};
    }

    primitive_argument_type sum_operation::sum2d_axis1(arg_type&& arg) const
    {
        auto m = arg.matrix();
        blaze::DynamicVector<double> result(m.rows());
        for (std::size_t i = 0; i < m.rows(); ++i)
        {
            auto col = blaze::row(m, i);
            result[i] = std::accumulate(col.begin(), col.end(), 0.);
        }

        return primitive_argument_type{result};
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<primitive_argument_type> sum_operation::eval(
        std::vector<primitive_argument_type> const& operands,
        std::vector<primitive_argument_type> const& args) const
    {
        if (operands.empty() || operands.size() > 3)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "sum_operation::eval",
                execution_tree::generate_error_message(
                    "the sum_operation primitive requires exactly one, two, or "
                    "three operands",
                    name_, codename_));
        }

        for (auto const& i : operands)
        {
            if (!valid(i))
            {
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "sum_operation::eval",
                    execution_tree::generate_error_message(
                        "the sum_operation primitive requires that the "
                        "arguments given by the operands array are "
                        "valid",
                        name_, codename_));
            }
        }

        auto this_ = this->shared_from_this();
        return hpx::dataflow(hpx::launch::sync,
            hpx::util::unwrapping(
                [this_](std::vector<primitive_argument_type>&& args)
                    -> primitive_argument_type
                {
                    // Extract axis and keep_dims
                    // Presence of axis changes behavior for >2d cases
                    hpx::util::optional<std::int64_t> axis;
                    bool keep_dims = false;

                    // axis is argument #2
                    if (args.size() > 1)
                    {
                        if (valid(args[1]))
                            axis = execution_tree::extract_integer_value(
                                args[1], this_->name_, this_->codename_);

                        // keep_dims is argument #3
                        if (args.size() == 3)
                        {
                            keep_dims = execution_tree::extract_boolean_value(
                                args[2], this_->name_, this_->codename_);
                        }
                    }

                    // Extract the matrix
                    arg_type a = execution_tree::extract_numeric_value(
                        args[0], this_->name_, this_->codename_);

                    std::size_t a_dims = a.num_dimensions();
                    switch (a_dims)
                    {
                    case 0:
                        return this_->sum0d(std::move(a), axis, keep_dims);
                    case 1:
                        return this_->sum1d(std::move(a), axis, keep_dims);
                    case 2:
                        return this_->sum2d(std::move(a), axis, keep_dims);
                    default:
                        HPX_THROW_EXCEPTION(hpx::bad_parameter,
                            "sum_operation::eval",
                            execution_tree::generate_error_message(
                                "operand a has an invalid "
                                "number of dimensions",
                                this_->name_, this_->codename_));
                    }
                }),
            detail::map_operands(
                operands, functional::value_operand{}, args, name_, codename_));
    }

    hpx::future<primitive_argument_type> sum_operation::eval(
        std::vector<primitive_argument_type> const& args) const
    {
        if (operands_.empty())
        {
            return eval(args, noargs);
        }
        return eval(operands_, args);
    }
}}}
