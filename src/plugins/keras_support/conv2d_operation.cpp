// Copyright (c) 2019 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>
#include <phylanx/execution_tree/primitives/node_data_helpers.hpp>
#include <phylanx/ir/node_data.hpp>
#include <phylanx/plugins/keras_support/conv2d_operation.hpp>

#include <hpx/datastructures/optional.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/util.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <blaze/Math.h>
#include <blaze_tensor/Math.h>

///////////////////////////////////////////////////////////////////////////////
namespace phylanx { namespace execution_tree { namespace primitives {
    ///////////////////////////////////////////////////////////////////////////
    match_pattern_type const conv2d_operation::match_data = {
        hpx::util::make_tuple("conv2d", std::vector<std::string>{R"(
            conv2d(_1, _2_kernel,
            __arg(_3_padding, "valid"),
            __arg(_4_strides, 1),
            __arg(_5_dilation_rate, 1))
        )"},
            &create_conv2d_operation, &create_primitive<conv2d_operation>,
            R"(x, kernel, padding, strides, dilation_rate
        Args:

            x (array) : a 3d array consiting of batch, in_length and
                in_channels dimensions.
            kernel (array) : a 3d array consisting of filter_length,
                in_channels and out_channels dimension. Note that the
                in_channels should be the same in kernel and original array.
            padding (optional, string) : padding mode, `valid` by default. It
                can be either `valid`, `same` or `causal`. `vaild` means no
                padding. `same` results the output with the same shape as
                original array in case of unit strides. `causal` zero pads the
                array in a way that no output element depend on the input
                elements of its future.
            strides (optional, integer) : the step to apply convolution over
                array. It sets to 1 by default.
            dilation_rate (optional, integer) : indicates the dilation rate,
                the rate to sample the array in each step of convolution, 1
                by default.

        Returns:

        2D convolution (or 2D mathematical cross-correlation))")};

    ///////////////////////////////////////////////////////////////////////////
    conv2d_operation::conv2d_operation(primitive_arguments_type&& operands,
        std::string const& name, std::string const& codename)
      : primitive_component_base(std::move(operands), name, codename)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    primitive_argument_type conv2d_operation::calculate_conv2d(
        ir::node_data<T>&& arg, ir::node_data<T>&& kernel) const
    {
        if (kernel.num_dimensions() != 2)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "conv2d_operation::calculate_conv2d",
                generate_error_message(
                    "the conv2d_operation primitive requires the "
                    "kernel to be 2D"));
        }
        auto arg_m = arg.matrix();
        auto kernel_m = kernel.matrix();
        blaze::DynamicMatrix<T> result(arg_m.rows() - kernel_m.rows() + 1,
            arg_m.columns() - kernel_m.columns() + 1, 0);

        for (std::size_t i = 0; i < arg_m.rows() - kernel_m.rows() + 1; ++i)
        {
            for (std::size_t j = 0;
                 j < arg_m.columns() - kernel_m.columns() + 1;
                 ++j)
            {
                auto tmp = blaze::submatrix(
                    arg_m, i, j, kernel_m.rows(), kernel_m.columns());
                result(i, j) = blaze::sum(tmp % kernel_m);
            }
        }
        return primitive_argument_type{result};
    }

    template <typename T>
    primitive_argument_type conv2d_operation::calculate_conv2d(
        ir::node_data<T>&& arg, ir::node_data<T>&& kernel,
        ir::range&& shape) const
    {
        if (kernel.num_dimensions() != 1)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "conv2d_operation::calculate_conv2d",
                generate_error_message(
                    "the kernel is not in the right format"));
        }
        if (shape.size() != 4)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "conv2d_operation::calculate_conv2d",
                generate_error_message(
                    "the shape should be in this format [filter_shape[0], "
                    "filter_shape[1], num_input_channels,num_filters]"));
        }

        auto it = shape.begin();
        auto filter_shape_0_r = extract_integer_value_strict(*it);
        auto filter_shape_1_r = extract_integer_value_strict(*++it);
        auto num_channels_r = extract_integer_value_strict(*++it);
        auto num_filters_r = extract_integer_value_strict(*++it);
        auto filter_shape_0 = filter_shape_0_r.scalar();
        auto filter_shape_1 = filter_shape_1_r.scalar();
        auto num_channels = num_channels_r.scalar();
        auto num_filters = num_filters_r.scalar();

        if (num_channels != 1)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "conv2d_operation::calculate_conv2d",
                generate_error_message("number of channels should be one"));
        }

        auto arg_m = arg.matrix();
        auto kernel_flattened = kernel.vector();
        //shape:[filter_shape[0], filter_shape[1], num_input_channels,num_filters]

        blaze::DynamicTensor<T> result(num_filters,
            arg_m.rows() - filter_shape_0 + 1,
            arg_m.columns() - filter_shape_1 + 1, 0);

        for (std::size_t k = 0; k < num_filters; ++k)
        {
            blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded> slice(
                &kernel_flattened[k * filter_shape_0 * filter_shape_1],
                filter_shape_0, filter_shape_1);
            for (std::size_t i = 0; i < result.rows(); ++i)
            {
                for (std::size_t j = 0; j < result.columns(); ++j)
                {
                    auto tmp = blaze::submatrix(
                        arg_m, i, j, slice.rows(), slice.columns());

                    result(k, i, j) = blaze::sum(tmp % tmp);
                }
            }
        }
        //output_shape=[num_filters,conv_size_row,conv_size_col]
        return primitive_argument_type{std::move(result)};
    }

    template <typename T>
    primitive_argument_type conv2d_operation::calculate_conv3d(
        ir::node_data<T>&& arg, ir::node_data<T>&& kernel) const
    {
        if (kernel.num_dimensions() != 2)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "conv2d_operation::calculate_conv3d",
                generate_error_message(
                    "the conv2d_operation primitive requires the "
                    "kernel to be 2D"));
        }
        //third dimension of the input is num_channels
        auto arg_m = arg.matrix();
        auto kernel_m = kernel.matrix();
        blaze::DynamicMatrix<T> result(arg_m.rows() - kernel_m.rows() + 1,
            arg_m.columns() - kernel_m.columns() + 1, 0);

        for (std::size_t i = 0; i < arg_m.rows() - kernel_m.rows() + 1; ++i)
        {
            for (std::size_t j = 0;
                 j < arg_m.columns() - kernel_m.columns() + 1;
                 ++j)
            {
                auto tmp = blaze::submatrix(
                    arg_m, i, j, kernel_m.rows(), kernel_m.columns());
                result(i, j) = blaze::sum(tmp % kernel_m);
            }
        }
        return primitive_argument_type{result};
    }

    template <typename T>
    primitive_argument_type conv2d_operation::calculate_conv3d(
        ir::node_data<T>&& arg, ir::node_data<T>&& kernel,
        ir::range&& shape) const
    {
        if (kernel.num_dimensions() != 1)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "conv2d_operation::calculate_conv3d",
                generate_error_message(
                    "the kernel is not in the right format"));
        }

        if (shape.size() != 4)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "conv2d_operation::calculate_conv2d",
                generate_error_message(
                    "the shape shouyld be in this format [filter_shape[0], "
                    "filter_shape[1], num_input_channels,num_filters]"));
        }

        auto it = shape.begin();
        auto filter_shape_0_r = extract_integer_value_strict(*it);
        auto filter_shape_1_r = extract_integer_value_strict(*++it);
        auto num_channels_r = extract_integer_value_strict(*++it);
        auto num_filters_r = extract_integer_value_strict(*++it);
        auto filter_shape_0 = filter_shape_0_r.scalar();
        auto filter_shape_1 = filter_shape_1_r.scalar();
        auto num_channels = num_channels_r.scalar();
        auto num_filters = num_filters_r.scalar();

        auto arg_t = arg.tensor();
        //assuming num_channels is the same for input and kernel
        auto kernel_flattened = kernel.vector();

        //shape:[filter_shape[0], filter_shape[1], num_input_channels,num_filters]
        std::int64_t conv_size_row = arg_t.pages() - filter_shape_0 + 1;
        std::int64_t conv_size_col = arg_t.rows() - filter_shape_1 + 1;
        std::int64_t conv_size = conv_size_row * conv_size_col;

        blaze::DynamicVector<T> result(num_filters * conv_size * num_channels);
        for (std::size_t p = 0; p < num_filters; ++p)
        {
            for (std::size_t k = 0; k < num_channels; ++k)
            {
                auto start_index = k * conv_size;
                auto end_index = (k + 1) * conv_size;
                auto input_data = blaze::columnslice(arg_t, k);

                blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded> slice(
                    &kernel_flattened[p * num_channels +
                        k * filter_shape_0 * filter_shape_1],
                    filter_shape_0, filter_shape_1);

                for (std::size_t i = 0; i < conv_size_row; ++i)
                {
                    for (std::size_t j = 0; j < conv_size_col; ++j)
                    {
                        auto tmp = blaze::submatrix(
                            input_data, i, j, slice.rows(), slice.columns());

                        result[k * conv_size * num_channels +
                            i * conv_size_row + j] = blaze::sum(tmp % tmp);
                    }
                }
            }
        }

        ir::range output_shape(primitive_arguments_type{
            primitive_argument_type{std::move(num_filters)},
            primitive_argument_type{std::move(num_channels)},
            primitive_argument_type{std::move(conv_size_row)},
            primitive_argument_type{std::move(conv_size_col)}});
        //output_shape=[num_filters,conv_size_row,conv_size_col]
        return primitive_argument_type{
            primitive_arguments_type{primitive_argument_type{std::move(result)},
                primitive_argument_type{std::move(output_shape)}}};
    }

    template <typename T>
    primitive_argument_type conv2d_operation::calculate_conv(
        ir::node_data<T>&& arg, ir::node_data<T>&& kernel) const
    {
        if (arg.num_dimensions() == 2)
        {
            return calculate_conv2d(std::move(arg), std::move(kernel));
        }
        else if (arg.num_dimensions() == 3)
        {
            return calculate_conv3d(std::move(arg), std::move(kernel));
        }
        else
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "conv2d_operation::conv",
                generate_error_message(
                    "the conv2d_operation primitive requires "
                    "between 2 and 5 operands"));
    }

    template <typename T>
    primitive_argument_type conv2d_operation::calculate_conv(
        ir::node_data<T>&& arg, ir::node_data<T>&& kernel,
        ir::range&& shape) const
    {
        if (arg.num_dimensions() == 2)
        {
            return calculate_conv2d(
                std::move(arg), std::move(kernel), std::move(shape));
        }
        else if (arg.num_dimensions() == 3)
        {
            return calculate_conv3d(
                std::move(arg), std::move(kernel), std::move(shape));
        }
        else
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "conv2d_operation::conv",
                generate_error_message(
                    "the conv2d_operation primitive requires "
                    "between 2 and 5 operands"));
    }
    ///////////////////////////////////////////////////////////////////////////
    hpx::future<primitive_argument_type> conv2d_operation::eval(
        primitive_arguments_type const& operands,
        primitive_arguments_type const& args, eval_context ctx) const
    {
        if (operands.size() < 2 || operands.size() > 5)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "conv2d_operation::eval",
                generate_error_message(
                    "the conv2d_operation primitive requires "
                    "between 2 and 5 operands"));
        }

        for (auto const& i : operands)
        {
            if (!valid(i))
            {
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "conv2d_operation::eval",
                    generate_error_message(
                        "the conv2d_operation primitive requires that the "
                        "arguments given by the operands array are valid"));
            }
        }

        auto this_ = this->shared_from_this();
        return hpx::dataflow(hpx::launch::sync,
            hpx::util::unwrapping([this_ = std::move(this_)](
                                      primitive_arguments_type&& args)
                                      -> primitive_argument_type {
                std::size_t ndim_data = extract_numeric_value_dimension(
                    args[0], this_->name_, this_->codename_);

                if (is_list_operand_strict(args[1]))
                {
                    auto r = extract_list_value(
                        args[1], this_->name_, this_->codename_);
                    auto it = r.begin();
                    auto kernel = *it;
                    auto shape = *++it;
                    switch (extract_common_type(args[0]))
                    {
                    case node_data_type_bool:
                        return this_->calculate_conv(
                            extract_boolean_value(std::move(args[0]),
                                this_->name_, this_->codename_),
                            extract_boolean_value(std::move(kernel),
                                this_->name_, this_->codename_),
                            extract_list_value(std::move(shape), this_->name_,
                                this_->codename_));

                    case node_data_type_int64:
                        return this_->calculate_conv(
                            extract_integer_value(std::move(args[0]),
                                this_->name_, this_->codename_),
                            extract_integer_value(std::move(kernel),
                                this_->name_, this_->codename_),
                            extract_list_value(std::move(shape), this_->name_,
                                this_->codename_));
                    case node_data_type_unknown:
                        HPX_FALLTHROUGH;
                    case node_data_type_double:
                        return this_->calculate_conv(
                            extract_numeric_value(std::move(args[0]),
                                this_->name_, this_->codename_),
                            extract_numeric_value(std::move(kernel),
                                this_->name_, this_->codename_),
                            extract_list_value(std::move(shape), this_->name_,
                                this_->codename_));
                    default:
                        HPX_THROW_EXCEPTION(hpx::bad_parameter,
                            "conv2d_operation::eval",
                            this_->generate_error_message(
                                "type not supported"));
                    }
                }
                switch (extract_common_type(args[0]))
                {
                case node_data_type_bool:
                    return this_->calculate_conv(
                        extract_boolean_value(
                            std::move(args[0]), this_->name_, this_->codename_),
                        extract_boolean_value(std::move(args[1]), this_->name_,
                            this_->codename_));

                case node_data_type_int64:
                    return this_->calculate_conv(
                        extract_integer_value(
                            std::move(args[0]), this_->name_, this_->codename_),
                        extract_integer_value(std::move(args[1]), this_->name_,
                            this_->codename_));

                case node_data_type_unknown:
                    HPX_FALLTHROUGH;
                case node_data_type_double:
                    return this_->calculate_conv(
                        extract_numeric_value(
                            std::move(args[0]), this_->name_, this_->codename_),
                        extract_numeric_value(std::move(args[1]), this_->name_,
                            this_->codename_));
                default:
                    HPX_THROW_EXCEPTION(hpx::bad_parameter,
                        "conv2d_operation::eval",
                        this_->generate_error_message("type not supported"));
                }
            }),
            detail::map_operands(operands, functional::value_operand{}, args,
                name_, codename_, std::move(ctx)));
    }
}}}
