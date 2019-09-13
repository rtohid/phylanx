// Copyright (c) 2019 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PHYLANX_KERAS_SUPPORT_CONV2D_OPERATION)
#define PHYLANX_KERAS_SUPPORT_CONV2D_OPERATION

#include <phylanx/config.hpp>
#include <phylanx/execution_tree/primitives/base_primitive.hpp>
#include <phylanx/execution_tree/primitives/primitive_component_base.hpp>

#include <hpx/datastructures/optional.hpp>
#include <hpx/lcos/future.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace phylanx { namespace execution_tree { namespace primitives {
    /// \brief returns 2D convoltion
    /// \param x              a tensor
    /// \param kernel         a tensor, the filter
    /// \param padding        Padding mode, either `valid`, `same` or `causal`
    /// \param strides        The step to apply convolution
    /// \param dilation_rate  The rate to sample x in each step

    class conv2d_operation
      : public primitive_component_base
      , public std::enable_shared_from_this<conv2d_operation>
    {
    protected:
        hpx::future<primitive_argument_type> eval(
            primitive_arguments_type const& operands,
            primitive_arguments_type const& args,
            eval_context ctx) const override;

    public:
        static match_pattern_type const match_data;

        conv2d_operation() = default;

        conv2d_operation(primitive_arguments_type&& operands,
            std::string const& name, std::string const& codename);

    private:
        template <typename T>
        primitive_argument_type calculate_conv2d(
            ir::node_data<T>&& arg, ir::node_data<T>&& kernel) const;
        template <typename T>
        primitive_argument_type calculate_conv2d(ir::node_data<T>&& arg,
            ir::node_data<T>&& kernel,
            ir::range&& shape) const;
        template <typename T>
        primitive_argument_type calculate_conv3d(
            ir::node_data<T>&& arg, ir::node_data<T>&& kernel) const;
        template <typename T>
        primitive_argument_type calculate_conv3d(ir::node_data<T>&& arg,
            ir::node_data<T>&& kernel,
            ir::range&& shape) const;
        template <typename T>
        primitive_argument_type calculate_conv(ir::node_data<T>&& arg,
            ir::node_data<T>&& kernel) const;
        template <typename T>
        primitive_argument_type calculate_conv(ir::node_data<T>&& arg,
            ir::node_data<T>&& kernel,
            ir::range&& shape) const;
    };

    inline primitive create_conv2d_operation(hpx::id_type const& locality,
        primitive_arguments_type&& operands, std::string const& name = "",
        std::string const& codename = "")
    {
        return create_primitive_component(
            locality, "conv2d", std::move(operands), name, codename);
    }
}}}

#endif
