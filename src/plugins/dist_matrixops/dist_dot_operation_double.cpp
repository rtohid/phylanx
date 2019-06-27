//  Copyright (c) 2017-2019 Hartmut Kaiser
//  Copyright (c) 2017 Parsa Amini
//  Copyright (c) 2019 Bita Hasheminezhad
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>
#include <phylanx/ir/node_data.hpp>
#include <phylanx/plugins/dist_matrixops/dist_dot_operation.hpp>
#include <phylanx/plugins/dist_matrixops/dist_dot_operation_impl.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace phylanx { namespace dist_matrixops { namespace primitives
{
    // explicitly instantiate the required functions

    ///////////////////////////////////////////////////////////////////////////
    template execution_tree::primitive_argument_type dist_dot_operation::dot0d(
        ir::node_data<double>&&, ir::node_data<double>&&) const;

    template execution_tree::primitive_argument_type dist_dot_operation::dot1d(
        ir::node_data<double>&&, ir::node_data<double>&&) const;

    template execution_tree::primitive_argument_type dist_dot_operation::dot2d(
        ir::node_data<double>&&, ir::node_data<double>&&) const;

#if defined(PHYLANX_HAVE_BLAZE_TENSOR)
    template execution_tree::primitive_argument_type dist_dot_operation::dot3d(
        ir::node_data<double>&&, ir::node_data<double>&&) const;
#endif
}}}
