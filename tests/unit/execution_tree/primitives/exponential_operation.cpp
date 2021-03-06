//   Copyright (c) 2017 Bibek Wagle
//
//   Distributed under the Boost Software License, Version 1.0. (See accompanying
//   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <phylanx/phylanx.hpp>

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>
#include <utility>

void test_exponential_operation_0d()
{
    phylanx::execution_tree::primitive lhs =
        phylanx::execution_tree::primitives::create_variable(
            hpx::find_here(), phylanx::ir::node_data<double>(5.0));

    phylanx::execution_tree::primitive exponential =
        phylanx::execution_tree::primitives::create_exponential_operation(
            hpx::find_here(),
            std::vector<phylanx::execution_tree::primitive_argument_type>{
                std::move(lhs)});

    hpx::future<phylanx::execution_tree::primitive_argument_type> f =
        exponential.eval();
    HPX_TEST_EQ(std::exp(5.0),
        phylanx::execution_tree::extract_numeric_value(f.get())[0]);
}

void test_exponential_operation_0d_lit()
{
    phylanx::ir::node_data<double> lhs(5.0);

    phylanx::execution_tree::primitive exponential =
        phylanx::execution_tree::primitives::create_exponential_operation(
            hpx::find_here(),
            std::vector<phylanx::execution_tree::primitive_argument_type>{
                std::move(lhs)
            });

    hpx::future<phylanx::execution_tree::primitive_argument_type> f =
        exponential.eval();
    HPX_TEST_EQ(std::exp(5.0),
        phylanx::execution_tree::extract_numeric_value(f.get())[0]);
}

void test_exponential_operation_2d()
{
    blaze::Rand<blaze::DynamicMatrix<double>> gen{};
    blaze::DynamicMatrix<double> m = gen.generate(42UL, 42UL);

    phylanx::execution_tree::primitive lhs =
        phylanx::execution_tree::primitives::create_variable(
            hpx::find_here(), phylanx::ir::node_data<double>(m));

    phylanx::execution_tree::primitive exponential =
        phylanx::execution_tree::primitives::create_exponential_operation(
            hpx::find_here(),
            std::vector<phylanx::execution_tree::primitive_argument_type>{
                std::move(lhs)
            });

    hpx::future<phylanx::execution_tree::primitive_argument_type> f =
        exponential.eval();

    blaze::DynamicMatrix<double> expected = blaze::exp(m);
    HPX_TEST_EQ(
        phylanx::ir::node_data<double>(std::move(expected)),
        phylanx::execution_tree::extract_numeric_value(f.get()));
}

int main(int argc, char* argv[])
{
    test_exponential_operation_0d();
    test_exponential_operation_0d_lit();

    test_exponential_operation_2d();

    return hpx::util::report_errors();
}


