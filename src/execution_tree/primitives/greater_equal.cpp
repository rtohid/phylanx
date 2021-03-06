//  Copyright (c) 2017-2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>
#include <phylanx/execution_tree/primitives/greater_equal.hpp>
#include <phylanx/ir/node_data.hpp>

#include <hpx/include/lcos.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/util.hpp>
#include <hpx/throw_exception.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace phylanx { namespace execution_tree { namespace primitives
{
    ///////////////////////////////////////////////////////////////////////////
    primitive create_greater_equal(hpx::id_type const& locality,
        std::vector<primitive_argument_type>&& operands,
        std::string const& name, std::string const& codename)
    {
        static std::string type("__ge");
        return create_primitive_component(
            locality, type, std::move(operands), name, codename);
    }

    match_pattern_type const greater_equal::match_data =
    {
        hpx::util::make_tuple("__ge",
            std::vector<std::string>{"_1 >= _2", "__ge(_1, _2)"},
            &create_greater_equal, &create_primitive<greater_equal>)
    };

    ///////////////////////////////////////////////////////////////////////////
    greater_equal::greater_equal(std::vector<primitive_argument_type>&& operands,
            std::string const& name, std::string const& codename)
      : primitive_component_base(std::move(operands), name, codename)
    {}

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    primitive_argument_type greater_equal::greater_equal0d1d(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        // TODO: SIMD functionality should be added, blaze implementation
        // is not currently available
        if (rhs.is_ref())
        {
            rhs = blaze::map(rhs.vector(),
                [&](double x) { return (x >= lhs.scalar()); });
        }
        else
        {
            rhs.vector() = blaze::map(rhs.vector(),
                [&](double x) { return (x >= lhs.scalar()); });
        }

        return primitive_argument_type(
            ir::node_data<std::uint8_t>{std::move(rhs)});
    }

    template <typename T>
    primitive_argument_type greater_equal::greater_equal0d2d(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        // TODO: SIMD functionality should be added, blaze implementation
        // is not currently available
        if (rhs.is_ref())
        {
            rhs = blaze::map(rhs.matrix(),
                [&](double x) { return (x >= lhs.scalar()); });
        }
        else
        {
            rhs.matrix() = blaze::map(rhs.matrix(),
                [&](double x) { return (x >= lhs.scalar()); });
        }

        return primitive_argument_type(
            ir::node_data<std::uint8_t>{std::move(rhs)});
    }

    template <typename T>
    primitive_argument_type greater_equal::greater_equal0d(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        std::size_t rhs_dims = rhs.num_dimensions();
        switch(rhs_dims)
        {
        case 0:
            return primitive_argument_type(
                ir::node_data<std::uint8_t>{lhs.scalar() >= rhs.scalar()});

        case 1:
            return greater_equal0d1d(std::move(lhs), std::move(rhs));

        case 2:
            return greater_equal0d2d(std::move(lhs), std::move(rhs));

        default:
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::greater_equal0d",
                execution_tree::generate_error_message(
                    "the operands have incompatible number of "
                        "dimensions",
                    name_, codename_));
        }
    }

    template <typename T>
    primitive_argument_type greater_equal::greater_equal1d0d(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        // TODO: SIMD functionality should be added, blaze implementation
        // is not currently available
        if (lhs.is_ref())
        {
            lhs = blaze::map(lhs.vector(),
                [&](double x) { return (x >= rhs.scalar()); });
        }
        else
        {
            lhs.vector() = blaze::map(lhs.vector(),
                [&](double x) { return (x >= rhs.scalar()); });
        }

        return primitive_argument_type(
            ir::node_data<std::uint8_t>{std::move(lhs)});
    }

    template <typename T>
    primitive_argument_type greater_equal::greater_equal1d1d(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        std::size_t lhs_size = lhs.dimension(0);
        std::size_t rhs_size = rhs.dimension(0);

        if (lhs_size != rhs_size)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::greater_equal1d1d",
                execution_tree::generate_error_message(
                    "the dimensions of the operands do not match",
                    name_, codename_));
        }

        // TODO: SIMD functionality should be added, blaze implementation
        // is not currently available
        if (lhs.is_ref())
        {
            lhs = blaze::map(lhs.vector(), rhs.vector(),
                [&](double x, double y) { return (x >= y); });
        }
        else
        {
            lhs.vector() = blaze::map(lhs.vector(), rhs.vector(),
                [&](double x, double y) { return (x >= y); });
        }

        return primitive_argument_type(
            ir::node_data<std::uint8_t>{std::move(lhs)});
    }

    template <typename T>
    primitive_argument_type greater_equal::greater_equal1d2d(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        auto cv = lhs.vector();
        auto cm = rhs.matrix();

        if (cv.size() != cm.columns())
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::greater_equal1d2d",
                execution_tree::generate_error_message(
                    "the dimensions of the operands do not match",
                    name_, codename_));
        }

        // TODO: SIMD functionality should be added, blaze implementation
        // is not currently available
        if (rhs.is_ref())
        {
            blaze::DynamicMatrix<double> m{cm.rows(), cm.columns()};

            for (size_t i = 0UL; i != cm.rows(); i++)
                blaze::row(m, i) = blaze::map(blaze::row(cm, i),
                    blaze::trans(cv),
                    [](double x, double y) { return x >= y; });
            return primitive_argument_type(
                ir::node_data<std::uint8_t>{std::move(m)});
        }

        for (size_t i = 0UL; i != cm.rows(); i++)
            blaze::row(cm, i) = blaze::map(blaze::row(cm, i),
                blaze::trans(cv),
                [](double x, double y) { return x >= y; });
        return primitive_argument_type(
            ir::node_data<std::uint8_t>{std::move(rhs)});
    }

    template <typename T>
    primitive_argument_type greater_equal::greater_equal1d(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        std::size_t rhs_dims = rhs.num_dimensions();
        switch(rhs_dims)
        {
        case 0:
            return greater_equal1d0d(std::move(lhs), std::move(rhs));

        case 1:
            return greater_equal1d1d(std::move(lhs), std::move(rhs));

        case 2:
            return greater_equal1d2d(std::move(lhs), std::move(rhs));

        default:
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::greater_equal1d",
                execution_tree::generate_error_message(
                    "the operands have incompatible number of "
                        "dimensions",
                    name_, codename_));
        }
    }

    template <typename T>
    primitive_argument_type greater_equal::greater_equal2d0d(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        std::size_t lhs_size = lhs.dimension(0);
        std::size_t rhs_size = rhs.dimension(0);

        // TODO: SIMD functionality should be added, blaze implementation
        // is not currently available
        if (lhs.is_ref())
        {
            lhs = blaze::map(lhs.matrix(),
                [&](double x) { return (x >= rhs.scalar()); });
        }
        else
        {
            lhs.matrix() = blaze::map(lhs.matrix(),
                [&](double x) { return (x >= rhs.scalar()); });
        }

        return primitive_argument_type(
            ir::node_data<std::uint8_t>{std::move(lhs)});
    }

    template <typename T>
    primitive_argument_type greater_equal::greater_equal2d1d(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        auto cv = rhs.vector();
        auto cm = lhs.matrix();

        if (cv.size() != cm.columns())
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::greater_equal2d1d",
                execution_tree::generate_error_message(
                    "the dimensions of the operands do not match",
                    name_, codename_));
        }

        // TODO: SIMD functionality should be added, blaze implementation
        // is not currently available
        if (lhs.is_ref())
        {
            blaze::DynamicMatrix<double> m{cm.rows(), cm.columns()};

            for (size_t i = 0UL; i != cm.rows(); i++)
                blaze::row(m, i) = blaze::map(blaze::row(cm, i),
                    blaze::trans(cv),
                    [](double x, double y) { return x >= y; });
            return primitive_argument_type(
                ir::node_data<std::uint8_t>{std::move(m)});
        }

        for (size_t i = 0UL; i != cm.rows(); i++)
            blaze::row(cm, i) = blaze::map(blaze::row(cm, i),
                blaze::trans(cv),
                [](double x, double y) { return x >= y; });

        return primitive_argument_type(
            ir::node_data<std::uint8_t>{std::move(lhs)});
    }

    template <typename T>
    primitive_argument_type greater_equal::greater_equal2d2d(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        auto lhs_size = lhs.dimensions();
        auto rhs_size = rhs.dimensions();

        if (lhs_size != rhs_size)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::greater_equal2d2d",
                execution_tree::generate_error_message(
                    "the dimensions of the operands do not match",
                    name_, codename_));
        }

        // TODO: SIMD functionality should be added, blaze implementation
        //       is not currently available
        if (lhs.is_ref())
        {
            lhs = blaze::map(lhs.matrix(), rhs.matrix(),
                [&](double x, double y) { return (x >= y); });
        }
        else
        {
            lhs.matrix() = blaze::map(lhs.matrix(), rhs.matrix(),
                [&](double x, double y) { return (x >= y); });
        }

        return primitive_argument_type(
            ir::node_data<std::uint8_t>{std::move(lhs)});
    }

    template <typename T>
    primitive_argument_type greater_equal::greater_equal2d(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        std::size_t rhs_dims = rhs.num_dimensions();
        switch(rhs_dims)
        {
        case 0:
            return greater_equal2d0d(std::move(lhs), std::move(rhs));

        case 1:
            return greater_equal2d1d(std::move(lhs), std::move(rhs));

        case 2:
            return greater_equal2d2d(std::move(lhs), std::move(rhs));

        default:
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::greater_equal2d",
                execution_tree::generate_error_message(
                    "the operands have incompatible number of "
                        "dimensions",
                    name_, codename_));
        }
    }

    template <typename T>
    primitive_argument_type greater_equal::greater_equal_all(
        ir::node_data<T>&& lhs, ir::node_data<T>&& rhs) const
    {
        std::size_t lhs_dims = lhs.num_dimensions();
        switch (lhs_dims)
        {
        case 0:
            return greater_equal0d(std::move(lhs), std::move(rhs));

        case 1:
            return greater_equal1d(std::move(lhs), std::move(rhs));

        case 2:
            return greater_equal2d(std::move(lhs), std::move(rhs));

        default:
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::greater_equal_all",
                execution_tree::generate_error_message(
                    "left hand side operand has unsupported number of "
                        "dimensions",
                    name_, codename_));
        }
    }

    struct greater_equal::visit_greater_equal
    {
        template <typename T1, typename T2>
        primitive_argument_type operator()(T1, T2) const
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::eval",
                execution_tree::generate_error_message(
                    "left hand side and right hand side are incompatible "
                        "and can't be compared",
                    greater_equal_.name_, greater_equal_.codename_));
        }

        primitive_argument_type operator()(
            ir::node_data<primitive_argument_type>&&,
            ir::node_data<primitive_argument_type>&&) const
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::eval",
                execution_tree::generate_error_message(
                    "left hand side and right hand side are incompatible "
                        "and can't be compared",
                    greater_equal_.name_, greater_equal_.codename_));
        }

        primitive_argument_type operator()(std::vector<ast::expression>&&,
            std::vector<ast::expression>&&) const
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::eval",
                execution_tree::generate_error_message(
                    "left hand side and right hand side are incompatible "
                        "and can't be compared",
                    greater_equal_.name_, greater_equal_.codename_));
        }

        primitive_argument_type operator()(
            ast::expression&&, ast::expression&&) const
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::eval",
                execution_tree::generate_error_message(
                    "left hand side and right hand side are incompatible "
                        "and can't be compared",
                    greater_equal_.name_, greater_equal_.codename_));
        }

        primitive_argument_type operator()(primitive&&, primitive&&) const
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::eval",
                execution_tree::generate_error_message(
                    "left hand side and right hand side are incompatible "
                        "and can't be compared",
                    greater_equal_.name_, greater_equal_.codename_));
        }

        template <typename T>
        primitive_argument_type operator()(T && lhs, T && rhs) const
        {
            return primitive_argument_type(
                ir::node_data<std::uint8_t>{lhs >= rhs});
        }

        primitive_argument_type operator()(
            util::recursive_wrapper<
                std::vector<primitive_argument_type>>&&,
            util::recursive_wrapper<
                std::vector<primitive_argument_type>>&&) const
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::eval",
                execution_tree::generate_error_message(
                    "left hand side and right hand side are incompatible "
                        "and can't be compared",
                    greater_equal_.name_, greater_equal_.codename_));
        }

        primitive_argument_type operator()(
            ir::node_data<double>&& lhs, std::int64_t&& rhs) const
        {
            if (lhs.num_dimensions() != 0)
            {
                return greater_equal_.greater_equal_all(
                    std::move(lhs), operand_type(std::move(rhs)));
            }
            return primitive_argument_type(
                ir::node_data<std::uint8_t>{lhs[0] >= rhs});
        }

        primitive_argument_type operator()(
            std::int64_t&& lhs, ir::node_data<double>&& rhs) const
        {
            if (rhs.num_dimensions() != 0)
            {
                return greater_equal_.greater_equal_all(
                    operand_type(std::move(lhs)), std::move(rhs));
            }
            return primitive_argument_type(
                ir::node_data<std::uint8_t>{lhs >= rhs[0]});
        }

        primitive_argument_type operator()(
            ir::node_data<std::uint8_t>&& lhs, std::int64_t&& rhs) const
        {
            if (lhs.num_dimensions() != 0)
            {
                return greater_equal_.greater_equal_all(std::move(lhs),
                    ir::node_data<std::uint8_t>{rhs != 0});
            }
            return primitive_argument_type(
                ir::node_data<std::uint8_t>{lhs[0] >= rhs});
        }

        primitive_argument_type operator()(
            std::int64_t&& lhs, ir::node_data<std::uint8_t>&& rhs) const
        {
            if (rhs.num_dimensions() != 0)
            {
                return greater_equal_.greater_equal_all(
                    ir::node_data<std::uint8_t>{lhs != 0},
                    std::move(rhs));
            }
            return primitive_argument_type(
                ir::node_data<std::uint8_t>{lhs >= rhs[0]});
        }

        primitive_argument_type operator()(
            operand_type&& lhs, operand_type&& rhs) const
        {
            return greater_equal_.greater_equal_all(
                std::move(lhs), std::move(rhs));
        }

        primitive_argument_type operator()(ir::node_data<std::uint8_t>&& lhs,
            ir::node_data<std::uint8_t>&& rhs) const
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::eval",
                execution_tree::generate_error_message(
                    "left hand side and right hand side can't be compared",
                    greater_equal_.name_, greater_equal_.codename_));
        }

        greater_equal const& greater_equal_;
    };

    hpx::future<primitive_argument_type> greater_equal::eval(
        std::vector<primitive_argument_type> const& operands,
        std::vector<primitive_argument_type> const& args) const
    {
        if (operands.size() != 2)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::eval",
                execution_tree::generate_error_message(
                    "the greater_equal primitive requires exactly two "
                        "operands",
                    name_, codename_));
        }

        if (!valid(operands[0]) || !valid(operands[1]))
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "greater_equal::eval",
                execution_tree::generate_error_message(
                    "the greater_equal primitive requires that the "
                        "arguments given by the operands array are valid",
                    name_, codename_));
        }

        auto this_ = this->shared_from_this();
        return hpx::dataflow(hpx::launch::sync, hpx::util::unwrapping(
            [this_](primitive_argument_type&& op1,
                    primitive_argument_type&& op2)
            ->  primitive_argument_type
            {
                return primitive_argument_type(
                    util::visit(visit_greater_equal{*this_},
                        std::move(op1.variant()),
                        std::move(op2.variant())));
            }),
            literal_operand(operands[0], args, name_, codename_),
            literal_operand(operands[1], args, name_, codename_));
    }

    //////////////////////////////////////////////////////////////////////////
    // Implement '>=' for all possible combinations of lhs and rhs
    hpx::future<primitive_argument_type> greater_equal::eval(
        std::vector<primitive_argument_type> const& args) const
    {
        if (operands_.empty())
        {
            return eval(args, noargs);
        }
        return eval(operands_, args);
    }
}}}
