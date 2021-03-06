//  Copyright (c) 2017-2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/config.hpp>
#include <phylanx/execution_tree/primitives/file_read.hpp>
#include <phylanx/ir/node_data.hpp>
#include <phylanx/util/serialization/ast.hpp>
#include <phylanx/util/serialization/execution_tree.hpp>

#include <hpx/include/lcos.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/util.hpp>
#include <hpx/throw_exception.hpp>

#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace phylanx { namespace execution_tree { namespace primitives
{
    ///////////////////////////////////////////////////////////////////////////
    primitive create_file_read(hpx::id_type const& locality,
        std::vector<primitive_argument_type>&& operands,
        std::string const& name, std::string const& codename)
    {
        static std::string type("file_read");
        return create_primitive_component(
            locality, type, std::move(operands), name, codename);
    }

    match_pattern_type const file_read::match_data =
    {
        hpx::util::make_tuple("file_read",
            std::vector<std::string>{"file_read(_1)"},
            &create_file_read, &create_primitive<file_read>)
    };

    ///////////////////////////////////////////////////////////////////////////
    file_read::file_read(std::vector<primitive_argument_type>&& operands,
            std::string const& name, std::string const& codename)
      : primitive_component_base(std::move(operands), name, codename)
    {}

    // read data from given file and return content
    hpx::future<primitive_argument_type> file_read::eval(
        std::vector<primitive_argument_type> const& args) const
    {
        if (operands_.size() != 1)
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "phylanx::execution_tree::primitives::file_read::eval",
                execution_tree::generate_error_message(
                    "the file_read primitive requires exactly one "
                        "literal argument",
                    name_, codename_));
        }

        if (!valid(operands_[0]))
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "phylanx::execution_tree::primitives::file_read::eval",
                execution_tree::generate_error_message(
                    "the file_read primitive requires that the given "
                        "operand is valid",
                    name_, codename_));
        }

        std::string filename =
            string_operand_sync(operands_[0], args, name_, codename_);
        std::ifstream infile(filename.c_str(),
            std::ios::binary | std::ios::in | std::ios::ate);

        if (!infile.is_open())
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "phylanx::execution_tree::primitives::file_read::eval",
                execution_tree::generate_error_message(
                    "couldn't open file: " + filename,
                    name_, codename_));
        }

        std::streamsize count = infile.tellg();
        infile.seekg(0);

        std::vector<char> data;
        data.resize(count);

        if (!infile.read(data.data(), count))
        {
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "phylanx::execution_tree::primitives::file_read::eval",
                execution_tree::generate_error_message(
                    "couldn't read expected number of bytes from file: " +
                        filename,
                    name_, codename_));
        }

        // assume data in file is result of a serialized primitive_argument_type
        primitive_argument_type val;
        phylanx::util::unserialize(data, val);

        return hpx::make_ready_future(std::move(val));
    }
}}}
