# Copyright (c) 2017-2018 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    allow_empty_blocks_278
    assign_wrong_variable_245
    cout_noargs_prints_nothing_159
    cout_prints_twice_155
    define_loses_values_177
    expression_topology_213
    list_of_variables_244
    negative_integral_value_131
    no_integers_313
    out_of_scope_variable_232
    support_nil_314
    vector_of_variables_244
   )

set(assign_wrong_variable_245_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(cout_noargs_prints_nothing_159_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(cout_prints_twice_155_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(define_loses_values_177_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(out_of_scope_variable_232_FLAGS COMPONENT_DEPENDENCIES iostreams)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add executable
  add_phylanx_executable(${test}_test
    SOURCES ${sources}
    ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Regression/ExecutionTree/")

  add_phylanx_regression_test("execution_tree" ${test} ${${test}_PARAMETERS})

  add_phylanx_pseudo_target(tests.regressions.execution_tree_dir.${test})
  add_phylanx_pseudo_dependencies(tests.regressions.execution_tree_dir
    tests.regressions.execution_tree_dir.${test})
  add_phylanx_pseudo_dependencies(tests.regressions.execution_tree_dir.${test}
    ${test}_test_exe)

endforeach()


