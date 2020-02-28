# Copyright (c) 2020 R. Tohid
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from __future__ import absolute_import

from flow.control import Function


def func(x):
    return x + 1


def test_function():
    py_funcion = Function(func)
    _src = inspect.getsource(func)
    _ast = ast.parse(_src)

    _cfg = py_funcion.pst.cfg
    assert py_funcion.fn == func
    assert _cfg.python_code == _src
    assert ast.dump(_cfg.ast) == ast.dump(_ast)