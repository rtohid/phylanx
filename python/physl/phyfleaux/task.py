from __future__ import absolute_import
from __future__ import annotations

__license__ = """
Copyright (c) 2021 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""

import ast
import inspect

from collections import OrderedDict
from typing import Union
from types import FunctionType


class Task:
    def __init__(self, fn: Union[FunctionType, Task]) -> None:

        self.fn = fn
        self.id = self.fn.__hash__()

        self.py_code = fn.__code__
        self.py_ast = ast.parse((inspect.getsource(fn)))

        self.called = 0

    def __call__(self, *args, **kwargs):
        # inspect.signature(fn).bind()
        self.called += 1
        return self.fn(*args, **kwargs)

    def __hash__(self):
        return self.id
