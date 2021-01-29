from __future__ import absolute_import
from __future__ import annotations

__license__ = """ 
Copyright (c) 2021 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""

import ast
import inspect

from abc import ABC
from typing import Any

from physl.control import Task
from physl.transformations import Transpiler


class Expr(ABC):
    def __init__(self,
                 name: str,
                 scope,
                 lineno: int = None,
                 col_offset: int = None):
        self.name = name
        self.scope = scope
        self.lineno = lineno
        self.col_offset = col_offset


class Variable(Expr):
    def __init__(self,
                 name: str,
                 scope,
                 lineno: int,
                 col_offset: int,
                 dtype=None):
        super().__init__(name, scope, lineno, col_offset)
        self.type = dtype

    def __eq__(self, other):
        self_ = (self.name, self.scope, self.lineno, self.col_offset,
                 self.type)
        other_ = (other.name, other.scope, other.lineno, other.col_offset,
                  self.type)
        return self_ == other_


class Array(Variable):
    def __init__(self,
                 name: str,
                 scope,
                 lineno: int,
                 col_offset: int,
                 dimension: int = None,
                 shape: int = None):
        super().__init__(name, scope, lineno, col_offset)

        if dimension and dimension < 1:
            raise ValueError(
                f"Arrays must have 1 or more dimension(s). {dimension} given.")
        self.dimensionality = dimension

        if shape and not len(shape) == dimension:
            raise ValueError(
                f"Array dimensionality({dimension}) does not match the shape({shape})."
            )
        self.shape = shape

    def __eq__(self, other):
        self_ = (self.name, self.scope, self.lineno, self.col_offset,
                 self.dimensionality, self.shape)
        other_ = (other.name, other.scope, other.lineno, other.col_offset,
                  other.dimensionality, other.shape)
        return self_ == other_


class Function(Expr):
    def __init__(self,
                 name: str,
                 scope,
                 lineno: int,
                 col_offset: int,
                 dtype: int,
                 str=None):
        super().__init__(name, scope, lineno, col_offset)
        self.args_list = list()
        self.dtype = dtype

    def add_arg(self, argument):
        if isinstance(argument, list):
            self.args_list.extend(argument)
        else:
            self.args_list.append(argument)

    def insert_arg(self, argument, position):
        self.args_list.insert(position, argument)

    def prepend_arg(self, argument):
        self.insert_arg(argument, 0)

    def arguments(self):
        return self.args_list

    def __eq__(self, other):
        self_ = (self.name, self.scope, self.lineno, self.col_offset,
                 self.args_list, self.dtype)
        other_ = (other.name, other.scope, other.lineno, other.col_offset,
                  other.args_list, other.dtype)
        return self_ == other_


class FunctionCall(Function):
    pass


class PhySL(Transpiler):
    def transpile(self, node):
        self.target = super().visit_(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        args_specs = inspect.getfullargspec(self.task.fn)
        fn_params = args_specs.args

        fn_name = node.name


    def visit_Module(self, node: ast.Module) -> str:
        return self.visit_(node.body[0])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.task(*args, **kwargs)
