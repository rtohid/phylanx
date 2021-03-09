from __future__ import absolute_import
from __future__ import annotations

__license__ = """
Copyright (c) 2021 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""
import ast
import inspect
import numpy

from abc import abstractmethod
from collections import defaultdict, deque, OrderedDict, UserList
from types import FunctionType
from typing import Any, Iterator, List, NoReturn, Union


class Load:
    ...


class Store:
    ...


class IR(UserList):
    def add_node(self, node: ast.AST) -> NoReturn:
        self.insert(0, node)


class IRNode:
    ...


class Buffer(IRNode):
    Table = defaultdict()

    def __init__(self,
                 name: str,
                 namespace: List,
                 slice_: list() = List,
                 context: None = Union[Load, Store]) -> None:
        self.name = name
        self.namespace = namespace
        self.slice = slice_
        self.context = context

    def is_array(self):
        if self.slice_:
            return True
        else:
            return False


class Call(IRNode):
    def __init__(self, name: str, namespace: List, args: List) -> None:
        self.name = name
        self.namespace = namespace
        self.args = args


class Function(IRNode):
    def __init__(self, name: str, namespace: List) -> None:
        self.name = name
        self.namespace = namespace
        self.id = (name, namespace)

        self.arg_list = list()
        self.body = OrderedDict()
        self.num_returns = 0
        self.returns = OrderedDict()

    def add_expression(self, expr):
        self.body[expr] = expr

    def add_statement(self, statement: Statement):
        self.body[statement.name] = statement

    def add_return(self, return_val):
        self.num_returns += 1
        self.returns[self.num_returns] = return_val


class Input(IRNode):
    ...


class IterSpace:
    def __init__(self, space: Union[str, Call, List]) -> None:
        self.space = space


class Var:
    def __init__(self, name: str, iter_space: IterSpace) -> None:
        self.name = name
        self.iter_space = iter_space


class Loop:
    def __init__(self, iterator: str, bounds: Var, body: Union[Statement,
                                                               Loop]) -> None:
        self.iterator = iterator
        self.bounds = bounds
        self.body = body


class Statement:
    statements = OrderedDict()

    def __init__(self, lhs: Buffer, rhs: IRNode):
        self.name = 'S' + self.statement_num()
        self.lhs = lhs
        self.rhs = rhs
        self.add_statement(self)

    @classmethod
    def statement_num(cls):
        return str(len(cls.statements) + 1)

    @classmethod
    def add_statement(cls, statement: Statement) -> None:
        cls.statements[statement.name] = statement

    @property
    def lhs(self):
        return self._lhs

    @lhs.setter
    def lhs(self, targets):
        self._lhs = targets

    @property
    def rhs(self):
        return self._rhs

    @rhs.setter
    def rhs(self, expr):
        self._rhs = expr


class Namespace:
    complete_namespace = deque()

    def __init__(self, namespace: str = ''):
        self.namespace = namespace

    def __enter__(self):
        if self.namespace:
            self.push(self.namespace)
        return self

    def __exit__(self, type, value, traceback):
        if self.namespace:
            self.pop()

    @classmethod
    def get_namespace(cls):
        return tuple(cls.complete_namespace)

    @classmethod
    def push(cls, namespace):
        cls.complete_namespace.append(namespace)

    @classmethod
    def pop(cls):
        cls.complete_namespace.pop()

    @classmethod
    def __str__(cls):
        return '+'.join(cls.complete_namespace)


class IRBuilder(ast.NodeVisitor):
    def __init__(self, fn: FunctionType) -> None:
        self.fn = fn

        py_src = inspect.getsource(fn)
        py_ast = ast.parse(py_src)
        self.ir = IR()
        self.init_ir(py_ast)
        # self.symbol_table = TaskTable(self.task)
        self.src = self.build(py_ast)

    def get_ir(self) -> IR:
        return self.ir

    def add_node(self, node: IRNode):
        self.ir.add_node(node)

    @abstractmethod
    def build(self, node):
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    def init_ir(self, ast_: ast.AST) -> NoReturn:
        self.visit(ast_)

    def visit_Assign(self, node: ast.Assign) -> Any:
        targets = [self.visit(target) for target in node.targets]
        for target in targets:
            target.context = Load()

        if 1 < len(targets):
            raise NotImplementedError(
                'Mutli-target assignments are not supported.')

        value = self.visit(node.value)
        statement = Statement(targets[0], value)
        self.add_node((statement, node))
        return statement

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        ops = {
            ast.Add: '__add',
            ast.Sub: '__sub',
            ast.Mult: '__mul',
            ast.Div: '__div',
            ast.Mod: '__mod',
            ast.Pow: 'power'
        }
        left = self.visit(node.left)
        right = self.visit(node.right)
        args = [left, right]
        op = ops[type(node.op)]

        with Namespace() as ns:
            return Call(op, ns.get_namespace(), args)

    def visit_Call(self, node: ast.Call) -> Any:
        name = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        with Namespace() as ns:
            return Call(name, ns.get_namespace(), args)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_ExtSlice(self, node: ast.ExtSlice) -> Any:
        return list(self.visit(slice_) for slice_ in node.dims)

    def visit_For(self, node: ast.For) -> Any:
        target = self.visit(node.target)
        iter_space = IterSpace(self.visit(node.iter))
        body = list(self.visit(statement) for statement in node.body)
        iterator = Var(target, iter_space)
        loop = Loop(target, iterator, body)
        return loop

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        fn_name = node.name

        with Namespace(fn_name) as ns:
            fn = Function(fn_name, ns.get_namespace())

            for statement in node.body:
                if isinstance(statement, ast.Return):
                    fn.add_return(self.visit(statement))
                elif isinstance(statement, Statement):
                    fn.add_statement(statement)
                else:
                    fn.add_expression(statement)


        # self.add_node((fn, node))
        return fn

    def visit_Index(self, node: ast.Index) -> Any:
        return self.visit(node.value)

    def visit_List(self, node: ast.List) -> Any:
        lst = list(self.visit(element) for element in node.elts)
        return lst

    def visit_Module(self, node: ast.Module) -> ast.Module:
        file_name = inspect.getfile(self.fn).split('/')[-1].split('.')[0]
        module_name = "__phy_task+" + file_name

        with Namespace(module_name):
            for n in node.body:
                self.visit(n)
        
        return

    def visit_Name(self, node: ast.Name) -> str:
        return node.id

    def visit_Slice(self, node: ast.Slice) -> Any:
        try:
            lower = self.visit(node.lower)
        except:
            lower = None

        try:
            upper = self.visit(node.upper)
        except:
            upper = None

        try:
            step = self.visit(node.step)
        except:
            step = 1

        return (lower, upper, step)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        slices = list()

        def _NestedSubscript(node: ast.Subscript) -> None:
            slices.append(self.visit(node.slice))
            if isinstance(node.value, ast.Subscript):
                _NestedSubscript(node.value)
            else:
                slices.append(self.visit(node.value))

        slices.append(self.visit(node.slice))
        if isinstance(node.value, ast.Subscript):
            _NestedSubscript(node.value)
        else:
            slices.append(self.visit(node.value))

        slices.reverse()

        with Namespace() as ns:
            if isinstance(slices, list):
                buffer = Buffer(slices[0], ns.get_namespace())
                buffer.slice = slices[1:]
            else:
                buffer = Buffer(slices, ns.get_namespace())

        return buffer

    def visit_Return(self, node: ast.Return) -> Any:
        value = self.visit(node.value)
        return value

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        tpl = tuple(self.visit(element) for element in node.elts)
        return tpl


class Polyhedral:
    def __init__(self, fn: FunctionType) -> None:
        self.fn = fn
        self.ir: IRBuilder = build_ir(fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def build_ir(fn: FunctionType):
    ir: IRBuilder = IRBuilder(fn)
    return ir.get_ir()
