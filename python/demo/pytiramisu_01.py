from __future__ import absolute_import
from __future__ import annotations

__license__ = """
Copyright (c) 2021 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""

import ast
import inspect
import symtable

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from numpy import ndarray
from typing import Any, List, Union


class Namespace:
    complete_namespace = deque()

    def __init__(self, namespace):
        self.namespace = namespace

    def __enter__(self):
        self.push(self.namespace)
        return self

    def __exit__(self, type, value, traceback):
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


class Symbol:
    def __init__(self, name: str, namespace: Namespace) -> None:
        self.name = name
        self.namespace = namespace


class Fn(Symbol):
    def __init__(self, name: str, namespace: Namespace) -> None:
        super().__init__(name, namespace)


class Var(Symbol):
    def __init__(self, name: str, namespace: Namespace) -> None:
        super().__init__(name, namespace)


class SymbolTable:
    table = defaultdict()

    def __init__(self) -> None:
        ...

    @classmethod
    @abstractmethod
    def add_symbol(cls, symbol):
        ...

    @classmethod
    @abstractmethod
    def check_symbol(cls, symbol_name, symbol_type, namespace=None):
        ...


class TaskTable(SymbolTable):
    """Table of symbols within a task.

    Holds symbols' information that'll be used during transpilation.
    """
    def __init__(self, task: Task) -> None:
        src = task.py_src
        fn = task.fn
        file_name = inspect.getfile(fn).split('/')[-1].split('.')[0]
        table_name = "__phy_task+" + file_name
        self.py_table = symtable.symtable(src, file_name,
                                          "exec").get_children()[0]

        # self.table = defaultdict(lambda: list())
        self.build_table(table_name, self.py_table)
        self.register_table(self)

    def build_table(self, table_name: str, table: symtable.SymbolTable):
        with Namespace(table_name) as ns:
            symbols = table.get_symbols()
            namespace = ns.get_namespace()
            for sym in symbols:
                self.table[namespace].append(sym)

            children = table.get_children()
            for child in children:
                self.build_table(child.get_name(), child)

    @classmethod
    def register_table(cls, self: TaskTable):
        cls.global_table[self.py_table.get_id()] = self.table


class FnTable(SymbolTable):
    def __init__(self, task: Task) -> None:
        ...

    def create_table(self):
        ...

    @classmethod
    def add_symbol(cls, symbol):
        FnTable.table

    def check_symbol(cls, symbol_name, symbol_type, namespace=None):
        ...


class VarTable(SymbolTable):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def add_symbol(cls, symbol):
        VarTable.table


class ClassTable(SymbolTable):
    ...


_tables = {'function': FnTable, 'class': ClassTable}


class Load:
    ...


class Store:
    ...


class Del:
    ...


class Attribute:
    def __init__(self, lineno, col_offset) -> None:
        self._lineno = lineno
        self._col_offset = col_offset

    def gen_code(self):
        return f"${self.lineno}${self.col_offset}"

    @property
    def lineno(self):
        return self._lineno

    @lineno.setter
    def lineno(self, lineno):
        self._lineno = lineno

    @property
    def col_offset(self):
        return self._col_offset

    @col_offset.setter
    def col_offset(self, col_offset):
        self._col_offset = col_offset

    def __eq__(self, o: Attribute) -> bool:
        return (self.lineno, self.col_offset) == (o.lineno, o.col_offset)


class Expr(ABC):
    def __init__(self,
                 name: str,
                 namespace,
                 lineno: int = None,
                 col_offset: int = None):
        self.name = name
        self.namespace = namespace
        self.loc = Attribute(lineno, col_offset)

    def namespace_str(self):
        return str(self.namespace)


class Variable(Expr):
    def __init__(self,
                 name: str,
                 namespace,
                 lineno: int,
                 col_offset: int,
                 dtype=None):
        super().__init__(name, namespace, lineno, col_offset)
        self.type = dtype

    def __eq__(self, o: Variable):
        self_ = (self.name, self.namespace, self.loc)
        other_ = (o.name, o.namespace, o.lineno, o.loc)
        return self_ == other_


class Argument(Expr):
    def gen_code(self):
        return f'{self.name}{self.loc.gen_code()}'


class Buffer(Variable):
    def __init__(self,
                 name: str,
                 namespace,
                 lineno: int,
                 col_offset: int,
                 dimension: int = None,
                 shape: int = None):
        super().__init__(name, namespace, lineno, col_offset)

        if dimension and dimension < 1:
            raise ValueError(
                f"Arrays must have 1 or more dimension(s). {dimension} given.")
        self.dimensionality = dimension

        if shape and not len(shape) == dimension:
            raise ValueError(
                f"Array dimensionality({dimension}) does not match the shape({shape})."
            )
        self.shape = shape

    def __eq__(self, o: Buffer):
        self_ = (self.name, self.namespace, self.loc, self.dimensionality,
                 self.shape)
        other_ = (o.name, o.namespace, o.loc, o.dimensionality, o.shape)
        return self_ == other_


class PhyControl:
    def __init__(self, predicate, body):
        self.predicate = None
        self.body = list()


class Function(Expr):
    def __init__(self, name: str, namespace, lineno: int, col_offset: int):
        super().__init__(name, namespace, lineno, col_offset)
        self.arg_list = list()
        self.body = OrderedDict()
        self.returns = OrderedDict()

    def add_arg(self, argument):
        self.arg_list.append(argument)

    def add_args(self, arguments: List):
        self.arg_list.extend(arguments)

    def insert_arg(self, argument, position):
        self.arg_list.insert(position, argument)

    def prepend_arg(self, argument):
        self.insert_arg(argument, 0)

    def get_arguments(self):
        return self.arg_list

    def add_statement(self, statement: Expr) -> None:
        self.body.append(statement)

    def add_return(self, statement: Expr) -> None:
        self.returns.append(statement)

    def get_statements(self):
        return self.body

    def gen_code(self):
        fn_name = self.name
        args = ', '.join(arg.gen_code() for arg in self.arg_list)
        new_line = '\n'
        body = f"block({f',{new_line}'.join(b.gen_code() for b in self.body)})"
        return f"define({fn_name, args, body})"

    def __eq__(self, o: Function) -> bool:
        self_ = (self.name, self.namespace, self.loc, self.arg_list)
        other_ = (o.name, o.namespace, o.loc, o.args_list)
        return self_ == other_


class FunctionCall(Function):
    def __init__(self, name: str, args, namespace, lineno: int,
                 col_offset: int):
        super().__init__(name, namespace, lineno, col_offset)
        self.args = args


class Statement:
    def __init__(self, targets: List, value: Any) -> None:
        self.targets = targets
        self.value = value

    def gen_code(self):
        return


class IRBuilder(ast.NodeTransformer):
    def __init__(self, fn: FunctionType) -> None:
        self.fn = fn

        py_src = inspect.getsource(fn)
        py_ast = ast.parse(py_src)
        # self.symbol_table = TaskTable(self.task)
        self.src = self.build(py_ast)

    @abstractmethod
    def build(self, node):
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    def get_src(self):
        return str(self.src)


class IR(IRBuilder):
    def build(self, node):
        self.target = super().visit(node)

    def visit_arg(self, node: ast.arg) -> Any:
        namespace = Namespace.get_namespace()
        arg_name = node.arg
        return PhyArg(arg_name, namespace, node.col_offset, node.col_offset)

    def visit_arguments(self, node: ast.arguments) -> Any:
        args = [self.visit(n) for n in node.args]
        setattr(node, 'ir', args)
        return node

    def visit_Assign(self, node: ast.Assign) -> Any:
        targets = [self.visit(target) for target in node.targets]
        value = self.visit(node.value)
        setattr(node, 'ir', Statement(targets, value))
        return node

    def visit_Call(self, node: ast.Call) -> Any:
        fn_name = self.visit(node.func)
        fn_args = [self.visit(arg) for arg in node.args]
        namespace = Namespace.get_namespace()
        return FunctionCall(fn_name, fn_args, namespace, node.lineno,
                            node.col_offset)

    def visit_Compare(self, node: ast.Compare) -> Any:
        ops_ = {
            ast.Eq: '__eq',
            ast.NotEq: '__ne',
            ast.Lt: '__lt',
            ast.LtE: '__le',
            ast.Gt: '__gt',
            ast.GtE: '__ge',
            ast.Is: NotImplementedError(f"Phylanx: {ast.Is}"),
            ast.IsNot: NotImplementedError(f"Phylanx: {ast.IsNot}"),
            ast.In: NotImplementedError(f"Phylanx: {ast.In}"),
            ast.NotIn: NotImplementedError(f"Phylanx: {ast.NotIn}")
        }
        left = self.visit(node.left)
        comparators = [self.visit(c) for c in node.comparators]
        ops = [ops_[type(op)] for op in node.ops]
        for op in ops:
            if not isinstance(op, str):
                raise op

        # TODO
        if len(ops) > 1:
            raise NotImplementedError(
                "Multi-target assignment is not supported")

        return PhyCompare(left, ops[0], comparators[0])

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Expr(self, node: ast.Expr) -> Any:
        return self.visit(node.value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        fn_name = node.name

        with Namespace(fn_name) as ns:
            fn = Function(fn_name, ns, node.lineno, node.col_offset)
            args = self.visit(node.args)
            for arg in args.ir:
                fn.add_arg(arg)

            for stmnt in node.body:
                statement = self.visit(stmnt)
                fn.add_statement(statement)
                if isinstance(node.body, ast.Return):
                    fn.add_return(statement)

        setattr(node, 'ir', fn)

        return node

    def visit_If(self, node: ast.If) -> Any:
        predicate = self.visit(node.test)
        body = [self.visit(b) for b in node.body]
        orelse = [self.visit(o) for o in node.orelse]

        return PhyIf(predicate, body, orelse)

    def visit_Module(self, node: ast.Module) -> Any:
        file_name = inspect.getfile(self.fn).split('/')[-1].split('.')[0]
        module_name = "__phy_task+" + file_name

        with Namespace(module_name):
            setattr(node, 'ir', self.visit(node.body[0]))
            return node

    def visit_Name(self, node: ast.Name) -> Any:
        ir = Variable(node.id, Namespace.get_namespace(), node.lineno,
                      node.col_offset)
        setattr(node, 'ir', ir)
        return node

    def visit_Return(self, node: ast.Return) -> Any:
        return self.visit(node.value)

    def __str__(self) -> str:
        return self.target.gen_code()


class Profile:
    def __init__(self) -> None:
        self.called = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.called += 1


class Task:
    def __init__(self, fn: FunctionType) -> None:

        self.fn = fn
        self.name = fn.__name__
        self.dtype = None
        self.id = self.fn.__hash__()

        self.profile = Profile()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.profile()
        return self.fn(*args, **kwargs)

    def __hash__(self) -> int:
        return self.id


class Transformer(ABC):
    def __init__(self, fn: Union[FunctionType, Task]) -> None:
        if isinstance(fn, FunctionType):
            self.task: Task = Task(fn)
        elif isinstance(fn, Task):
            self.task: Task = fn
        else:
            raise TypeError


class Polyhedral:
    def __init__(self, fn: Union[FunctionType, Task]) -> None:
        self.ir: IR = IR(fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.pytiramisu(*args, **kwargs)


buf0 = ndarray(10, dtype=int)

@Polyhedral
def function0(buf0: ndarray):
    for i in range(10):
        buf0[i] = 3 + 4
    return buf0


# print(function0(buf0))