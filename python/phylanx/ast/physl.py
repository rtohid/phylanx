# Copyright (c) 2017 Hartmut Kaiser
# Copyright (c) 2018 Steven R. Brandt
# Copyright (c) 2018 R. Tohid
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import re
import ast
import inspect
import phylanx.execution_tree
from phylanx import compiler_state

def primitive_name(method_name):
    """Given a method_name, returns the corresponding Phylanx primitive.

    This primarily used for mapping NumPy methods to Phylanx primitives, but there are
    also other functions in python that would map to primitives with different name in
    Phylanx, e.g., `print` is mapped to `cout`.
    """

    methods = {
        "det": "determinant",
        "diagonal": "diag",
        "print": "cout",
        "sqrt": "square_root",
    }

    primitive_name = methods.get(method_name)
    if primitive_name == None: primitive_name = method_name
    return primitive_name

def physl_fmt(src, tag=4):
    """Pretty print PhySL source code"""

    # Remove line number info
    src = re.sub(r'\$\d+', '', src)

    # The regex below matches one of the following three
    # things in order of priority:
    # 1: a quoted string, with possible \" or \\ embedded
    # 2: a set of balanced parenthesis
    # 3: a single character
    pat = re.compile(r'"(?:\\.|[^"\\])*"|\([^()]*\)|.')
    indent = 0
    tab = 4
    for s in re.findall(pat, src):
        if s in " \t\r\b\n":
            pass
        elif s == '(':
            print(s)
            indent += 1
            print(" " * indent * tab, end="")
        elif s == ')':
            indent -= 1
            print("", sep="")
            print(" " * indent * tab, end="")
            print(s, end="")
        elif s == ',':
            print(s)
            print(" " * indent * tab, end="")
        else:
            print(s, end="", sep="")
    print("", sep="")


def dump_info(a, depth=0):
    "Print detailed information about an AST"

    nm = a.__class__.__name__
    print("  " * depth, end="")
    iter_children = True
    if nm == "Num":
        if type(a.n) == int:
            print("%s=%d" % (nm, a.n))
        else:
            print("%s=%f" % (nm, a.n))
    elif nm == "Global":
        print("Global:", dir(a))
    elif nm == "Str":
        print("%s='%s'" % (nm, a.s))
    elif nm == "Name":
        print("%s='%s'" % (nm, a.id))
    elif nm == "arg":
        print("%s='%s'" % (nm, a.arg))
    elif nm == "Slice":
        print("Slice:")
        print("  " * depth, end="")
        print("  Upper:")
        if a.upper is not None:
            dump_info(a.upper, depth + 4)
        print("  " * depth, end="")
        print("  Lower:")
        if a.lower is not None:
            dump_info(a.lower, depth + 4)
        print("  " * depth, end="")
        print("  Step:")
        if a.step is not None:
            dump_info(a.step, depth + 4)
    elif nm == "If":
        iter_children = False
        print(nm)
        dump_info(a.test, depth)
        for n in a.body:
            dump_info(n, depth + 1)
        if len(a.orelse) > 0:
            print("  " * depth, end="")
            print("Else")
            for n in a.orelse:
                dump_info(n, depth + 1)
    else:
        print(nm)
    for (f, v) in ast.iter_fields(a):
        if type(f) == str and type(v) == str:
            print("%s:attr[%s]=%s" % ("  " * (depth + 1), f, v))
    if iter_children:
        for n in ast.iter_child_nodes(a):
            dump_info(n, depth + 1)


def print_physl(physl_src):
    print(re.sub(r'\$\d+', '', physl_src))


def get_symbol_info(a, name):
    return '%s$%d$%d' % (name, a.lineno, a.col_offset)


def get_node(node, **kwargs):
    if node is None:
        return None
    args = [arg for arg in ast.iter_child_nodes(node)]
    params = {"name": None, "num": None}
    for k in kwargs:
        if k not in params:
            raise Exception("Invalid argument '%s'" % k)
        else:
            params[k] = kwargs[k]
    name = params["name"]
    num = params["num"]
    if name is not None:
        for i in range(len(args)):
            a = args[i]
            nm = a.__class__.__name__
            if nm == name:
                if num is None or num == i:
                    return a
    elif num is not None:
        if num < len(args):
            return args[num]

    # if we can't find what we're looking for...
    return None


def remove_line(a):
    return re.sub(r'\$.*', '', a)


def convert_to_phylanx_type(v):
    t = type(v)
    try:
        import numpy
        if t == numpy.ndarray:
            return phylanx.execution_tree.var(v)
    except NotImplementedError:
        pass
    return v


class PhySL:
    """Python AST to PhySL Transdcer."""

    compiler_state = None

    def __init__(self, func, tree, kwargs):
        self.defs = set()
        self.wrapped_function = func
        self.fglobals = kwargs['fglobals']

        # Add all global variables to 'already defined' list.
        for arg in tree.body[0].args.args:
            self.defs.add(arg)


        self.ir = tuple(map(self.apply_rule, tree.body))
        self.__src__ = self.generate_physl(self.ir[0])

        if kwargs.get("debug"):
            physl_fmt(self.__src__)
            print(end="", flush="")

        if "compiler_state" in kwargs:
            PhySL.compiler_state = kwargs['compiler_state']
        elif PhySL.compiler_state is None:
            PhySL.compiler_state = compiler_state()

        phylanx.execution_tree.compile(self.__src__, PhySL.compiler_state)

    def generate_physl(self, ir):
        if len(ir) == 2 and isinstance(ir[0], str) and isinstance(ir[1], tuple):
            return ir[0] + '(' + ', '.join([self.generate_physl(i) for i in ir[1]]) + ')'
        elif isinstance(ir, list):
            return ', '.join([self.generate_physl(i) for i in ir])
        else:
            return ir

    #def generate_physl(self, ir):
    #    src = ''
    #    if isinstance(ir, str):
    #        src = ir
    #    elif isinstance(ir, list):
    #        src += ' '.join(map(str, [self.generate_physl(e) for e in ir]))
    #    elif isinstance(ir, tuple):
    #        src += '('
    #        src += ', '.join(map(str, [self.generate_physl(e) for e in ir]))
    #        src += ')'
    #    return src

    def call(self, args):
        nargs = tuple(convert_to_phylanx_type(a) for a in args)
        fname = self.wrapped_function.__name__
        return phylanx.execution_tree.eval(fname, PhySL.compiler_state, *nargs)

# #############################################################################
# Transducer rules

    def _Add(self, node):
        """Leaf node returning raw string of the 'add operation'."""

        return '__add'

    def _And(self, node):
        """Leaf node returning raw string of the 'and operation'."""

        return '__and'

    def _arg(self, node):
        """class arg(arg, annotation)

        A single argument in a list. In Python 3.

        `arg` is a raw string of the argument name.
        `annotation` is its annotation, such as a `Str` or `Name` node.

        TODO: 
            add support for `annotation` which is ignored at this time. Maybe
            we can use this to type information!
        """

        return node.arg

    def _arguments(self, node):
        """class arguments(args, vararg, kwonlyargs, kwarg, defaults, kw_defaults)

        The arguments for a function. In Python 3.
        `args` and `kwonlyargs` are lists of arg nodes.
        `vararg` and `kwarg` are single arg nodes, referring to the *args,
        **kwargs parameters.
        `defaults` is a list of default values for arguments that can be passed
        positionally. If there are fewer defaults, they correspond to the last
        n arguments.
        `kw_defaults` is a list of default values for keyword-only arguments. If
        one is None, the corresponding argument is required.
        """
        if node.vararg or node.kwarg:
            raise (Exception("Phylanx does not support *args and **kwargs"))
        args = list(map(self.apply_rule, node.args))
        args = ', '.join(map(str, args))
        return args

    def _Assign(self, node):
        """class Assign(targets, value)

        `targets` is a list of nodes which are assigned a value.
        `value` is a single node which gets assigned to `targets`.
        
        TODO:
            Add support for multi-target (a,b, ... = iterable) and chain (a = b = ... )
            assignments.
        """

        if len(node.targets) > 1:
            raise Exception("Phylanx does not support chain assignments.")
        if isinstance(node.targets[0], ast.Tuple):
            raise Exception(
                "Phylanx does not support multi-target assignments.")

        symbol = self.apply_rule(node.targets[0])
        if isinstance(symbol, str):
            symbol_name = re.sub(r'\$\d+', '', symbol)
            if symbol_name in self.defs:
                op = get_symbol_info(node.targets[0], "store")
            else:
                op = get_symbol_info(node.targets[0], "define")
                # TODO:
                # For now `self.defs` is a set which includes names of symbols with no
                # extra information about them. We may want to make it a disctionary
                # with the symbol names as keys and list of symbol_infos to keep track
                # of the symbol.
                self.defs.add(symbol_name)
        else:
            op = get_symbol_info(node.targets[0], "set")



        target = self.apply_rule(node.targets[0])
        value = self.apply_rule(node.value)

        return [op, (target, value)]

    def _Attribute(self, node):
        """class Attribute(value, attr, ctx)

        Note:
            In a sense value (`node.value`) specifies the name space of it's
            attribute (`node.attr`). At this point Phylanx does not (exactly)
            support such concept. For now, we quietly ignore all the values
            and only consider the attribute.
        """

        # Ignore the value for now. See the note above in the method's docstring.
        #value = self.apply_rule(node.value)

        return primitive_name(node.attr)

    def _AugAssign(self, node):
        """class AugAssign(target, op, value)"""

        symbol = get_symbol_info(node, 'store')
        op = self.apply_rule(node.op)
        target = self.apply_rule(node.target)
        value = self.apply_rule(node.value)

        return [symbol, (target, (op, (target, value)))]

    def _BinOp(self, node):
        """class BinOp(left, op, right)"""

        op = self.apply_rule(node.op)
        left = self.apply_rule(node.left)
        right = self.apply_rule(node.right)

        return [op, (left, right)]

    def _BoolOp(self, node):
        """class BoolOp(left, op, right)"""

        op = self.apply_rule(node.op)
        values = list(map(self.apply_rule, node.values))

        return [op, (values, )]

    def _Call(self, node):
        """class Call(func, args, keywords, starargs, kwargs)
        
        TODO(?):
            Add support for keywords, starargs, and kwargs
        """

        symbol = self.apply_rule(node.func)
        args = tuple(self.apply_rule(arg) for arg in node.args)
        return [symbol, args]

    def _Compare(self, node):
        """class Compare(left, ops, comparators)
        
        A comparison of two or more values. 
        `left` is the first value in the comparison
        `ops` is the list of operators
        `comparators` is the list of values after the first (`left`).
        """

        if (len(node.ops) == 1):
            left = self.apply_rule(node.left)
            op = self.apply_rule(node.ops[0])
            right = self.apply_rule(node.comparators[0])

            return [op, (left, right)]
        else:
            # if we're dealing with more than one comparison, we canonicalize the
            # comparisons in to the form of chained logical ands. e.g., a < b < c
            # becomes: ([and_operation ((less b, c), (less a, b))])
            # TODO: Make sure to respect Python operator precedence.
            comparison = []
            for i in range(len(node.ops)):
                op = self.apply_rule(node.ops[-i])
                left = self.apply_rule(node.comparators[-i - 1])
                right = self.apply_rule(node.comparators[-i])
                if comparison:
                    comparison = ['__and', (*comparison, (op, (left, right)))]
                else:
                    comparison = [*comparison, (op, (left, right))]

            op = self.apply_rule(node.ops[0])
            left = self.apply_rule(node.left)
            right = self.apply_rule(node.comparators[0])
            if comparison:
                comparison = ['__and', (*comparison, (op, (left, right)))]
            else:
                comparison = [op, (left, right)]

            return comparison

    def _Div(self, node):
        """Leaf node returning raw string of the 'division operation'."""
        return '__div'

    def _Eq(self, node):
        """Leaf node returning raw string of the 'equality operation'."""
        return '__eq'

    def _Expr(self, node):
        """class Expr(value)
        
        `value` holds one of the other rules.
        """

        return self.apply_rule(node.value)

    def _ExtSlice(self, node):
        """class ExtSlice(dims)
        
        Advanced slicing. 
        `dims` holds a list of `Slice` and `Index` nodes.
        """

        slicing = list(map(self.apply_rule, node.dims))

        return slicing

    def _For(self, node):
        """class For(target, iter, body, orelse)

        A for loop.
        `target` holds the variable(s) the loop assigns to, as a single Name, Tuple or
        List node.
        `iter` holds the item to be looped over, again as a single node.
        `body` contain lists of nodes to execute.
        `orelse` same as `body`, however, those in orelse are executed if the loop
        finishes normally, rather than via a break statement.
        """

        # this lookup table helps us to choose the right mapping function based on the 
        # type of the iteration space (make_list, range, or prange).
        mapping_function = {
            'make_list': 'map',
            'range': 'map',
            'prange': 'parallel_map'
        }

        target = self.apply_rule(node.target)
        iteration_space = self.apply_rule(node.iter)

        # extract the type of the iteration space- used as the lookup key in 
        # `mapping_function` dictionary above.
        symbol_name = mapping_function[iteration_space[0].split('$', 1)[0]]
        symbol = get_symbol_info(node, symbol_name)

        # replace keyword `prange` to `range` for compatibility with Phylanx.
        iteration_space[0] = iteration_space[0].replace('prange', 'range')
        body = self.block(node.body)
        #orelse = self.block(node.orelse)
        return [symbol, (['lambda', (target, body)], iteration_space)]
        #return [symbol, (target, iteration_space, body, orelse)]

    def _FunctionDef(self, node):
        """class FunctionDef(name, args, body, decorator_list, returns)
        
        `name` is a raw string of the function name.
        `args` is a arguments node.
        `body` is the list of nodes inside the function.
        `decorator_list` is the list of decorators to be applied, stored
            outermost first (i.e. the first in the list will be applied last).
        `returns` is the return annotation (Python 3 only).

        Notes:
            We ignore decorator_list and returns.
        """

        op = get_symbol_info(node, 'define')
        symbol = get_symbol_info(node, node.name)
        args = self.apply_rule(node.args)
        body = self.block(node.body)

        if (args):
            return [op, (symbol, args, body)]
        else:
            return [op, (symbol, body)]

    def _Gt(self, node):
        """Leaf node returning raw string of the 'greater than operation'."""
        return '__gt'

    def _GtE(self, node):
        """Leaf node returning raw string of the 'greater than or equal operation'."""
        return '__ge'

    def _If(self, node):
        """class IfExp(test, body, orelse)
       
       `test` holds a single node, such as a Compare node.
       `body` and `orelse` each hold a list of nodes.
       """

        symbol = '%s' % get_symbol_info(node, 'if')
        test = self.apply_rule(node.test)
        body = self.block(node.body)
        orelse = self.block(node.orelse)

        return [symbol, (test, body, orelse)]

    def _In(self, node):
        raise Exception("`In` operator is not defined in Phylanx")

    def _Index(self, node):
        """class Index(value)

        Simple subscripting with a single value.
        """
        return self.apply_rule(node.value)

    def _Is(self, node):
        raise Exception("`Is` operator is not defined in Phylanx")

    def _IsNot(self, node):
        raise Exception("`IsNot` operator is not defined in Phylanx")

    def _Lambda(self, node):
        """class Lambda(args, body)

        `body` is a single node.
        """
        symbol = get_symbol_info(node, 'lambda')
        args = self.apply_rule(node.args)
        body = self.block(node.body)

        return [symbol, (args, body)]

    def _List(self, node):
        """class List(elts, ctx)"""

        op = get_symbol_info(node, 'make_list')
        content = tuple(map(self.apply_rule, node.elts))
        return [op, (*content, )]

    def _Lt(self, node):
        """Leaf node returning raw string of the 'less than operation'."""
        return '__lt'

    def _LtE(self, node):
        """Leaf node returning raw string of the 'less than or equal operation'."""
        return '__le'

    def _Module(self, node):
        module = list(self.apply_rule(m) for m in node.body)

        return module

    def _Mult(self, node):
        """Leaf node returning raw string of the 'multiplication operation'."""

        return '__mul'

    def _Name(self, node):
        """class Name(id, ctx)

        A variable name.
        `id` holds the name as a string.
        `ctx` is one of `Load`, `Store`, `Del`.
        """

        symbol = get_symbol_info(node, primitive_name(node.id))
        symbol_name = re.sub(r'\$\d+', '', symbol)
        return symbol

    def _NameConstant(self, node):
        name_constants = {None: 'nil', False: 'false', True: 'true'}
        return name_constants[node.value] + get_symbol_info(node, '')

    def _Not(self, node):
        """Leaf node returning raw string of the 'not operation'."""
        return '__not'

    def _NotEq(self, node):
        """Leaf node returning raw string of the 'not equal operation'."""
        return '__ne'

    def _NotIn(self, node):
        raise Exception("`NotIn` operator is not defined in Phylanx")

    def _Num(self, node):
        """class Name(n)"""

        return str(node.n)

    def _Or(self, node):
        """Leaf node returning raw string of the 'or operation'."""
        return '__or'

    def _Pow(self, node):
        """Leaf node returning raw string of the 'power operation'."""
        return 'power'

    def _Return(self, node):
        """class Return(value)

        TODO:
            implement return-from primitive (see section Function Return Values on
            https://goo.gl/wT6X4P). At this time Phylanx only supports returns from the
            end of the function!
        """

        return self.apply_rule(node.value)

    def _Slice(self, node):
        """class Slice(lower, upper, step)"""

        symbol = 'make_list'

        lower = self.apply_rule(node.lower)
        if lower is None: lower = str(0)

        upper = self.apply_rule(node.upper)
        if upper is None: upper = 'nil'

        step = self.apply_rule(node.step)
        if step is None: step = str(1)

        slice_ = self.generate_physl([symbol, (lower, upper, step)])

        return slice_

    def _Str(self, node):
        """class Str(s)"""

        return '"' + node.s + '"'

    def _Sub(self, node):
        return '__sub'

    def _Subscript(self, node):
        """class Subscript(value, slice, ctx)"""

        op = '%s' % get_symbol_info(node, 'slice')
        value = self.apply_rule(node.value)
        slice_ = self.apply_rule(node.slice)

        return [op, (value, [slice_])]

    def _Tuple(self, node):
        """class Tuple(elts, ctx)"""

        expr = tuple(map(self.apply_rule, node.elts))
        return expr

    def _UAdd(self, node):
        """

        TODO:
        Make sure we do not have the equivalent of this in PhySL. Otherwise, add support.
        """

        raise Exception("`UAdd` operation is not defined in PhySL.")

    def _UnaryOp(self, node):
        """class UnaryOp(op, operand)"""

        op = self.apply_rule(node.op)
        operand = self.apply_rule(node.operand)

        return [op, (operand,)]

    def _USub(self, node):
        """Leaf node returning raw string of the 'negative operation'."""
        return '__minus'

    def _While(self, node):
        """class While(test, body, orelse)

        TODO:
        Figure out what `orelse` attribute may contain. From my experience this is always
        an empty list!
        """

        symbol = '%s' % get_symbol_info(node, 'while')
        test = self.block(node.test)
        body = self.block(node.body)
        return [symbol, (test, body)]

# #############################################################################

    def block(self, node):
        """Returns a map representation of a PhySL bolck."""

        if isinstance(node, list):
            block = tuple(map(self.apply_rule, node))
        else:
            block = (self.apply_rule(node), )

        return ['block', block]

    def apply_rule(self, node):
        if node is not None:
            node_name = node.__class__.__name__
            return eval('self._%s' % node_name)(node)
