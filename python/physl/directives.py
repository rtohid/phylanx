from __future__ import absolute_import
from __future__ import annotations

from physl.phylanx.ast.transducer import Phylanx

__license__ = """ 
Copyright (c) 2021 R. Tohid (@rtohid)

Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
"""

from typing import Union, Any
from types import FunctionType

from physl.phyfleaux.task import Task
# from physl.phyfleaux.analysis.polyhedral import Polytope


def task(__task_arg, **kwargs):

    if isinstance(__task_arg, FunctionType):
        return Task(__task_arg)
    elif isinstance(__task_arg, Task):
        return __task_arg
    else:
        raise TypeError(f"Function {task} expects {FunctionType} or {Task}")


# def polyhedral(fn: Union[Task, FunctionType]) -> Polytope:
#     """Attempts to detect SCoPs and apply polyhedral transformations.

#     :arg fn: python function.

#     Directs Phyfleaux to apply polyhedral transformations on affine iteration
#     spaces in :func:`fn`.

#     reads:
#     -----
#     https://polyhedral.info/

#     https://en.wikipedia.org/wiki/Polytope_model
#     https://en.wikipedia.org/wiki/Affine_space
#     """

#     task_ = task(fn)

#     return Polytope(task_)
