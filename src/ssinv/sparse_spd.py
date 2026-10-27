"""Reusable sparse factorization for the incremental mixed-field tangent solve."""
from __future__ import annotations

import ctypes
from contextlib import contextmanager

import numpy as np
from scipy.sparse import triu
from threadpoolctl import threadpool_limits


class _Factor:
    def __init__(self, solver, upper):
        self.solver, self.upper = solver, upper

    def solve(self, rhs):
        solution = self.solver.solve(self.upper, np.asarray(rhs, dtype=np.float64))
        if not np.isfinite(solution).all():
            raise FloatingPointError('Nonfinite PARDISO solution')
        return solution


@contextmanager
def factor_spd(matrix, threads=2):
    """Factor one positive-definite system and release it after all right sides."""
    from pypardiso import PyPardisoSolver

    upper = triu(matrix, format='csr')
    if upper.nnz >= np.iinfo(np.int32).max or upper.shape[0] >= np.iinfo(np.int32).max:
        raise ValueError('PARDISO requires 32-bit CSR indices')
    solver = PyPardisoSolver(mtype=2)
    initialize = solver.libmkl.pardisoinit
    initialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32)]
    initialize.restype = None
    initialize(solver.pt.ctypes.data, ctypes.byref(ctypes.c_int32(2)),
               solver.iparm.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    for index, value in ((1, 1), (2, 2), (8, 2), (18, -1), (27, 1)):
        solver.set_iparm(index, value)
    try:
        with threadpool_limits(limits=threads):
            solver.factorize(upper)
            yield _Factor(solver, upper)
    finally:
        solver.free_memory(everything=True)
