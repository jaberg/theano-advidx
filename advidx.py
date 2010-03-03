""" 
"""

__docformat__ = "restructuredtext en"

import sys # for sys.maxint
from copy import copy
import traceback #for overriding Op.__call__
if sys.version_info >= (2,5):
  import functools

import numpy, theano
from theano.gof import Op

class FullAdvancedSubtensor(Op):
    """
    self.idxlist is a list
    len <= x.ndim

    self.idxlist[i] is one of
    - a TensorType with integer dtype
    - an integer
    - an ellipsis object
    - a slice with start/stop/step one of
      - a TensorType with int dtype and ndim=0
      - an integer

    """
    def __init__(self, getitem_args):
        """
        Extract types and constants from the getitem_args.
        
        `getitem_args` may include variables, but these are not saved in the Op.
        """
        self.view_map = {0:[0]}
        self.n_in = 1
        self.n_out = 1
        if getitem_args:
            getitem_vars, idx_tuple = zip(
                    *[self.extract_idxlist(i, view_map=self.view_map)
                        for i in getitem_args])
            for giv in getitem_vars:
                self.n_in += len(giv)
        else:
            idx_tuple = ()
        self.idx_tuple = idx_tuple

    @staticmethod
    def extract_idxlist(entry, slice_ok=True, view_map={}):
        """Convert arguments to __getitem__ to the thing you need for the idx_tuple.

        """
        scal_types = [scal.int64, scal.int32, scal.int16, scal.int8]
        tensor_types = [bscalar, iscalar, lscalar]

        if isinstance(entry, gof.Variable):
            variable = [entry]
            entry = entry.type
        else:
            variable = []

        if entry == Ellipsis:
            return [], entry

        if isinstance(entry, TensorType):
            if entry.dtype[:3] not in ('int', 'uin'):
                raise TypeError(entry)
            if not numpy.all(entry.broadcastable):
                # this is a real array, 
                
                # we don't allow arrays in slices
                if not slice_ok: 
                    raise TypeError(entry)

                # real arrays trigger advanced indexing
                # which cannot work in-place
                if 0 in view_map:
                    del view_map[0]
            return variable, entry
        
        if slice_ok and isinstance(entry, slice):

            start = entry.start
            stop = entry.stop
            step = entry.step

            if start is None:
                start_vars = []
                start = 0
            else:
                start_vars, start = FullAdvancedSubtensor.extract_idxlist(
                        start, False, view_map)

            if stop is None:
                stop_vars = []
                stop = sys.maxint
            else:
                stop_vars, stop = FullAdvancedSubtensor.extract_idxlist(
                        stop, False, view_map)

            if step is None:
                step_vars = []
                step = 1
            else:
                step_vars, step = FullAdvancedSubtensor.extract_idxlist(
                        step, False, view_map)

            print 'SSS', start_vars, stop_vars, step_vars
            return (start_vars+stop_vars+step_vars), slice(start,stop,step)

        if isinstance(entry, int):
            return [], entry

        raise TypeError(Subtensor.e_indextype, entry)

    def make_node(self, x, *getitem_vars):
        assert 1+len(getitem_vars) == self.n_in

        odtype = x.type.dtype
        
        # the broadcastable pattern is 

    def __call__(self, x, getitem_args):
        """Override Op.__call__ to interpret getitem_args """
        getitem_vars, idx_tuple = zip(
                self.extract_idxlist(i, view_map=self.view_map)
                    for i in getitem_args)
        node = self.make_node(x, *getitem_vars)
        node.tag.trace = traceback.extract_stack()[:-1]
        return node.outputs[0]

