import unittest, sys

import theano.tensor as TT

FAS = TT.FullAdvancedSubtensor

class T_idxlist(unittest.TestCase):
    def test0(self):
        f = FAS([])
        assert f.idx_tuple == ()
        assert f.view_map == {0:[0]}


    def test_ellipsis(self):

        f = FAS([Ellipsis, 3])
        assert f.idx_tuple == (Ellipsis, 3,)
        assert f.view_map == {0:[0]}


    def test1_int(self):
        f = FAS([0])
        assert f.idx_tuple == (0,)
        assert f.view_map == {0:[0]}

        f = FAS([-1])
        assert f.idx_tuple == (-1,)
        assert f.view_map == {0:[0]}

        assert f.n_in == 1

    def test1_var(self):
        idx = TT.iscalar()
        f = FAS([idx])
        assert f.idx_tuple == (TT.iscalar,)
        assert f.view_map == {0:[0]}
        assert f.n_in == 2

        f = FAS([-idx])
        assert f.idx_tuple == (TT.iscalar,)
        assert f.view_map == {0:[0]}
        assert f.n_in == 2

    def test1_basic_slice(self):
    
        idx = TT.iscalar()
        f = FAS([slice(0,idx,1)])
        assert f.idx_tuple == (slice(0,TT.iscalar,1),)
        assert f.view_map == {0:[0]}
        assert f.n_in == 2

        f = FAS([slice(-idx,idx,None)])
        assert f.idx_tuple == (slice(TT.iscalar, TT.iscalar, 1),)
        assert f.view_map == {0:[0]}
        assert f.n_in == 3

        f = FAS([slice(None, None, None)])
        assert f.idx_tuple == (slice(0, sys.maxint, 1),)
        assert f.view_map == {0:[0]}
        assert f.n_in == 1



    def test1_ndarray(self):

        i0 = TT.iscalar()
        i1 = TT.lvector()
        i2 = TT.bmatrix()

        
        f = FAS([i0])
        assert f.idx_tuple == (i0.type,)
        assert f.view_map == {0:[0]}
        assert f.n_in == 2

        f = FAS([i1])
        assert f.idx_tuple == (i1.type,)
        assert f.view_map == {}
        assert f.n_in == 2

        f = FAS([i2])
        assert f.idx_tuple == (i2.type,)
        assert f.view_map == {}
        assert f.n_in == 2


    def test3_ndarray(self):

        i0 = TT.iscalar()
        i1 = TT.lvector()
        i2 = TT.bmatrix()
        
        f = FAS([i1, slice(None, i0, -1), i2])
        assert f.n_in == 4
        assert f.idx_tuple == (i1.type, slice(0, i0.type, -1), i2.type,)
        assert f.view_map == {}

    def test_illegal_things(self):
        i0 = TT.iscalar()
        i1 = TT.lvector()
        i2 = TT.bmatrix()
        self.failUnlessRaises(TypeError, FAS, [i1, slice(None, i2, -1), i0])
        self.failUnlessRaises(TypeError, FAS, [i1, slice(None, None, i2), i0])
        self.failUnlessRaises(TypeError, FAS, [i1, slice(i2, None, -1), i0])

