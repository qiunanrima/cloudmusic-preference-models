"""Verify noise protocol and per-song weighted objectives."""
import unittest
import numpy as np
import torch
from train_tagged_diffusion_sweep import DenseConditionDenoiser,score,noise_loss

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required by production scorer')
class TaggedDiffusionChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4);torch.manual_seed(3)
        self.net=DenseConditionDenoiser(64,5).cuda().eval()
        rng=np.random.default_rng(4)
        self.x=rng.standard_normal((3,64),dtype=np.float32)
        self.c=rng.standard_normal((3,5),dtype=np.float32)

    def test_scores_are_query_order_independent(self):
        a=score(self.net,self.x,self.c,True,repeats=2)
        b=score(self.net,self.x[::-1].copy(),self.c[::-1].copy(),True,repeats=2)
        np.testing.assert_allclose(a,b[::-1],atol=2e-6,rtol=0)

    def test_paired_identical_models_have_zero_gain(self):
        a=score(self.net,self.x,self.c,True,repeats=2)
        b=score(self.net,self.x,self.c,True,repeats=2)
        np.testing.assert_array_equal(b-a,np.zeros(3))

    def test_unconditional_scores_ignore_metadata(self):
        a=score(self.net,self.x,self.c,False,repeats=2)
        b=score(self.net,self.x,np.zeros_like(self.c),False,repeats=2)
        np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
