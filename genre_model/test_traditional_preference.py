import unittest
import numpy as np
from traditional_preference_experiment import kernel_score

class KernelChecks(unittest.TestCase):
    def test_density_is_invariant_to_global_weight_scale(self):
        similarity=np.array([[.8,.4,-np.inf],[.3,.7,.2]])
        weights=np.array([1.,2.,3.])
        np.testing.assert_allclose(kernel_score(similarity,weights,.1),kernel_score(similarity,weights*10,.1))

    def test_masked_exact_match_has_no_contribution(self):
        score=kernel_score(np.array([[-np.inf,.5]]),np.array([100.,1.]),.1)
        np.testing.assert_allclose(score,[-5.])

    def test_topk_support_keeps_behavior_strength(self):
        sim=np.array([[.8,.4,.1]])
        a=kernel_score(sim,np.ones(3),.1,1)
        b=kernel_score(sim,np.ones(3)*3,.1,1)
        np.testing.assert_allclose(b-a,np.log(3))

if __name__=='__main__':unittest.main()
