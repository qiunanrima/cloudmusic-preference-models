"""Leakage and autoregressive causality checks for conditional AR."""
import copy
import unittest
import numpy as np
import torch
from train_tagged_ar_experiment import semantic_estimates, TaggedAR

class TaggedARChecks(unittest.TestCase):
    def test_heldout_labels_never_enter_conditions(self):
        raw=np.eye(6,dtype=np.float32)
        rows=[{'文件':str(i),'风格':'rock' if i%2 else 'folk','BPM':100+i} for i in range(6)]
        train=np.array([0,1,2,3]);ids=np.arange(6)
        a,_=semantic_estimates(raw,train,rows,raw,ids)
        altered=copy.deepcopy(rows)
        for i in (4,5):altered[i].update(风格='SECRET_HELDOUT',BPM=9999)
        b,_=semantic_estimates(raw,train,altered,raw,ids)
        np.testing.assert_array_equal(a,b)

    def test_self_label_excluded_even_in_global_fallback(self):
        raw=np.eye(4,dtype=np.float32)
        rows=[{'文件':'0','风格':'solo'},{'文件':'1'},{'文件':'2'},{'文件':'3'}]
        c,_=semantic_estimates(raw,np.array([0,1,2]),rows,raw,np.arange(4))
        self.assertEqual(float(c[0,0]),0.)

    def test_first_prediction_has_no_future_audio_dependency(self):
        torch.manual_seed(1)
        net=TaggedAR(5).eval()
        x=torch.randn(2,192);c=torch.randn(2,5)
        # Modify only the final target chunk. Earlier conditional log densities
        # must be independent of it; capture their mixture parameters.
        captured=[]
        hook=net.output.register_forward_hook(lambda _m,_i,o:captured.append(o.detach().clone()))
        with torch.no_grad():
            net.log_prob(x,c)
            x[:,-16:]+=100
            net.log_prob(x,c)
        hook.remove()
        torch.testing.assert_close(captured[0],captured[1])

if __name__=='__main__':unittest.main()
