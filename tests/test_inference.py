import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

spec = importlib.util.spec_from_file_location('inference',Path(__file__).resolve().parents[1]/'src/infer_fold555.py')
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)


def meta(frames):
    return pd.DataFrame({'video_id':1,'agent_id':'mouse1','target_id':'self','video_frame':frames})


class InferenceTests(unittest.TestCase):
    def test_last_frame_and_single_constant_interval_are_preserved(self):
        result = infer.intervals(pd.DataFrame({'rear':[.9,.9,.9]}),meta([100,101,102]),{'rear':.5})
        self.assertEqual(result[['start_frame','stop_frame']].values.tolist(),[[100,103]])

    def test_frame_gaps_and_thresholds(self):
        result = infer.intervals(pd.DataFrame({'rear':[.9,.9,.9,.1,.9]}),meta([1,2,5,6,7]),{'rear':.5})
        self.assertEqual(result[['start_frame','stop_frame']].values.tolist(),[[1,3],[5,6],[7,8]])

    def test_argmax_uses_winning_action_threshold(self):
        result = infer.intervals(pd.DataFrame({'rear':[.8,.1],'freeze':[.7,.9]}),meta([5,6]),{'rear':.85,'freeze':.5})
        self.assertEqual(result.action.tolist(),['freeze'])
        self.assertEqual(result[['start_frame','stop_frame']].values.tolist(),[[6,7]])

    def test_ensemble_matches_average_and_rejects_reordered_features(self):
        x = pd.DataFrame({'a name':np.arange(30),'b':np.arange(30)%3})
        models = [LGBMClassifier(n_estimators=3,min_child_samples=2,n_jobs=1,verbosity=-1,random_state=i).fit(x,np.arange(30)%2) for i in range(5)]
        bundle = SimpleNamespace(is_fitted=True,estimators=models)
        expected = np.mean([m.predict_proba(x)[:,1] for m in models],axis=0)
        np.testing.assert_array_equal(infer.predict_bundle(bundle,x,1),expected)
        with self.assertRaisesRegex(ValueError,'ordering'):
            infer.predict_bundle(bundle,x[['b','a name']],1)

    def test_invalid_probabilities_rejected(self):
        for p in (np.array([np.nan]),np.array([1.1]),np.array([-.1])):
            with self.assertRaises(ValueError):
                infer.check_probabilities(p)

    def test_overlapping_or_unlabelled_submission_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'test_tracking/Lab').mkdir(parents=True)
            pd.DataFrame({'video_frame':range(10)}).to_parquet(root/'test_tracking/Lab/1.parquet')
            dataset = pd.DataFrame([dict(video_id=1,lab_id='Lab',behaviors_labeled='["mouse1,self,rear"]')])
            valid = pd.DataFrame([(1,'mouse1','self','rear',0,3),(1,'mouse1','self','rear',4,10)],columns=infer.COLUMNS)
            infer.validate_submission(valid,dataset,root)
            invalid = valid.copy();invalid.loc[1,'start_frame']=2
            with self.assertRaisesRegex(ValueError,'Overlapping'):
                infer.validate_submission(invalid,dataset,root)
            invalid = valid.copy();invalid.loc[1,'action']='invented'
            with self.assertRaisesRegex(ValueError,'unlabelled'):
                infer.validate_submission(invalid,dataset,root)


if __name__ == '__main__':
    unittest.main()
