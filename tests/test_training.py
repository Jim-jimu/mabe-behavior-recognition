"""Synthetic integration tests; no competition data or scores are fabricated."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sklearn.base import clone

import joblib
import numpy as np
import pandas as pd

RETRAIN = Path(__file__).resolve().parents[1] / "src"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, RETRAIN / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_data(root):
    for folder in ('train_tracking', 'train_annotation', 'test_tracking'):
        (root / folder / 'SyntheticLab').mkdir(parents=True)
    parts = ['body_center', 'ear_left', 'ear_right', 'nose', 'tail_base']
    rows = []
    rng = np.random.default_rng(3407)
    for video in range(6):
        rows.append(dict(video_id=video, lab_id='SyntheticLab', mouse1_strain='x', mouse2_strain='x',
                         mouse3_strain=None, mouse4_strain=None, frames_per_second=30,
                         pix_per_cm_approx=10, video_width=640, video_height=480,
                         arena_width_cm=64, arena_height_cm=48, arena_shape='rectangle',
                         arena_type='enclosure', tracking_method='synthetic',
                         body_parts_tracked=json.dumps(parts),
                         behaviors_labeled=json.dumps(['mouse1,self,rear', 'mouse1,mouse2,approach'])))
        offset=1000*video
        values = [(offset+frame, mouse, part, 200+mouse*30+8*np.sin(frame/10+i)+rng.normal(),
                   200+mouse*30+10*np.cos(frame/15+i)+rng.normal())
                  for frame in range(100) for mouse in (1,2) for i,part in enumerate(parts)]
        pd.DataFrame(values, columns=['video_frame','mouse_id','bodypart','x','y']).to_parquet(root/'train_tracking'/'SyntheticLab'/f'{video}.parquet', index=False)
        pd.DataFrame([(1,1,'rear',offset+20,offset+50),(1,2,'approach',offset+40,offset+80)],
                     columns=['agent_id','target_id','action','start_frame','stop_frame']).to_parquet(root/'train_annotation'/'SyntheticLab'/f'{video}.parquet',index=False)
    pd.DataFrame(rows).to_csv(root/'train.csv',index=False)
    pd.DataFrame(rows[:1]).assign(video_id=99).to_csv(root/'test.csv',index=False)
    # Some declared keypoints may be absent from a video's actual tracking file.
    # Later videos then add feature columns; metadata must still remain last.
    first=root/'train_tracking'/'SyntheticLab'/'0.parquet'
    tracking=pd.read_parquet(first)
    tracking[tracking.bodypart!='ear_right'].to_parquet(first,index=False)


def configure(module):
    module.CFG.section_start=0
    module.CFG.threshold_trials=3
    module.CFG.model.set_params(n_estimators=6,min_child_weight=0.01,min_child_samples=2)


def finish(module):
    if module.log_file is not None and not module.log_file.closed:
        module.log_file.close()


class ResumableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='mabe-resume-tests-')
        cls.root=Path(cls.temp.name)
        cls.data=cls.root/'data'
        make_data(cls.data)
        cls.environment=patch.dict(os.environ,{'MABE_DATA_DIR':str(cls.data),
                'MABE_OUTPUT_BASE':str(cls.root/'outputs'),'MABE_CACHE_DIR':str(cls.root/'scratch'),
                'MABE_MODEL_JOBS':'1','MABE_FEATURE_JOBS':'1','MABE_RESUME_DIR':''})
        cls.environment.start()
        cls.new=load_module('mabe_new','train_fold555_disk_safe.py')
        configure(cls.new)
        cls.new.main()
        cls.new_out=Path(cls.new.CFG.output_dir)

    @classmethod
    def tearDownClass(cls):
        finish(cls.new)
        cls.environment.stop()
        cls.temp.cleanup()

    def test_fresh_saves_complete_five_fold_tasks(self):
        self.assertEqual(json.loads((self.new_out/'run_status.json').read_text())['status'],'complete')
        for action in ('rear','approach'):
            folder=self.new_out/'0'/action
            model=joblib.load(next(folder.glob('*_trainer_*.pkl')))
            self.assertEqual(len(model.estimators),5)
            predictions=pd.read_parquet(folder/'oof_predictions.parquet')
            self.assertEqual(set(predictions.fold),set(range(5)))
            self.assertEqual(predictions.groupby('video_id').fold.nunique().max(),1)
            self.assertTrue((folder/'task_complete.json').exists())
            self.assertFalse((folder/'checkpoints').exists())

    def test_completed_tasks_do_not_refit(self):
        for previous in (self.new_out,):
            module=load_module('resume_complete','train_fold555_disk_safe.py')
            configure(module)
            with patch.dict(os.environ,{'MABE_RESUME_DIR':str(previous)}), patch.object(module,'fit_one_fold',side_effect=AssertionError('Completed task must not refit')):
                module.main()
            output=Path(module.CFG.output_dir)
            self.assertEqual(len(module.checkpoint_scores),2)
            for action in ('rear','approach'):
                before=next((previous/'0'/action).glob('*_trainer_*.pkl'))
                after=next((output/'0'/action).glob('*_trainer_*.pkl'))
                self.assertEqual(module.digest_file(before),module.digest_file(after))

    def test_interrupted_action_resumes_completed_fold(self):
        module=load_module('interrupt_model','train_fold555_disk_safe.py')
        configure(module)
        real=module.fit_one_fold
        calls=[]
        def fail_after_one(*args,**kwargs):
            fold=args[5]
            if fold==1:
                raise RuntimeError('simulated interruption')
            calls.append(fold)
            return real(*args,**kwargs)
        try:
            with patch.object(module,'fit_one_fold',side_effect=fail_after_one):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    module.main()
        finally:
            finish(module)
        interrupted=Path(module.CFG.output_dir)
        self.assertTrue((interrupted/'0/rear/checkpoints/fold_0.json').is_file())
        resumed=load_module('resume_fold','train_fold555_disk_safe.py')
        configure(resumed)
        original=resumed.fit_one_fold
        refits=[]
        def watch(*args,**kwargs):
            refits.append((args[1],args[5]))
            return original(*args,**kwargs)
        with patch.dict(os.environ,{'MABE_RESUME_DIR':str(interrupted)}),patch.object(resumed,'fit_one_fold',side_effect=watch):
            resumed.main()
        self.assertNotIn(('rear',0),refits)
        self.assertEqual(len(refits),9)
        for action in ('rear','approach'):
            expected=joblib.load(self.new_out/'0'/action/'oof_pred_probs.pkl')
            actual=joblib.load(Path(resumed.CFG.output_dir)/'0'/action/'oof_pred_probs.pkl')
            np.testing.assert_array_equal(expected,actual)

    def test_changed_training_parameters_rejected(self):
        module=load_module('bad_resume','train_fold555_disk_safe.py')
        configure(module)
        module.CFG.model.set_params(max_depth=3)
        try:
            with patch.dict(os.environ,{'MABE_RESUME_DIR':str(self.new_out)}):
                with self.assertRaisesRegex(ValueError,'model parameters differ'):
                    module.main()
        finally:
            finish(module)

    def test_feature_workers_bounded_and_model_threads_configurable(self):
        module=load_module('parallel_cache','train_fold555_disk_safe.py')
        configure(module)
        with patch.dict(os.environ,{'MABE_FEATURE_JOBS':'2','MABE_MODEL_JOBS':'2'}):
            module.main()
        self.assertEqual(module.CFG.feature_jobs,1)
        self.assertEqual(module.CFG.model_jobs,min(2,module.cpu_count()))
        for action in ('rear','approach'):
            a=joblib.load(self.new_out/'0'/action/'oof_pred_probs.pkl')
            b=joblib.load(Path(module.CFG.output_dir)/'0'/action/'oof_pred_probs.pkl')
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)

    def test_saved_trainer_loads_without_training_module(self):
        path=next((self.new_out/'0/rear').glob('*_trainer_*.pkl'))
        code='''import sys, joblib, numpy as np
model=joblib.load(sys.argv[1])
prediction=model.predict(np.zeros((3,model.estimators[0].n_features_in_),dtype=np.float32))
assert np.asarray(prediction).shape==(3,)
assert np.isfinite(prediction).all()
'''
        subprocess.run([sys.executable,'-I','-c',code,str(path)],cwd=self.root,check=True,capture_output=True,text=True)

    def test_changed_data_rejected(self):
        module=load_module('bad_data_resume','train_fold555_disk_safe.py')
        configure(module)
        try:
            with patch.dict(os.environ,{'MABE_RESUME_DIR':str(self.new_out)}), patch.object(module,'dataset_signature',return_value='different-data'):
                with self.assertRaisesRegex(ValueError,'competition data changed'):
                    module.main()
        finally:
            finish(module)

    def test_corrupted_fold_checkpoint_rejected(self):
        module=load_module('bad_checkpoint','train_fold555_disk_safe.py')
        model=joblib.load(next((self.new_out/'0/rear').glob('*_trainer_*.pkl'))).estimators[0]
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            folder=Path(temporary)
            module.save_fold_checkpoint(folder,0,model,np.array([0.1,0.9]),'signature')
            with (folder/'fold_0_predictions.npy').open('ab') as stream:
                stream.write(b'corrupted')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                module.read_fold_checkpoint(folder,0,'signature',2,model.n_features_in_)

    def test_zero_disk_cache_recomputes_identical_predictions(self):
        module=load_module('no_disk_cache','train_fold555_disk_safe.py')
        configure(module)
        with patch.dict(os.environ,{'MABE_CACHE_GB':'0'}), patch.object(np.lib.format,'open_memmap',side_effect=AssertionError('No full-fold matrix file allowed')):
            module.main()
        self.assertEqual(module.feature_store.peak_bytes,0)
        for action in ('rear','approach'):
            expected=joblib.load(self.new_out/'0'/action/'oof_pred_probs.pkl')
            actual=joblib.load(Path(module.CFG.output_dir)/'0'/action/'oof_pred_probs.pkl')
            np.testing.assert_array_equal(expected,actual)
        self.assertFalse(list(Path(module.CFG.output_dir).parent.glob('*.zip')))

    def test_task_budget_pauses_then_resumes_to_complete(self):
        module=load_module('task_limit','train_fold555_disk_safe.py')
        configure(module)
        with patch.dict(os.environ,{'MABE_MAX_NEW_TASKS':'1'}):
            module.main()
        previous=Path(module.CFG.output_dir)
        self.assertEqual(json.loads((previous/'run_status.json').read_text())['status'],'paused')
        self.assertTrue((previous/'0/rear/task_complete.json').exists())
        self.assertFalse((previous/'0/approach/task_complete.json').exists())
        resumed=load_module('resume_task_limit','train_fold555_disk_safe.py')
        configure(resumed)
        with patch.dict(os.environ,{'MABE_RESUME_DIR':str(previous),'MABE_MAX_NEW_TASKS':'1'}):
            resumed.main()
        output=Path(resumed.CFG.output_dir)
        self.assertEqual(json.loads((output/'run_status.json').read_text())['status'],'complete')
        self.assertEqual(resumed.new_tasks_this_run,1)
        for action in ('rear','approach'):
            np.testing.assert_array_equal(joblib.load(self.new_out/'0'/action/'oof_pred_probs.pkl'),
                                          joblib.load(output/'0'/action/'oof_pred_probs.pkl'))

    def test_output_budget_stops_before_fitting(self):
        module=load_module('output_limit','train_fold555_disk_safe.py')
        configure(module)
        with patch.dict(os.environ,{'MABE_OUTPUT_GB':'0.01'}), patch.object(module,'fit_one_fold',side_effect=AssertionError('Should pause before training')):
            module.main()
        status=json.loads((Path(module.CFG.output_dir)/'run_status.json').read_text())
        self.assertEqual(status['status'],'paused')
        self.assertIn('Output budget',status['error'])

    def test_cache_evicts_old_files_within_budget(self):
        module=load_module('cache_limit','train_fold555_disk_safe.py')
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            root=Path(temporary)
            store=module.FeatureStore(root,18*1024**2)
            rng=np.random.default_rng(42)
            for i in range(4):
                folder=root/str(i)
                folder.mkdir()
                values=rng.normal(size=(32768,8)).astype(np.float32)
                features=pd.DataFrame(values,columns=[f'f{x}' for x in range(8)])
                record={'directory':str(folder),'columns':list(features.columns)}
                store.put(record,features)
                self.assertLessEqual(store.bytes,store.budget)
            self.assertFalse((root/'0/features.parquet').exists())
            np.testing.assert_array_equal(store.get_rows(record,np.array([0,8191,8192,32767])),values[[0,8191,8192,32767]])

    def test_streaming_sample_seed_matches_dense_fit(self):
        module=load_module('streaming_sampling','train_fold555_disk_safe.py')
        rng=np.random.default_rng(42)
        values=rng.normal(size=(1200,24)).astype(np.float32)
        values[::13,0]=np.nan
        values[:,23]=0
        labels=(values[:,1]+values[:,3]>0).astype(np.int8)
        class Sequence(module.lgb.Sequence):
            columns=[f'f{i}' for i in range(values.shape[1])]
            def __len__(self):return len(values)
            def __getitem__(self,key):return values[key].astype(np.float64)
        configure(module)
        module.CFG.model.set_params(n_jobs=1,subsample_for_bin=128)
        frame=pd.DataFrame(values,columns=Sequence.columns)
        expected=clone(module.CFG.model).fit(frame,labels)
        actual=module.fit_sequence_model(Sequence(),labels)
        np.testing.assert_array_equal(expected.predict_proba(frame),actual.predict_proba(frame))

    def test_time_budget_pauses_after_a_completed_fold(self):
        module=load_module('time_limit','train_fold555_disk_safe.py')
        configure(module)
        original=module.check_budget
        def stop(stage,*args,**kwargs):
            if stage=='fold 2':
                module.run_started-=4*3600
            return original(stage,*args,**kwargs)
        with patch.object(module,'check_budget',side_effect=stop):
            module.main()
        folder=Path(module.CFG.output_dir)
        self.assertEqual(json.loads((folder/'run_status.json').read_text())['status'],'paused')
        self.assertTrue((folder/'0/rear/checkpoints/fold_0.json').exists())
        self.assertFalse((folder/'0/rear/checkpoints/fold_1.json').exists())


if __name__=='__main__':
    unittest.main()
