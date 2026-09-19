"""Verify a completed fold555 checkpoint and predict the visible competition test set.

Uses the exact feature functions and category vocabulary saved with the checkpoint.
Does not train models, create placeholder predictions, or submit to the leaderboard.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_features(checkpoint, data):
    environment = json.loads((checkpoint/'environment.json').read_text())
    for name, expected in environment['packages'].items():
        if version(name) != expected:
            raise ValueError(f'Install {name}=={expected} to match this checkpoint')
    spec = importlib.util.spec_from_file_location('saved_mabe_features', checkpoint/'train_fold555.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    categories = json.loads((checkpoint/'meta_categories.json').read_text())
    module.META_CAT_CATEGORIES.update(categories)
    module.META_CAT_KNOWN_SET.update({k: set(v) for k, v in categories.items()})
    module.META_CAT_ENCODERS.update({k: {v:i for i,v in enumerate(values)} for k,values in categories.items()})
    module.META_CAT_UNK.update({k: len(v) for k, v in categories.items()})
    module.train = pd.read_csv(data/'train.csv')
    module.test = pd.read_csv(data/'test.csv')
    for frame in (module.train, module.test):
        frame['n_mice'] = 4-frame[[f'mouse{i}_strain' for i in range(1,5)]].isna().sum(axis=1)
    configurations = json.loads((checkpoint/'body_part_configurations.json').read_text())
    if list(np.unique(module.train.body_parts_tracked)) != configurations:
        raise ValueError('Training metadata does not match saved section numbering')
    module.arena_data = pd.concat([f[['video_id','arena_width_cm','arena_height_cm','arena_shape']]
                                  for f in (module.train,module.test)]).drop_duplicates('video_id').set_index('video_id')
    return module, configurations


def check_probabilities(values):
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Prediction contains invalid probabilities')


def predict_bundle(bundle, features, threads=2):
    if not bundle.is_fitted or len(bundle.estimators) != 5:
        raise ValueError('Expected five fitted models')
    total = np.zeros(len(features), dtype=np.float64)
    # LightGBM replaces spaces in stored feature names with underscores.
    # Keep the checkpoint's original column order while comparing that encoding.
    names = [str(c).replace(' ', '_') for c in features.columns]
    if len(names) != len(set(names)):
        raise ValueError('Feature names collide after LightGBM normalization')
    for model in bundle.estimators:
        if model.n_features_in_ != features.shape[1] or list(model.feature_name_) != names:
            raise ValueError('Model feature names or ordering differ')
        if list(model.classes_) != [0, 1]:
            raise ValueError('Unexpected class ordering')
        model.set_params(n_jobs=threads)
        predicted = model.predict_proba(features)[:,1]
        check_probabilities(predicted)
        total += predicted
    return total/len(bundle.estimators)


def intervals(probabilities, meta, thresholds):
    """Original argmax/threshold rule, preserving the final interval and frame gaps."""
    if not len(meta) or not len(probabilities.columns):
        return pd.DataFrame(columns=COLUMNS)
    if len(probabilities) != len(meta):
        raise ValueError('Prediction and frame counts differ')
    identities = meta[['video_id','agent_id','target_id']].drop_duplicates()
    if len(identities) != 1:
        raise ValueError('Convert one video/agent/target at a time')
    values = probabilities.to_numpy()
    check_probabilities(values)
    frames = meta.video_frame.to_numpy(dtype=np.int64)
    if (np.diff(frames) <= 0).any():
        raise ValueError('Frame indices must be strictly increasing')
    winner = np.argmax(values, axis=1)
    cutoffs = np.array([thresholds[a] for a in probabilities.columns])
    labels = np.where(values[np.arange(len(values)),winner] >= cutoffs[winner],winner,-1)
    starts = np.flatnonzero(np.r_[True, (labels[1:] != labels[:-1]) | (np.diff(frames) != 1)])
    ends = np.r_[starts[1:],len(frames)]
    identity = identities.iloc[0]
    rows = [(int(identity.video_id),str(identity.agent_id),str(identity.target_id),
             str(probabilities.columns[labels[start]]),int(frames[start]),int(frames[end-1]+1))
            for start,end in zip(starts,ends) if labels[start] >= 0]
    return pd.DataFrame(rows, columns=COLUMNS)


def audit(checkpoint, thresholds, output, threads, audit_oof):
    rows = []
    for marker in sorted(checkpoint.glob('*/*/task_complete.json')):
        task = json.loads(marker.read_text())
        folder = marker.parent
        for name, expected in task['sha256'].items():
            if digest(folder/name) != expected:
                raise ValueError(f'Corrupted checkpoint: {folder.name}/{name}')
        score = task['score']
        section,kind,action = str(score['section']),score['kind'],score['action']
        cutoff = float(joblib.load(folder/'threshold.pkl'))
        if cutoff != thresholds[kind][section][action] or not 0 <= cutoff <= 1:
            raise ValueError('Threshold files disagree')
        columns = json.loads((folder/'feature_columns.json').read_text())
        model_files = list(folder.glob('*_trainer_*.pkl'))
        if len(model_files) != 1 or len(columns) != len(set(columns)):
            raise ValueError('Ambiguous model or duplicated columns')
        bundle = joblib.load(model_files[0])
        # Synthetic inputs check serialization/predictability only, not model accuracy.
        synthetic = pd.DataFrame(np.vstack([np.zeros(len(columns)),np.full(len(columns),np.nan)]),columns=columns)
        prediction = predict_bundle(bundle, synthetic, threads)
        row = dict(section=section,kind=kind,action=action,folds=len(bundle.estimators),features=len(columns),
                   threshold=cutoff,smoke_test='passed',saved_binary_f1=score['binary F1 score'])
        if audit_oof:
            tp = fp = fn = count = 0
            folds_seen = set()
            for batch in pq.ParquetFile(folder/'oof_predictions.parquet').iter_batches(
                    batch_size=65536,columns=['label','prediction','fold']):
                frame = batch.to_pandas()
                y,p = frame.label.to_numpy(),frame.prediction.to_numpy()
                check_probabilities(p)
                if not np.isin(y,[0,1]).all() or not np.isin(frame.fold,[0,1,2,3,4]).all():
                    raise ValueError('Invalid out-of-fold labels or fold indices')
                predicted = p >= cutoff
                tp += int(((y == 1)&predicted).sum())
                fp += int(((y == 0)&predicted).sum())
                fn += int(((y == 1)&~predicted).sum())
                count += len(frame)
                folds_seen.update(frame.fold.unique().tolist())
            f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
            if folds_seen != set(range(5)) or not np.isclose(f1,score['binary F1 score'],atol=1e-12):
                raise ValueError(f'Saved validation score mismatch: {section}/{action}')
            row.update(oof_rows=count,recomputed_binary_f1=f1)
        rows.append(row)
        del bundle,synthetic,prediction
        gc.collect()
        if len(rows)%10 == 0:
            print(f'AUDIT {len(rows)} tasks verified',flush=True)
    if len(rows) != json.loads((checkpoint/'run_status.json').read_text())['trained_tasks']:
        raise ValueError('Task inventory is incomplete')
    pd.DataFrame(rows).to_csv(output/'model-audit.csv',index=False)
    return rows


def validate_submission(submission, dataset, data):
    if list(submission.columns) != COLUMNS or submission.empty:
        raise ValueError('No genuine model predictions were generated')
    if submission.isna().any().any() or submission.duplicated().any():
        raise ValueError('Missing values or duplicate prediction rows')
    for row in dataset.itertuples():
        sample = submission[submission.video_id == row.video_id]
        if sample.empty:
            continue
        allowed = {tuple(x.replace("'",'').split(',')) for x in json.loads(row.behaviors_labeled)}
        tracking = pq.read_table(data/'test_tracking'/row.lab_id/f'{row.video_id}.parquet',columns=['video_frame'])
        frames = tracking['video_frame'].to_numpy()
        if not all((r.agent_id,r.target_id,r.action) in allowed for r in sample.itertuples()):
            raise ValueError('Prediction uses an unlabelled behavior or mouse pair')
        if ((sample.start_frame < frames.min()) | (sample.stop_frame > frames.max()+1)
                | (sample.start_frame >= sample.stop_frame)).any():
            raise ValueError('Prediction interval is outside the video')
    if not set(submission.video_id) <= set(dataset.video_id):
        raise ValueError('Unknown video in submission')
    for _, group in submission.groupby(['video_id','agent_id','target_id']):
        group = group.sort_values('start_frame')
        if (group.start_frame.to_numpy()[1:] < group.stop_frame.to_numpy()[:-1]).any():
            raise ValueError('Overlapping prediction intervals')


def run(checkpoint, data, output, threads=2, audit_oof=False, export_inputs=False):
    checkpoint,data,output = map(Path,(checkpoint,data,output))
    output.mkdir(parents=True,exist_ok=True)
    if json.loads((checkpoint/'run_status.json').read_text())['status'] != 'complete':
        raise ValueError('Checkpoint training is not complete')
    module,configurations = load_features(checkpoint,data)
    thresholds = joblib.load(checkpoint/'thresholds.pkl')
    audits = audit(checkpoint,thresholds,output,threads,audit_oof)
    predictions,coverage,record_predictions = [],[],[]
    legacy_rows = 0
    for row in module.test.itertuples():
        if row.body_parts_tracked not in configurations:
            raise ValueError(f'Unseen body part configuration for video {row.video_id}')
        section = configurations.index(row.body_parts_tracked)
        subset = module.test[module.test.video_id == row.video_id]
        parts = json.loads(row.body_parts_tracked)
        if len(parts)>5:
            parts = [p for p in parts if p not in module.drop_body_parts]
        expected = {tuple(x.replace("'",'').split(',')) for x in json.loads(row.behaviors_labeled)}
        observed = set()
        for kind,tracking,meta,actions in module.generate_mouse_data(subset,'test',str(data/'test_tracking')):
            agent,target = str(meta.agent_id.iloc[0]),str(meta.target_id.iloc[0])
            relevant = [a for a in actions if a in thresholds.get(kind,{}).get(str(section),{})]
            if not relevant:
                continue
            features,_,_ = module.make_features(kind,tracking,meta,parts,section)
            probabilities = pd.DataFrame(index=np.arange(len(meta)))
            for action in relevant:
                folder = checkpoint/str(section)/str(action)
                columns = json.loads((folder/'feature_columns.json').read_text())
                model = joblib.load(next(folder.glob('*_trainer_*.pkl')))
                aligned = features.reindex(columns=columns)
                probabilities[action] = predict_bundle(model,aligned,threads)
                observed.add((agent,target,str(action)))
                record_predictions.append(dict(video_id=int(row.video_id),section=section,kind=kind,
                    agent_id=agent,target_id=target,action=str(action),frames=len(meta),
                    expected_features=len(columns),missing_features=len(set(columns)-set(features.columns)),
                    min_probability=float(probabilities[action].min()),max_probability=float(probabilities[action].max())))
                del model,aligned
                gc.collect()
            limits = thresholds[kind][str(section)]
            original = module.predict_multiclass(probabilities,meta,limits)
            legacy_rows += len(original)
            predictions.append(intervals(probabilities,meta,limits))
            # Retain real frame probabilities so both exports can be inspected/reproduced.
            trace = meta[['video_id','agent_id','target_id','video_frame']].reset_index(drop=True)
            trace = pd.concat([trace,probabilities.add_prefix('probability_')],axis=1)
            trace.to_parquet(output/f'probabilities-{row.video_id}-{agent}-{target}.parquet',index=False)
            del features,tracking,meta,probabilities,trace
            gc.collect()
        missing = sorted(expected-observed)
        coverage.append(dict(video_id=int(row.video_id),expected_behaviors=len(expected),
                             predicted_behaviors=len(observed),unavailable_models=missing))
        print(f'PREDICT video={row.video_id}, behavior coverage={len(observed)}/{len(expected)}',flush=True)
    result = pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(columns=COLUMNS)
    validate_submission(result,module.test,data)
    result = result.sort_values(['video_id','agent_id','target_id','start_frame']).reset_index(drop=True)
    result.index.name = 'row_id'
    result.to_csv(output/'submission.csv')
    reloaded = pd.read_csv(output/'submission.csv')
    assert list(reloaded.columns) == ['row_id']+COLUMNS
    assert reloaded.row_id.tolist() == list(range(len(reloaded)))
    report = dict(status='passed',checkpoint_tasks=len(audits),checkpoint_folds=sum(r['folds'] for r in audits),
                  visible_test_videos=len(module.test),predicted_videos=int(result.video_id.nunique()),
                  prediction_rows=len(result),coverage=coverage,prediction_records=record_predictions,
                  original_interval_rows=legacy_rows,interval_export='argmax + saved thresholds; final interval retained; gaps split',
                  oof_scores_recomputed=audit_oof,submission_sha256=digest(output/'submission.csv'),
                  scope='Visible competition test data only; no hidden-test or leaderboard evaluation')
    (output/'inference-verification.json').write_text(json.dumps(report,indent=2))
    if export_inputs:
        target = output/'verification_inputs'
        target.mkdir(exist_ok=True)
        for name in ('train.csv','test.csv','sample_submission.csv'):
            if (data/name).exists():
                shutil.copy2(data/name,target/name)
        for row in module.test.itertuples():
            path = Path('test_tracking')/row.lab_id/f'{row.video_id}.parquet'
            (target/path).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(data/path,target/path)
    print('INFERENCE VERIFIED:',json.dumps({k:report[k] for k in ('checkpoint_tasks','checkpoint_folds','visible_test_videos','prediction_rows')}),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--audit-oof',action='store_true')
    parser.add_argument('--export-inputs',action='store_true')
    args = parser.parse_args()
    run(args.checkpoint,args.data,args.output,args.threads,args.audit_oof,args.export_inputs)
