"""Offline competition inference using the verified final checkpoint."""
import argparse
import gc
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from infer_fold555 import COLUMNS, digest, intervals, load_features, predict_bundle, validate_submission


def load_bundle(folder, kind, cutoff):
    marker = json.loads((folder/'task_complete.json').read_text())
    if marker['score']['kind'] != kind:
        raise ValueError(f'Wrong model kind: {folder}')
    files = list(folder.glob('*_trainer_*.pkl'))
    if len(files) != 1:
        raise ValueError(f'Ambiguous model: {folder}')
    for name in (files[0].name,'feature_columns.json','threshold.pkl'):
        if digest(folder/name) != marker['sha256'][name]:
            raise ValueError(f'Corrupted checkpoint: {folder}/{name}')
    if float(joblib.load(folder/'threshold.pkl')) != cutoff:
        raise ValueError('Saved threshold files disagree')
    columns = json.loads((folder/'feature_columns.json').read_text())
    original = joblib.load(files[0])
    # Retain only inference models, releasing OOF arrays and training group IDs.
    compact = SimpleNamespace(is_fitted=original.is_fitted,estimators=original.estimators)
    del original
    gc.collect()
    return compact,columns


def run(checkpoint, data, output, threads=4):
    checkpoint,data,output = map(Path,(checkpoint,data,output))
    status = json.loads((checkpoint/'run_status.json').read_text())
    if status['status'] != 'complete' or status['trained_tasks'] != 84:
        raise ValueError('Expected the completed 84-task checkpoint')
    module,configurations = load_features(checkpoint,data)
    thresholds = joblib.load(checkpoint/'thresholds.pkl')
    output.mkdir(parents=True,exist_ok=True)
    csv = output/'submission.csv'
    pd.DataFrame(columns=['row_id']+COLUMNS).to_csv(csv,index=False)
    row_count = predicted_videos = records = 0
    coverage = []
    cache = {}
    cached_section = None
    start = perf_counter()
    # Group by keypoint configuration to reuse small, inference-only model bundles.
    for row in module.test.sort_values(['body_parts_tracked','video_id']).itertuples():
        if row.body_parts_tracked not in configurations:
            raise ValueError(f'Unseen body part configuration: video {row.video_id}')
        section = configurations.index(row.body_parts_tracked)
        if section != cached_section:
            cache.clear()
            gc.collect()
            cached_section = section
        subset = module.test[module.test.video_id == row.video_id]
        parts = json.loads(row.body_parts_tracked)
        if len(parts)>5:
            parts = [p for p in parts if p not in module.drop_body_parts]
        expected = {tuple(x.replace("'",'').split(',')) for x in json.loads(row.behaviors_labeled)}
        observed = set()
        predictions = []
        for kind,tracking,meta,actions in module.generate_mouse_data(subset,'test',str(data/'test_tracking')):
            limits = thresholds.get(kind,{}).get(str(section),{})
            relevant = [a for a in actions if a in limits]
            if not relevant:
                continue
            features,_,_ = module.make_features(kind,tracking,meta,parts,section)
            probabilities = pd.DataFrame(index=np.arange(len(meta)))
            for action in relevant:
                key = (kind,str(action))
                if key not in cache:
                    cache[key] = load_bundle(checkpoint/str(section)/str(action),kind,limits[action])
                bundle,columns = cache[key]
                aligned = features.reindex(columns=columns)
                probabilities[action] = predict_bundle(bundle,aligned,threads)
                observed.add((str(meta.agent_id.iloc[0]),str(meta.target_id.iloc[0]),str(action)))
                del aligned
            predictions.append(intervals(probabilities,meta,limits))
            records += 1
            del features,tracking,meta,probabilities
            gc.collect()
        result = pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(columns=COLUMNS)
        if not result.empty:
            validate_submission(result,subset,data)
            result = result.sort_values(['video_id','agent_id','target_id','start_frame']).reset_index(drop=True)
            result.insert(0,'row_id',np.arange(row_count,row_count+len(result)))
            result.to_csv(csv,mode='a',header=False,index=False)
            row_count += len(result)
            predicted_videos += 1
        coverage.append(dict(video_id=int(row.video_id),section=section,
            expected_behaviors=len(expected),predicted_behaviors=len(observed),
            unavailable_models=sorted(expected-observed),prediction_rows=len(result)))
        print(f'PREDICT video={row.video_id}, section={section}, behaviors={len(observed)}/{len(expected)}, '
              f'intervals={len(result)}, completed={len(coverage)}/{len(module.test)}, elapsed={perf_counter()-start:.1f}s',flush=True)
        del predictions,result
        gc.collect()
    if not row_count:
        raise ValueError('No genuine predictions were generated')
    report = dict(status='passed',checkpoint_tasks=84,checkpoint_folds=420,
        input_videos=len(module.test),predicted_videos=predicted_videos,prediction_records=records,
        prediction_rows=row_count,coverage=coverage,submission_sha256=digest(csv),
        elapsed_seconds=perf_counter()-start,scope='Predictions only; competition scoring is performed by Kaggle')
    (output/'submission-report.json').write_text(json.dumps(report,indent=2))
    print('SUBMISSION READY:',json.dumps({k:v for k,v in report.items() if k!='coverage'}),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args()
    run(args.checkpoint,args.data,args.output,args.threads)
