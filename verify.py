"""Reconcile the run independently of the training and scoring code."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parent


def load_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def verify():
    out = ROOT/'outputs'
    s = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    pred = load_csv(ROOT/'data'/'holdout_predictions.csv')
    assert len(pred) == s['row_counts']['holdout']
    assert len({r['geo_cell'] for r in pred}) == s['group_counts']['holdout']
    for model in s['holdout']:
        errors = [float(r[model['method']]) - float(r['actual']) for r in pred]
        expected = {'rmse': math.sqrt(fmean(e*e for e in errors)),
                    'mae': fmean(abs(e) for e in errors), 'bias': fmean(errors)}
        for key, value in expected.items():
            assert math.isclose(model[key], value, abs_tol=1e-8), (model['method'], key)
    expected_cv = defaultdict(list)
    for r in load_csv(out/'cv_trials.csv'):
        key = (r['method'], r['trial'], r['trees'])
        expected_cv[key].append(r)
    ranked = load_csv(out/'cv_ranked.csv')
    assert len(ranked) == len(expected_cv) == 44
    assert sum(map(len, expected_cv.values())) == 176
    for r in ranked:
        folds = expected_cv[(r['method'],r['trial'],r['trees'])]
        n = sum(int(f['n']) for f in folds)
        assert n == s['row_counts']['development']
        assert {f['fold'] for f in folds} == {'1','2','3','4'}
        pooled = math.sqrt(sum(float(f['sse']) for f in folds)/n)
        assert math.isclose(float(r['cv_rmse']), pooled, abs_tol=1e-9)
    assert ranked[0]['method'] == s['champion']
    assert s['audit']['raw'] == s['audit']['ineligible'] + s['audit']['duplicate_id'] + s['audit']['retained']
    assert s['audit']['retained'] == sum(s['row_counts'].values())
    print(f"Checked {len(ranked)} configurations and {len(pred):,} holdout listings; all scores reconcile.")


if __name__ == '__main__':
    verify()
