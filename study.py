"""Group-aware tuning, leakage-safe rent comparables and a locked holdout."""
from itertools import product
import json
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATS, ROOT, clean_listings, features_and_groups, load_raw

SEED = 42
N_FOLDS = 4
THREADS = 5
CHECKPOINTS = (150, 300, 500, 750)
REFERENCE_TREES = 500
REFERENCE = dict(depth=6, learning_rate=0.05, l2_leaf_reg=3, random_strength=1)
COMP_COLUMNS = ['comp_city', 'comp_state', 'comp_city_sqft', 'comp_city_count']
BANDS = ['Under $1k', '$1k–<$2k', '$2k–<$3.5k', '$3.5k+']


def split_groups(X, groups):
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(X, groups=groups))
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X.iloc[dev], groups=groups.iloc[dev]))
    return dev, test, folds


def add_comps(train, y, groups, query, exclude_own=False):
    """Smooth city -> state -> global; subtract the entire row's cell during fit."""
    if not train.index.equals(y.index) or not train.index.equals(groups.index):
        raise ValueError('Training features, rents and groups must be aligned.')
    if exclude_own and not query.index.equals(train.index):
        raise ValueError('Leave-cell-out values require the aligned training frame.')
    stats = train[['state', 'market']].assign(cell=groups, rent=y, per_sqft=y/train.square_feet)
    out = query.copy()

    def stats_for(keys):
        fields = ['rent', 'per_sqft']
        summary = stats.groupby(keys, sort=False)[fields].agg(['sum', 'count']) if keys else None
        if keys:
            key = out[keys[0]] if len(keys) == 1 else pd.MultiIndex.from_frame(out[keys])
            v = summary.reindex(key).fillna(0).to_numpy()
        else:
            v = np.tile([stats.rent.sum(), len(stats), stats.per_sqft.sum(), len(stats)], (len(query), 1))
        if exclude_own:
            cols = keys + ['cell']
            own = stats.groupby(cols, sort=False)[fields].agg(['sum', 'count'])
            key = stats.cell if not keys else pd.MultiIndex.from_frame(stats[cols])
            v = v - own.reindex(key).to_numpy()
        return v.T

    total, n, total_sqft, n_sqft = stats_for([])
    if np.any(n <= 0):
        raise ValueError('At least two geo cells are needed to build leave-cell-out comparables.')
    prior, prior_sqft = total/n, total_sqft/n_sqft
    st, sn, sts, sns = stats_for(['state'])
    state = (st + 25*prior)/(sn + 25)
    state_sqft = (sts + 25*prior_sqft)/(sns + 25)
    ct, cn, cts, cns = stats_for(['market'])
    out['comp_state'] = state
    out['comp_city'] = (ct + 15*state)/(cn + 15)
    out['comp_city_sqft'] = (cts + 15*state_sqft)/(cns + 15)
    out['comp_city_count'] = np.log1p(cn)
    return out


def metrics(actual, prediction):
    error = np.asarray(prediction) - np.asarray(actual)
    return {'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(np.abs(error))),
            'bias': float(np.mean(error)), 'n': len(error)}


def search_space():
    cb = [REFERENCE.copy()] + [dict(zip(REFERENCE, v)) for v in [
        (4,.10,3,.5), (5,.04,10,1.5), (5,.10,5,1), (6,.09,8,2),
        (7,.035,8,.5), (7,.07,3,1), (8,.04,5,1.5), (8,.08,10,1)]]
    rf = [dict(max_features=f, min_samples_leaf=m) for f,m in product([.45,.8], [1,3,8])]
    return cb, rf


def catboost(p, trees):
    return CatBoostRegressor(iterations=trees, **p, loss_function='RMSE', boosting_type='Plain',
        cat_features=CATS, one_hot_max_size=12, max_ctr_complexity=1, border_count=64,
        thread_count=THREADS, random_seed=SEED, allow_writing_files=False, verbose=False)


def sklearn_model(kind, X, p=None):
    nums = [x for x in X.columns if x not in CATS]
    numeric = [SimpleImputer(strategy='median', add_indicator=True)]
    if kind == 'Ridge':
        numeric.append(StandardScaler())
    prep = ColumnTransformer([
        ('num', make_pipeline(*numeric), nums),
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10), CATS),
    ])
    estimator = Ridge(alpha=10, solver='lsqr') if kind == 'Ridge' else RandomForestRegressor(
        n_estimators=180, **p, n_jobs=THREADS, random_state=SEED)
    return make_pipeline(prep, estimator)


def scores(trials, group_keys):
    grouped = trials.groupby(group_keys, as_index=False, sort=False).agg(
        sse=('sse','sum'), n=('n','sum'), fold_sd=('rmse','std'), avg_fit_seconds=('fit_seconds','mean'))
    grouped['cv_rmse'] = np.sqrt(grouped.sse / grouped.n)
    return grouped.sort_values(['cv_rmse', *group_keys]).reset_index(drop=True)


def prepare(X, y, g, a, b):
    t, v = X.iloc[a], X.iloc[b]
    yt, yv = y.iloc[a], y.iloc[b]
    gt = g.iloc[a]
    assert not set(gt) & set(g.iloc[b])
    return add_comps(t, yt, gt, t, exclude_own=True), yt, add_comps(t, yt, gt, v), yv


def predict(model, X, **kwargs):
    return np.maximum(0, model.predict(X, **kwargs))


def fit_model(name, train, target, options=None, trees=None):
    if name == 'Median':
        return None
    if name == 'CatBoost':
        return catboost(options, trees).fit(train, target)
    return sklearn_model(name, train, options).fit(train, target)


def bootstrap_difference(y, predictions, competing, groups):
    """Resample whole holdout cells, paired across fitted models."""
    cluster = pd.DataFrame({'a': (y-predictions)**2, 'b': (y-competing)**2,
                            'n': 1, 'cell': np.asarray(groups)}).groupby('cell')[['a','b','n']].sum().to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(cluster), size=(1200,len(cluster)))
    a,b,n = cluster[idx].sum(axis=1).T
    return np.quantile(np.sqrt(a/n) - np.sqrt(b/n), [.025,.975]).tolist()


def run(out, work):
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    clean, audit = clean_listings(raw)
    X, groups = features_and_groups(clean)
    y = clean.price.astype(float)
    dev, test, folds = split_groups(X, groups)
    fold_frames = [prepare(X, y, groups, dev[a], dev[b]) for a,b in folds]
    cb, forests = search_space()
    (out/'search_space.json').write_text(json.dumps({
        'catboost': cb, 'catboost_tree_counts': CHECKPOINTS,
        'random_forest': forests, 'random_forest_tree_count': 180,
        'trial_ids': 'Zero-based indices into each list.'}, indent=2)+'\n', encoding='utf-8')
    history, ablation = [], []

    def row(fold, method, trial, trees, yv, p, took):
        return dict(fold=fold, method=method, trial=trial, trees=trees, rmse=metrics(yv,p)['rmse'],
                    sse=float(np.square(yv-p).sum()), n=len(yv), fit_seconds=took)

    for fold,(t,yt,v,yv) in enumerate(fold_frames,1):
        print(f'Grouped CV {fold}/{N_FOLDS}: {len(t)} train, {len(v)} validation', flush=True)
        for kind, params in [('Median',[None]), ('Ridge',[None]), ('Random forest',forests), ('CatBoost',cb)]:
            for idx,p in enumerate(params):
                start = perf_counter()
                model = fit_model(kind,t,yt,p,max(CHECKPOINTS))
                elapsed = perf_counter()-start
                counts = CHECKPOINTS if kind == 'CatBoost' else [180 if kind == 'Random forest' else 0]
                for trees in counts:
                    if kind == 'Median':
                        estimate = np.full(len(v), yt.median())
                    elif kind == 'CatBoost':
                        estimate = predict(model,v,ntree_end=trees,thread_count=THREADS)
                    else:
                        estimate = predict(model,v)
                    history.append(row(fold,kind,idx,trees,yv,estimate,elapsed))
    trials = pd.DataFrame(history)
    ranked = scores(trials, ['method','trial','trees'])
    best = ranked.groupby('method',sort=False).head(1).copy()
    champion = best.iloc[0].method
    best_cat = best[best.method.eq('CatBoost')].iloc[0]
    best_rf = best[best.method.eq('Random forest')].iloc[0]
    fixed_cat = ranked[ranked.method.eq('CatBoost') & ranked.trial.eq(0) & ranked.trees.eq(REFERENCE_TREES)].iloc[0]
    p_cat, n_cat = cb[int(best_cat.trial)], int(best_cat.trees)

    for fold,(t,yt,v,yv) in enumerate(fold_frames,1):
        for label, drop, is_log in [('Without rent comparables',COMP_COLUMNS,False),('Log target with comparables',[],True)]:
            start = perf_counter()
            model = fit_model('CatBoost',t.drop(columns=drop),np.log1p(yt) if is_log else yt,p_cat,n_cat)
            pred = model.predict(v.drop(columns=drop), thread_count=THREADS)
            ablation.append(row(fold,label,int(best_cat.trial),n_cat,yv,np.maximum(0,np.expm1(pred) if is_log else pred),perf_counter()-start))
    comparisons = scores(pd.DataFrame(ablation), ['method','trial','trees'])

    # No holdout labels were used to select parameters or fit any features.
    print('Configurations locked; refitting for the final holdout', flush=True)
    train, yt, held, ytest = prepare(X,y,groups,dev,test)
    pred = pd.DataFrame({'actual': ytest, 'geo_cell': groups.iloc[test],
        'seen_market': X.market.iloc[test].isin(X.market.iloc[dev]),
        'price_band': pd.cut(ytest,[-np.inf,1000,2000,3500,np.inf],labels=BANDS,right=False)})
    holdout = []
    for line in best.itertuples():
        name = line.method
        p = p_cat if name == 'CatBoost' else forests[int(best_rf.trial)] if name == 'Random forest' else None
        model = fit_model(name,train,yt,p,int(line.trees))
        estimate = np.full(len(held),yt.median()) if name == 'Median' else predict(model,held)
        pred[name] = estimate
        holdout.append(dict(method=name, **metrics(ytest,estimate)))
        if name == 'CatBoost':
            sample = held.sample(n=min(300,len(held)), random_state=SEED)
            sh = model.get_feature_importance(Pool(sample,cat_features=CATS),type='ShapValues',thread_count=THREADS)
            np.testing.assert_allclose(sh.sum(axis=1), model.predict(sample,thread_count=THREADS), rtol=1e-8,atol=1e-6)
            pd.Series(np.abs(sh[:,:-1]).mean(axis=0),index=sample.columns,name='mean_abs_shap').sort_values(ascending=False).to_csv(out/'shap.csv',index_label='feature')
            model.save_model(work/'catboost.cbm')
    slices = []
    for dimension in ['price_band','seen_market']:
        for label, part in pred.groupby(dimension,observed=True):
            slices.append(dict(dimension=dimension,segment=str(label), **metrics(part.actual,part[champion])))
    pd.DataFrame(slices).to_csv(out/'holdout_slices.csv',index=False)
    pred.to_csv(work/'holdout_predictions.csv', index=False)
    trials.to_csv(out/'cv_trials.csv',index=False)
    ranked.to_csv(out/'cv_ranked.csv',index=False)
    pd.DataFrame(ablation).to_csv(out/'ablation_folds.csv',index=False)
    missing = clean.iloc[dev][['amenities','pets_allowed','bedrooms','bathrooms','latitude','longitude','cityname']].isna().sum()
    dates = pd.to_datetime(clean.time,unit='s',utc=True).dt.date
    summary = dict(audit=audit, date_start=str(dates.min()),date_end=str(dates.max()),
        group_counts={'development': int(groups.iloc[dev].nunique()), 'holdout': int(groups.iloc[test].nunique())},
        row_counts={'development':len(dev),'holdout':len(test)}, dev_missing=missing.to_dict(),
        dev_median=float(yt.median()), dev_p99=float(yt.quantile(.99)),
        folds=[dict(fold=i,train=len(a),validation=len(b),train_cells=int(groups.iloc[dev[a]].nunique()),
                    val_cells=int(groups.iloc[dev[b]].nunique())) for i,(a,b) in enumerate(folds,1)],
        candidate_counts={'CatBoost tree-prefix scores':len(cb)*len(CHECKPOINTS),'Random forest':len(forests)},
        champion=champion,cv=best[['method','cv_rmse','fold_sd','avg_fit_seconds']].to_dict('records'),
        holdout=holdout, cat_fixed_cv=float(fixed_cat.cv_rmse),
        cat_params=dict(p_cat,iterations=n_cat),rf_params=dict(forests[int(best_rf.trial)],n_estimators=180),
        ablations=comparisons[['method','cv_rmse']].to_dict('records'),
        cat_minus_rf_ci=bootstrap_difference(ytest,pred['CatBoost'],pred['Random forest'],groups.iloc[test]),
        holdout_seen_market_rows=int(pred.seen_market.sum()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame({'rent':y.iloc[dev],'square_feet':X.square_feet.iloc[dev],
                  'bedrooms':X.bedrooms.iloc[dev]}).to_csv(work/'development_eda.csv',index=False)
    print(json.dumps(summary,indent=2), flush=True)


if __name__ == '__main__':
    run(ROOT/'outputs',ROOT/'data')
