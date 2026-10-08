"""Nested-CV polynomial regression (CPU only)

Each fold fits its own term scaler. SVD is shared across ridge penalties only inside the same fold, 
feature subset and degree. Sparse paths share warm starts only inside that same boundary. Outer folds are 
never used to choose a model.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from polynomial import design, powers, predict, load_data, save_json


def metrics(y, pred):
    return {"mse": float(mean_squared_error(y, pred)), "r2": float(r2_score(y, pred))}


def candidates(config, features, degree, sparse):
    common = {"features": features, "degree": degree}
    out = [dict(common, model="ols" if a == 0 else "ridge", alpha=a, l1_ratio=None)
           for a in config["ridge_alphas"]]
    if len(features) == 3 and degree >= 6:
        out.extend(dict(common, model="ridge", alpha=a, l1_ratio=None, degree_penalty=q)
                   for q in config.get("degree_penalties", [])
                   for a in config["ridge_alphas"] if a > 0)
    if sparse:
        for ratio in [1.0] + config["elastic_ratios"]:
            out.extend(dict(common, model="lasso" if ratio == 1 else "elasticnet",
                            alpha=a, l1_ratio=ratio)
                       for a in sorted(config["sparse_alphas"], reverse=True))
        if len(features) == 6:
            out.extend(dict(common, model="relaxed_lasso", alpha=a, l1_ratio=1.,
                            refit_alpha=.1, refit_fraction=b)
                       for a in [.01, .02, .03] for b in [.5, 1.])
            out.extend(dict(common, model="adaptive_lasso", alpha=a, l1_ratio=1.,
                            initial_alpha=initial, adaptive_power=gamma)
                       for initial in [.003, .01] for gamma in [.5, 1.]
                       for a in [.01, .003, .001, .0003])
    return out


def fit_path(z, y, candidate_list, config):
    """Return coefficient vectors, marking nonconvergent candidates ineligible"""
    yc = y - y.mean()
    decompositions = {}
    estimators = {}
    sparse_cache = {}
    adaptive_paths = {}
    result = []
    for c in candidate_list:
        if c["model"] in ["ridge", "ols"]:
            q = c.get("degree_penalty", 0)
            if q not in decompositions:
                weights = (powers(len(c['features']), c['degree']).sum(axis=1) / c['degree']) ** (-float(q))
                u, s, vt = svd(z * weights, full_matrices=False, check_finite=False)
                decompositions[q] = weights, s, vt, u.T @ yc
            weights, s, vt, uy = decompositions[q]
            cutoff = np.finfo(float).eps * max(z.shape) * s[0]
            if c["alpha"] == 0:
                factors = np.divide(1, s, out=np.zeros_like(s), where=s > cutoff)
            else:
                factors = s / (s*s + c["alpha"])
            result.append((weights * (vt.T @ (factors * uy)), None))
            continue
        key = c["l1_ratio"]
        if c['model'] == 'adaptive_lasso':
            initial_key = (1., c['initial_alpha'])
            if initial_key not in sparse_cache:
                initial = dict(c, model='lasso', alpha=c['initial_alpha'])
                sparse_cache[initial_key] = fit_path(z, y, [initial], config)[0]
            initial_coef, initial_failure = sparse_cache[initial_key]
            if initial_failure:
                result.append((initial_coef, initial_failure))
                continue
            weights = np.maximum(np.abs(initial_coef), .01)**c['adaptive_power']
            path_key = (c['initial_alpha'], c['adaptive_power'])
            if path_key not in adaptive_paths:
                adaptive_paths[path_key] = Lasso(fit_intercept=False, warm_start=True,
                    precompute=True, tol=config['sparse_tol'])
            est = adaptive_paths[path_key]
            failures = []
            for budget in [config['sparse_max_iter'], config['sparse_retry_iter']]:
                est.set_params(alpha=c['alpha'], max_iter=budget)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    est.fit(z * weights, yc)
                failures = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                if not failures:
                    break
            result.append((weights*est.coef_, failures[0] if failures else None))
            continue
        if key not in estimators:
            cls = Lasso if key == 1 else ElasticNet
            extra = {} if key == 1 else {"l1_ratio": key}
            estimators[key] = cls(alpha=c["alpha"], fit_intercept=False,
                max_iter=config["sparse_max_iter"], tol=config["sparse_tol"],
                warm_start=True, precompute=True, **extra)
        est = estimators[key]
        est.set_params(alpha=c["alpha"])
        cache_key = (key, c['alpha'])
        if cache_key not in sparse_cache:
            failures = []
            for budget in dict.fromkeys([config['sparse_max_iter'], config.get('sparse_retry_iter', config['sparse_max_iter'])]):
                est.set_params(max_iter=budget)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    est.fit(z, yc)
                failures = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                if not failures:
                    break
            sparse_cache[cache_key] = est.coef_.copy(), failures[0] if failures else None
        coef, failure = sparse_cache[cache_key]
        coef = coef.copy()
        if c['model'] == 'relaxed_lasso' and not failure:
            active = np.abs(coef) > 1e-8
            if active.any():
                from sklearn.linear_model import Ridge
                refit = np.zeros_like(coef)
                refit[active] = Ridge(alpha=c['refit_alpha'], fit_intercept=False,
                                      solver='cholesky').fit(z[:, active], yc).coef_
                coef = (1-c['refit_fraction'])*coef + c['refit_fraction']*refit
        result.append((coef, failure))
    return result


def search(frame, config, problem, output):
    output.mkdir(parents=True, exist_ok=True)
    splits = list(KFold(config["inner_folds"], shuffle=True,
                        random_state=config["seed"]).split(frame))
    y = frame.y.to_numpy()
    records = []
    for group in config["problems"][str(problem)]:
        x = frame[group["features"]].to_numpy()
        for degree in range(1, group["max_degree"] + 1):
            cs = candidates(config, group["features"], degree, degree <= group["sparse_max_degree"])
            exponents = powers(x.shape[1], degree)
            raw = design(x, exponents)  # Stateless expansion has no fitted quantities.
            rows = [dict(candidate=c, fold_mse=[], failures=[]) for c in cs]
            for fold, (ti, vi) in enumerate(splits, 1):
                scaler = StandardScaler().fit(raw[ti])
                zt = np.asfortranarray(scaler.transform(raw[ti]))
                zv = scaler.transform(raw[vi])
                for row, (coef, failure) in zip(rows, fit_path(zt, y[ti], cs, config)):
                    pred = zv @ coef + y[ti].mean()
                    if failure or not np.isfinite(pred).all():
                        row["failures"].append({"fold": fold, "reason": failure or "nonfinite"})
                        row["fold_mse"].append(None)
                    else:
                        row["fold_mse"].append(float(np.mean((pred-y[vi])**2)))
            for row in rows:
                row["mean_mse"] = None if row["failures"] else float(np.mean(row["fold_mse"]))
            records.extend(rows)
            best_degree = min((r['mean_mse'] for r in rows if r['mean_mse'] is not None), default=None)
            print(f"  {len(group['features'])} inputs, degree {degree}: best inner MSE={best_degree}", flush=True)
    eligible = [r for r in records if r["mean_mse"] is not None]
    if problem == 2 and config.get('polynomial_averages', False):
        # Select members using only inner-CV scores. Their average is itself a polynomial and is evaluated 
        # on exactly the same inner splits.
        ordered = sorted((r for r in eligible if len(r['candidate']['features']) == 3),
                         key=lambda r: r['mean_mse'])
        distinct = []
        seen = set()
        for row in ordered:
            if row['candidate']['degree'] not in seen:
                seen.add(row['candidate']['degree'])
                distinct.append(row['candidate'])
        modes = [(f'top{n}', [r['candidate'] for r in ordered[:n]]) for n in [3,5,10]]
        modes += [(f'degrees{n}', distinct[:n]) for n in [3,5,10]]
        prediction_cache = {}
        for name, members in modes:
            failures = []
            for member in members:
                key = json.dumps(member, sort_keys=True)
                if key in prediction_cache:
                    continue
                pp = np.zeros(len(frame))
                try:
                    for ti, vi in splits:
                        fitted = fit_model(frame.iloc[ti], member, config, member['features'])
                        pp[vi] = predict(fitted, frame.iloc[vi])
                    prediction_cache[key] = pp
                except RuntimeError as error:
                    failures.append({'reason': str(error)})
            if failures:
                continue
            pp = np.mean([prediction_cache[json.dumps(m, sort_keys=True)] for m in members], axis=0)
            scores = [float(np.mean((pp[vi]-y[vi])**2)) for _, vi in splits]
            candidate = dict(model='polynomial_average', features=members[0]['features'],
                degree=max(m['degree'] for m in members), alpha=None, l1_ratio=None,
                averaging_rule=name, members=members)
            row = dict(candidate=candidate, fold_mse=scores, failures=[], mean_mse=float(np.mean(scores)))
            records.append(row)
            eligible.append(row)
            print(f"  polynomial average {name}: inner MSE={row['mean_mse']:.6f}", flush=True)
    best = min(eligible, key=lambda r: (r["mean_mse"], r["candidate"]["degree"], len(r["candidate"]["features"])))
    save_json(output / "search.json", records)
    pd.DataFrame([dict(**r["candidate"], mean_mse=r["mean_mse"],
        invalid_folds=len(r["failures"])) for r in records]).to_csv(output / "search.csv", index=False)
    return best["candidate"], best["mean_mse"], len(records), sum(bool(r["failures"]) for r in records)


def fit_model(frame, candidate, config, input_columns):
    if candidate['model'] == 'polynomial_average':
        members = [fit_model(frame, c, config, input_columns) for c in candidate['members']]
        largest = max(members, key=lambda m: len(m['coef']))
        coef = np.zeros(len(largest['coef']))
        for model in members:
            n = len(model['coef'])
            # All fits use the same rows and input order; monomials are prefixes
            if model['powers'] != largest['powers'][:n]:
                raise ValueError('Polynomial-average terms do not align')
            np.testing.assert_allclose(model['term_mean'], largest['term_mean'][:n], atol=1e-12)
            np.testing.assert_allclose(model['term_scale'], largest['term_scale'][:n], atol=1e-12)
            coef[:n] += np.asarray(model['coef'])/len(members)
        return {**largest, **candidate, 'coef': coef.tolist(),
                'nonzero_terms': int(np.sum(np.abs(coef)>1e-10))}
    x = frame[candidate["features"]].to_numpy()
    exponents = powers(x.shape[1], candidate["degree"])
    raw = design(x, exponents)
    scaler = StandardScaler().fit(raw)
    z = np.asfortranarray(scaler.transform(raw))
    coef, failure = fit_path(z, frame.y.to_numpy(), [candidate], config)[0]
    if failure:
        raise RuntimeError(f"Selected model failed convergence: {failure}")
    return dict(candidate, input_columns=input_columns, powers=exponents.tolist(),
        term_mean=scaler.mean_.tolist(), term_scale=scaler.scale_.tolist(),
        coef=coef.tolist(), intercept=float(frame.y.mean()),
        nonzero_terms=int(np.sum(np.abs(coef) > 1e-10)))


def run(problem, config, data_dir, output):
    config = {**config, **config.get("problem_settings", {}).get(str(problem), {})}
    start = time.time()
    roll = config["rollno"]
    cols = [f"x{i}" for i in range(1, (6 if problem == 1 else 3)+1)]
    train_file = data_dir / f"{roll}_train_var{problem}.csv"
    test_file = data_dir / f"{roll}_test_var{problem}.csv"
    train = load_data(train_file, cols, True)
    test = load_data(test_file, cols)
    dest = output / f"var{problem}"
    dest.mkdir(parents=True, exist_ok=True)
    save_json(dest / "manifest.json", dict(config=config, python=platform.python_version(),
        versions={p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "pandas"]},
        sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [train_file, test_file, Path(__file__), Path(__file__).with_name("polynomial.py")]},
        training_rows=len(train), test_rows=len(test),
        missing=int(train.isna().sum().sum()), duplicate_rows=int(train.duplicated().sum())))
    oof = np.zeros(len(train)); folds = np.zeros(len(train), dtype=int)
    summaries = []
    baseline_oof = {"reduced_input_ols": np.zeros(len(train)), "mean": np.zeros(len(train))}
    # Reduced-input ablation: low-degree OLS on a subset of the inputs
    ablation = dict(features=cols[:3] if problem == 1 else cols[:1], degree=3 if problem == 1 else 4,
                    model="ols", alpha=0.0, l1_ratio=None)
    for fold, (ti, vi) in enumerate(KFold(config["outer_folds"], shuffle=True,
                                      random_state=config["seed"]+1).split(train), 1):
        print(f"var{problem}: outer fold {fold} search", flush=True)
        fit = train.iloc[ti]; valid = train.iloc[vi]
        selected, inner_mse, count, failures = search(fit, config, problem, dest / f"outer_{fold}")
        model = fit_model(fit, selected, config, cols)
        pred = predict(model, valid); oof[vi] = pred; folds[vi] = fold
        result = dict(fold=fold, selected=selected, selection_mse=inner_mse, candidates=count,
                      invalid_candidates=failures, **metrics(valid.y, pred))
        summaries.append(result)
        baseline_oof["reduced_input_ols"][vi] = predict(fit_model(fit, ablation, config, cols), valid)
        baseline_oof["mean"][vi] = fit.y.mean()
        print(f"var{problem}: fold {fold}: MSE={result['mse']:.6f}, R2={result['r2']:.6f}, {selected}", flush=True)
        save_json(dest / "outer_folds.json", summaries)
    pd.DataFrame(dict(row=np.arange(len(train)), fold=folds, y=train.y,
                      prediction=oof, **baseline_oof)).to_csv(dest / "oof_predictions.csv", index=False)
    print(f"var{problem}: final search on all rows", flush=True)
    selected, selection_mse, count, failures = search(train, config, problem, dest / "final")
    model = fit_model(train, selected, config, cols)
    save_json(dest / "model.json", model)
    predictions = predict(model, test)
    pred_dir = output.parent / "submission"
    pred_dir.mkdir(exist_ok=True, parents=True)
    filename = pred_dir / f"{roll}_pred_var{problem}.csv"
    pd.DataFrame({"y": predictions}).to_csv(filename, index=False)
    reloaded = json.loads((dest / "model.json").read_text())
    np.testing.assert_allclose(predict(reloaded, test), pd.read_csv(filename).y, rtol=1e-12, atol=1e-12)
    summary = dict(problem=problem, selected=selected, nonzero_terms=model["nonzero_terms"],
        total_terms=len(model["coef"]), final_selection_mse=selection_mse,
        final_search_candidates=count, final_invalid_candidates=failures,
        pooled_oof=metrics(train.y, oof),
        mean_outer_mse=float(np.mean([f["mse"] for f in summaries])),
        std_outer_mse=float(np.std([f["mse"] for f in summaries], ddof=1)),
        mean_outer_r2=float(np.mean([f["r2"] for f in summaries])),
        outer_folds=summaries, baselines={k: metrics(train.y,v) for k,v in baseline_oof.items()},
        training_fit=metrics(train.y, predict(model, train)),
        prediction_rows=len(predictions), prediction_min=float(predictions.min()),
        prediction_max=float(predictions.max()), elapsed_seconds=time.time()-start)
    save_json(dest / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--problems", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    with threadpool_limits(limits=1):
        for problem in args.problems:
            run(problem, config, args.data_dir, args.output)
