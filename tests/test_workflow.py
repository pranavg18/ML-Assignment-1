"""Numerical and submission-contract checks using the standard library runner"""
import json
import hashlib
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from polynomial import powers, design, predict, load_data
from train import fit_model
from sklearn.metrics import mean_squared_error, r2_score


class WorkflowTests(unittest.TestCase):
    def test_total_degree_and_feature_values(self):
        x = np.random.default_rng(86).normal(size=(10, 3))
        exponents = powers(3, 4)
        self.assertEqual(exponents.shape, (34, 3))
        self.assertTrue(np.all(exponents.sum(axis=1) <= 4))
        np.testing.assert_allclose(design(x, exponents),
            PolynomialFeatures(4, include_bias=False).fit_transform(x))

    def test_ridge_matches_independent_estimator(self):
        rng = np.random.default_rng(86)
        frame = pd.DataFrame(rng.normal(size=(70, 3)), columns=['x1','x2','x3'])
        frame['y'] = 1.3 + frame.x1**3 - 2*frame.x2*frame.x3 + rng.normal(size=70)*0.1
        train, valid = frame.iloc[:50], frame.iloc[50:]
        c = dict(features=['x1','x2','x3'], degree=3, model='ridge', alpha=0.13, l1_ratio=None)
        model = fit_model(train, c, {}, c['features'])
        poly = PolynomialFeatures(3, include_bias=False)
        scaler = StandardScaler()
        z = scaler.fit_transform(poly.fit_transform(train[c['features']]))
        reg = Ridge(alpha=0.13, solver='svd').fit(z, train.y)
        expected = reg.predict(scaler.transform(poly.transform(valid[c['features']])))
        np.testing.assert_allclose(predict(model, valid), expected, atol=1e-10)
        # Serialization must preserve inference without pickle or training data.
        np.testing.assert_allclose(predict(json.loads(json.dumps(model)), valid), expected, atol=1e-10)

    def test_scaler_uses_only_supplied_training_rows(self):
        frame = pd.DataFrame({'x1':[0.,1.,2.], 'y':[1.,3.,5.]})
        c = dict(features=['x1'], degree=1, model='ols', alpha=0., l1_ratio=None)
        model = fit_model(frame, c, {}, ['x1'])
        self.assertEqual(model['term_mean'], [1.])
        far_away = pd.DataFrame({'x1':[100.]})
        np.testing.assert_allclose(predict(model, far_away), [201.])
        self.assertEqual(model['term_mean'], [1.])

    def test_degree_weighted_ridge_and_adaptive_lasso(self):
        rng = np.random.default_rng(321)
        frame = pd.DataFrame(rng.uniform(-1, 1, (90, 3)), columns=['x1','x2','x3'])
        frame['y'] = 2 + frame.x1**3 - frame.x2*frame.x3 + rng.normal(0, .1, 90)
        fit, valid = frame.iloc[:70], frame.iloc[70:]
        cols = ['x1','x2','x3']; exponents = powers(3, 3)
        scaler = StandardScaler().fit(design(fit[cols].values, exponents))
        z = scaler.transform(design(fit[cols].values, exponents))
        zv = scaler.transform(design(valid[cols].values, exponents))
        yc = fit.y.values-fit.y.mean()
        c = dict(features=cols, degree=3, model='ridge', alpha=.3, l1_ratio=None, degree_penalty=2)
        w = (exponents.sum(axis=1)/3)**(-2.)
        expected = Ridge(alpha=.3, fit_intercept=False, solver='svd').fit(z*w, yc).predict(zv*w)+fit.y.mean()
        np.testing.assert_allclose(predict(fit_model(fit,c,{},cols), valid), expected, atol=1e-9)
        cfg = dict(sparse_max_iter=100000, sparse_retry_iter=500000, sparse_tol=1e-8)
        c = dict(features=cols, degree=3, model='adaptive_lasso', alpha=.001,
                 l1_ratio=1., initial_alpha=.003, adaptive_power=1.)
        init = Lasso(alpha=.003, fit_intercept=False, max_iter=100000, tol=1e-8).fit(z,yc)
        w = np.maximum(abs(init.coef_), .01)
        est = Lasso(alpha=.001, fit_intercept=False, max_iter=100000, tol=1e-8).fit(z*w,yc)
        expected = est.predict(zv*w)+fit.y.mean()
        model = fit_model(fit,c,cfg,cols)
        np.testing.assert_allclose(predict(json.loads(json.dumps(model)), valid), expected, atol=1e-7)

    def test_saved_submission_contract(self):
        for problem in [1,2]:
            model_path = Path(f'results/var{problem}/model.json')
            if not model_path.exists():
                self.skipTest('Training has not finished yet')
            model = json.loads(model_path.read_text())
            self.assertLessEqual(model['degree'], 10 if problem == 1 else 20)
            self.assertTrue(np.all(np.asarray(model['powers']).sum(axis=1) <= model['degree']))
            frame = load_data(f'data/BT2024086_test_var{problem}.csv', model['input_columns'])
            result = pd.read_csv(f'submission/BT2024086_pred_var{problem}.csv')
            self.assertEqual(list(result.columns), ['y'])
            self.assertEqual(len(result), len(frame))
            self.assertTrue(np.isfinite(result.y).all())
            np.testing.assert_allclose(result.y, predict(model, frame), rtol=1e-12, atol=1e-12)

    def test_polynomial_average_is_exactly_collapsed(self):
        rng = np.random.default_rng(12)
        frame = pd.DataFrame(rng.uniform(-1,1,(70,3)), columns=['x1','x2','x3'])
        frame['y'] = 1 + frame.x1**3 - frame.x2*frame.x3 + rng.normal(0,.1,70)
        cols = ['x1','x2','x3']; fit=frame.iloc[:50]; valid=frame.iloc[50:]
        members=[dict(features=cols, degree=d, model='ridge', alpha=a, l1_ratio=None)
                 for d,a in [(2,.1),(3,1),(4,10)]]
        c=dict(features=cols, degree=4, model='polynomial_average', alpha=None,
               l1_ratio=None, averaging_rule='test', members=members)
        expected=np.mean([predict(fit_model(fit,m,{},cols),valid) for m in members],axis=0)
        collapsed=fit_model(fit,c,{},cols)
        self.assertEqual(len(collapsed['powers']), len(powers(3,4)))
        np.testing.assert_allclose(predict(json.loads(json.dumps(collapsed)),valid),expected,atol=1e-12)

    def test_audit_trail_matches_actual_outputs(self):
        for problem in [1,2]:
            directory = Path(f'results/var{problem}')
            if not (directory/'summary.json').exists():
                self.skipTest('Training has not finished yet')
            summary = json.loads((directory/'summary.json').read_text())
            manifest = json.loads((directory/'manifest.json').read_text())
            self.assertEqual(manifest['config'], json.loads(Path('config.json').read_text()))
            saved_model = json.loads((directory/'model.json').read_text())
            for key, value in summary['selected'].items():
                self.assertEqual(saved_model[key], value)
            for name, expected_hash in manifest['sha256'].items():
                # Code paths in the manifest may be absolute on the original host.
                path = Path(name)
                if not path.exists():
                    path = Path(path.name)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash)
            oof = pd.read_csv(directory/'oof_predictions.csv')
            train = pd.read_csv(f'data/BT2024086_train_var{problem}.csv')
            np.testing.assert_array_equal(oof.row, np.arange(len(train)))
            np.testing.assert_allclose(oof.y, train.y)
            self.assertEqual(set(oof.fold), {1,2,3,4,5})
            self.assertAlmostEqual(mean_squared_error(oof.y,oof.prediction), summary['pooled_oof']['mse'], places=12)
            self.assertAlmostEqual(r2_score(oof.y,oof.prediction), summary['pooled_oof']['r2'], places=12)


if __name__ == '__main__':
    unittest.main()
