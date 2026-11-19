import unittest

import numpy as np
import pandas as pd

import data as d
import study as s


def fixture(n=40):
    i = np.arange(n)
    return pd.DataFrame(dict(id=i, category='housing/rent/apartment', currency='USD',
        price_type='Monthly', price=800. + i * 40, square_feet=600. + i * 10,
        latitude=30.002 + 0.025 * (i // 2), longitude=-90.013, cityname='A', state='CO',
        bedrooms=1., bathrooms=1., amenities='Pool,Parking', pets_allowed='Cats',
        has_photo='Yes', fee='No', source='Site', time=1577359090,
        price_display='$4321', title='monthly $4321', body='monthly $4321', address='ignored'))


class DataTests(unittest.TestCase):
    def test_filters_to_monthly_usd_apartments_and_unique_ids(self):
        x = fixture(8)
        x.loc[0, 'price_type'] = 'Monthly|Weekly'
        x.loc[1, 'price'] = 0
        x.loc[2, 'square_feet'] = -10
        x.loc[3, 'currency'] = 'CAD'
        x.loc[4, 'category'] = 'housing/rent/home'
        x.loc[5, 'id'] = 6
        clean, audit = d.clean_listings(x)
        self.assertEqual(clean.id.tolist(), [6, 7])
        self.assertEqual(audit['excluded'], 6)

    def test_price_and_text_do_not_change_features(self):
        x = fixture()
        a, groups = d.features_and_groups(x)
        x[['price', 'title', 'body', 'price_display', 'id', 'time']] = -999
        b, group_b = d.features_and_groups(x)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_series_equal(groups, group_b)
        self.assertEqual(a.iloc[0].amenity_count, 2)
        self.assertEqual(a.iloc[0].amenity_pool, 1)

    def test_amenities_missing_and_mask_partial_coordinates(self):
        x = fixture(2)
        x.loc[0, 'amenities'] = None
        x.loc[0, 'latitude'] = np.nan
        a, groups = d.features_and_groups(x)
        self.assertEqual(a.iloc[0].amenity_count, 0)
        self.assertEqual(a.iloc[0].amenities_missing, 1)
        self.assertTrue(np.isnan(a.iloc[0].longitude))
        self.assertTrue(groups.iloc[0].startswith('missing:'))

    def test_geographic_blocks_are_disjoint_on_every_split(self):
        X, groups = d.features_and_groups(fixture())
        dev, test, folds = s.split_groups(X, groups)
        self.assertFalse(set(groups.iloc[dev]) & set(groups.iloc[test]))
        valid = []
        for a, b in folds:
            self.assertFalse(set(groups.iloc[dev[a]]) & set(groups.iloc[dev[b]]))
            valid.extend(b.tolist())
        self.assertEqual(sorted(valid), list(range(len(dev))))

    def test_training_comps_exclude_own_cell_but_scoring_comps_use_train(self):
        X, groups = d.features_and_groups(fixture())
        y = fixture().price
        actual = s.add_comps(X, y, groups, X, exclude_own=True)
        changed = y.copy()
        same = groups.eq(groups.iloc[0]); changed.loc[same] += 1500
        other = s.add_comps(X, changed, groups, X, exclude_own=True)
        pd.testing.assert_frame_equal(actual.loc[same, s.COMP_COLUMNS], other.loc[same, s.COMP_COLUMNS])
        self.assertNotEqual(actual.loc[~same, 'comp_city'].iloc[0], other.loc[~same, 'comp_city'].iloc[0])
        unseen = X.iloc[:1].assign(market='different', state='different')
        scored = s.add_comps(X, y, groups, unseen)
        self.assertAlmostEqual(scored.iloc[0].comp_city, y.mean())

    def test_holdout_label_changes_leave_all_fitted_features_unchanged(self):
        raw = fixture()
        X, groups = d.features_and_groups(raw)
        dev, test, _ = s.split_groups(X, groups)
        first = s.prepare(X, raw.price, groups, dev, test)
        altered = raw.price.copy()
        altered.iloc[test] += 2000
        second = s.prepare(X, altered, groups, dev, test)
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_series_equal(first[1], second[1])
        pd.testing.assert_frame_equal(first[2], second[2])
        self.assertTrue((first[3] != second[3]).all())

    def test_metric_is_dollar_rmse_and_residual_is_prediction_minus_actual(self):
        scores = s.metrics(np.array([1000., 1000.]), np.array([1200., 600.]))
        self.assertAlmostEqual(scores['rmse'], np.sqrt(100000))
        self.assertAlmostEqual(scores['mae'], 300.)
        self.assertAlmostEqual(scores['bias'], -100.)

    def test_search_budget_and_fixed_reference(self):
        cb, rf = s.search_space()
        self.assertEqual(len(cb), 9)
        self.assertEqual(len(rf), 6)
        self.assertEqual(len({tuple(p.items()) for p in cb}), 9)
        self.assertIn(s.REFERENCE_TREES, s.CHECKPOINTS)

    def test_shuffled_label_order_is_rejected_for_target_encoding(self):
        X, groups = d.features_and_groups(fixture())
        with self.assertRaisesRegex(ValueError, 'aligned'):
            s.add_comps(X, fixture().price.iloc[::-1], groups, X, exclude_own=True)
