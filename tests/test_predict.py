import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from predict import listing_input


class NewListingTests(unittest.TestCase):
    def test_minimal_profile_is_filled_with_missing_optional_values(self):
        raw = pd.DataFrame({'square_feet': [800], 'cityname': ['Albany'], 'state': ['NY']})
        result = listing_input(raw)
        self.assertEqual(result.square_feet.tolist(), [800])
        self.assertTrue(result.bedrooms.isna().all())
        self.assertTrue(result.latitude.isna().all())

    def test_rejects_bad_size_instead_of_scoring_it(self):
        for bad in [0, -10, 'bad', float('inf')]:
            with self.assertRaisesRegex(ValueError, 'positive'):
                listing_input(pd.DataFrame({'square_feet':[bad]}))

    def test_does_not_use_actual_rents_or_listing_descriptions(self):
        raw = pd.DataFrame({'square_feet': [950], 'cityname': ['Denver']})
        modified = raw.assign(price=3999, price_display='$3999', body='This costs $3999')
        assert_frame_equal(listing_input(raw), listing_input(modified))
