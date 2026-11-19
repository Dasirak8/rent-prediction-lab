"""Score new listings with the fitted historical CatBoost benchmark."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from data import ROOT, clean_listings, features_and_groups, load_raw
from study import THREADS, add_comps, predict, split_groups

NUMERIC = ['square_feet', 'bedrooms', 'bathrooms', 'latitude', 'longitude']
CATEGORICAL = ['cityname', 'state', 'amenities', 'pets_allowed', 'has_photo', 'fee', 'source']


def listing_input(listings):
    if listings.empty or 'square_feet' not in listings:
        raise ValueError('Supply at least one listing with positive square_feet.')
    result = listings.reindex(columns=NUMERIC + CATEGORICAL).copy()
    for col in NUMERIC:
        number = pd.to_numeric(result[col], errors='coerce')
        result[col] = number.where(np.isfinite(number))
    if not result.square_feet.gt(0).all():
        raise ValueError('Every square_feet value must be positive and finite.')
    return result


def score(listings):
    path = ROOT / 'data' / 'catboost.cbm'
    if not path.is_file():
        raise ValueError('Run python study.py first to fit the local historical model.')
    examples, _ = features_and_groups(listing_input(listings))
    historical, _ = clean_listings(load_raw())
    features, groups = features_and_groups(historical)
    dev, _, _ = split_groups(features, groups)
    transformed = add_comps(features.iloc[dev], historical.price.iloc[dev], groups.iloc[dev], examples)
    model = CatBoostRegressor().load_model(str(path))
    prices = predict(model, transformed, thread_count=THREADS)
    return pd.DataFrame({'listing_number': np.arange(1, len(listings) + 1),
                         'predicted_historical_asking_rent_usd': prices.round(2)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('listings_csv', type=Path)
    args = parser.parse_args()
    try:
        score(pd.read_csv(args.listings_csv)).to_csv(sys.stdout, index=False, lineterminator='\n')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
