"""Pinned source and target-independent listing features."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from py7zr import SevenZipFile
from py7zr.io import BytesIOFactory

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://archive.ics.uci.edu/static/public/555/apartment+for+rent+classified.zip'
BASE = 'apartments_for_rent_classified_10K'
ARCHIVE_HASH = '0006eb37dadab31c764b6fe3c8434760c812f633926bae5cb5c0725a79d1c05d'
CSV_HASH = '522df3a6170cdd83adf6ba1fe5a0c9143fbdb23bc36d671fbd870916727202fd'
CELL = 0.02
CATS = ['market', 'state', 'has_photo', 'pets_allowed', 'fee', 'source']
AMENITIES = ('parking', 'pool', 'gym', 'dishwasher', 'washer dryer', 'ac', 'patio/deck', 'storage')


def load_raw(path=ROOT / 'data' / 'apartments.7z'):
    cached = path.exists()
    if cached:
        compressed = path.read_bytes()
    else:
        with urlopen(SOURCE, timeout=60) as response, ZipFile(BytesIO(response.read())) as bundle:
            compressed = bundle.read(BASE + '.7z')
    if sha256(compressed).hexdigest() != ARCHIVE_HASH:
        raise ValueError('Unexpected UCI 10K archive digest.')
    factory = BytesIOFactory(20_000_000)
    name = BASE + '.csv'
    with SevenZipFile(BytesIO(compressed)) as archive:
        archive.extract(targets=[name], factory=factory)
    item = factory.get(name)
    item.seek(0)
    raw = item.read()
    if sha256(raw).hexdigest() != CSV_HASH:
        raise ValueError('Unexpected UCI CSV digest.')
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compressed)
    return pd.read_csv(BytesIO(raw), sep=';', encoding='cp1252')


def clean_listings(raw):
    monthly = raw.currency.eq('USD') & raw.price_type.eq('Monthly') & raw.category.eq('housing/rent/apartment')
    finite = np.isfinite(raw.price) & np.isfinite(raw.square_feet) & (raw.price > 0) & (raw.square_feet > 0)
    eligible = raw.loc[monthly & finite & raw.id.notna()]
    clean = eligible.drop_duplicates('id').reset_index(drop=True)
    audit = {'raw': len(raw), 'ineligible': len(raw) - len(eligible),
             'duplicate_id': len(eligible) - len(clean), 'retained': len(clean), 'excluded': len(raw)-len(clean)}
    return clean, audit


def category(values):
    return values.fillna('(missing)').astype(str).str.strip().str.casefold().replace('', '(missing)')


def features_and_groups(rows):
    out = rows[['square_feet', 'bedrooms', 'bathrooms', 'latitude', 'longitude']].astype(float).copy()
    coord = out.latitude.between(-90, 90) & out.longitude.between(-180, 180)
    out.loc[~coord, ['latitude', 'longitude']] = np.nan
    out['log_sqft'] = np.log1p(out.square_feet)
    out['bed_bath'] = out.bedrooms * out.bathrooms
    amenity = rows.amenities.fillna('').astype(str).apply(lambda x: {s.strip().lower() for s in x.split(',') if s.strip()})
    out['amenity_count'] = amenity.str.len()
    out['amenities_missing'] = amenity.str.len().eq(0).astype(int)
    for token in AMENITIES:
        safe_name = token.replace(' ', '_').replace('/', '_')
        out[f'amenity_{safe_name}'] = amenity.map(lambda tags: int(token in tags))
    for col in CATS[1:]:
        out[col] = category(rows[col])
    out['market'] = category(rows.cityname) + ', ' + out.state
    groups = 'missing:' + out.market
    lat_grid = np.floor(out.loc[coord, 'latitude'] / CELL).astype(int).astype(str)
    lon_grid = np.floor(out.loc[coord, 'longitude'] / CELL).astype(int).astype(str)
    groups.loc[coord] = lat_grid + ':' + lon_grid
    return out, groups.rename('geo_cell')
