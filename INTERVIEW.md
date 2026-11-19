# Interview walkthrough

## Start with the honest distinction

The competition entry was your R project. This is an AI-assisted Python follow-up on a public UCI snapshot. The two evaluations cannot be compared as if they were the same leaderboard. Be specific about the Python code and results you have actually rerun or inspected.

## 45-second version

> My R competition project used local rent averages to give a forest more location context. The follow-up kept that idea and tested it more carefully. In the original code those averages were calculated before validation, which allowed validation rents to bleed into the features. In Python, the follow-up split listings by geographic cell and built comparable rents from training data only, leaving each training row’s own cell out too. It tuned CatBoost and a random forest, compared them with a median and Ridge, then checked a separate holdout. The tuned forest won on RMSE: about $621 versus $627 for CatBoost on 1,848 held-out listings. CatBoost had lower MAE, and the RMSE gap was too uncertain to call a clear win. The biggest misses were on expensive listings.

## Questions an interviewer can ask

**How did the original feature leak?** A validation listing worth $3,000 might contribute to its city’s average. If that city has only one listing, the feature can effectively carry the answer. In the Python version, a validation listing’s rent cannot change its features: the aggregates use only other cells assigned to training. A training row’s comparables exclude its own cell too.

**Why group by geography instead of shuffling rows?** Listings in the same area can be very similar. Splitting by cell stops a single cell from appearing on both sides of each split. It does not prove transfer to new cities or eliminate all nearby duplicates across cell boundaries.

**What did you tune, exactly?** CatBoost adds trees one after another to improve previous predictions. The search tested how deep the trees could be, how big each update was, how strongly leaf values were penalized, how much randomness was used in split selection, and how many trees were used. The forest search tested how many features each split could consider and how many examples were required in a leaf. Both searches used the same four grouped folds.

**Why use both RMSE and MAE?** They are both measured in dollars per month. MAE is the average miss. RMSE penalizes big misses more, so it was the selection metric. A single expensive outlier can move RMSE sharply; the price-band plot shows who is most affected.

**What does removing a feature tell you?** Compare the error with and without the comparable-rent columns while holding the selected CatBoost settings fixed. It estimates whether those columns add information beyond coordinates and categories in this experiment. It is not a causal study.

**What is SHAP?** In this report, it helps explain which features moved CatBoost predictions away from their baseline. It does not prove an amenity changes market rent. Correlated variables can share credit.

**What would you do next?** Test a more recent snapshot with entire cities held out; check whether similar listings cross grid boundaries. A model for historical advertised rents should not be treated as a production tool for setting prices.

## One thing to practice

Open `study.py`, find `add_comps`, and explain how the subtraction prevents the rent of a near-neighbor in the same cell from entering that row’s feature. Then open `outputs/summary.json` and explain the validation and holdout scores separately.
