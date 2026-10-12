# Accuracy assessment and area estimation of the annual iceplant maps

Code and reference data for the accuracy assessment of the annual three-class iceplant maps
(2020, 2022, 2023, 2024, 2025) of the Jack and Laura Dangermond Preserve.
Accuracies and areas are estimated with the stratified estimators of
Olofsson et al. (2014), with 95% confidence intervals.

Map classes (raster values): **1 = iceplant, 2 = others** (impervious surfaces, water, roads, bare sand),
**3 = other vegetation**. These are the values of the classification rasters that were assessed,
not the training class ids of the segmentation model described in the main README.

## Workflow

1. **Pixel counts per class.** The number of pixels of each class is counted in the classification
   map of each year (`{year}_rawmodel.tif`, 0.6 m pixels). These counts are the sampling strata and
   the area weights of the estimators (`data/pixel_counts.csv`).
2. **Sample size.** The total sample size follows Cochran (1977) for stratified random sampling,
   with a target standard error of 0.0105 for overall accuracy and expected user's accuracies of
   0.88 (iceplant), 0.94 (others) and 0.91 (other vegetation).
3. **Allocation.** Proportional allocation would give too few points to the iceplant class, so a
   disproportional design is used: a fixed 300 points each for iceplant and other vegetation, and
   the remainder for others.
4. **Stratified random sampling.** Points are drawn at random within each map class
   (`notebooks/01_stratified_sampling.ipynb`).
5. **Reference labels.** Each point is visually interpreted in QGIS against independent sub-meter
   reference imagery, without knowledge of the map class.
6. **Accuracy and area estimation.** Error matrices, user's, producer's and overall accuracies and
   error-adjusted areas are computed for each year
   (`notebooks/02_accuracy_assessment_{year}.ipynb`).

| Year | Points | Iceplant | Others | Other vegetation |
|------|-------:|---------:|-------:|-----------------:|
| 2020 | 716 | 300 | 116 | 300 |
| 2022 | 712 | 300 | 112 | 300 |
| 2023 | 704 | 300 | 104 | 300 |
| 2024 | 704 | 300 | 104 | 300 |
| 2025 | 712 | 300 | 112 | 300 |

## Repository layout

```
data/
  pixel_counts.csv              pixels per map class and pixel size, by year
  validation_points_{year}.csv  reference points: PLOTID, LON, LAT (WGS84), map_class, ref_class
notebooks/
  01_stratified_sampling.ipynb        pixel counts, sample size, allocation and point sampling
  02_accuracy_assessment_{year}.ipynb error matrix, accuracies and area estimates for one year
results/
  accuracy_{year}.csv           summary table written by each accuracy notebook
```

## Running

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 02_accuracy_assessment_2020.ipynb
```

The accuracy notebooks only need the files in `data/`. The sampling notebook needs the
classification rasters (`{year}_rawmodel.tif`), which are not included because of their size; place
them in `notebooks/` to run it.

## Results

Overall accuracy (OA), and user's (UA) and producer's (PA) accuracy and error-adjusted area of the
iceplant class (± 95% confidence interval). Results for all classes are in `results/`.

| Year | OA (%) | UA iceplant (%) | PA iceplant (%) | Iceplant area (ha) | Iceplant share (%) |
|------|--------|-----------------|-----------------|--------------------|--------------------|
| 2020 | 91.43 ± 2.19 | 95.33 ± 2.39 | 81.82 ± 5.13 | 287.8 ± 18.9 | 26.91 ± 1.77 |
| 2022 | 94.63 ± 1.63 | 94.00 ± 2.69 | 84.64 ± 5.13 | 260.0 ± 16.9 | 24.29 ± 1.58 |
| 2023 | 95.44 ± 1.86 | 97.67 ± 1.71 | 90.54 ± 5.11 | 222.1 ± 13.0 | 20.76 ± 1.22 |
| 2024 | 93.48 ± 1.97 | 94.33 ± 2.62 | 82.94 ± 6.18 | 211.3 ± 16.5 | 19.76 ± 1.54 |
| 2025 | 91.99 ± 2.09 | 97.33 ± 1.83 | 74.20 ± 5.44 | 295.3 ± 22.0 | 27.58 ± 2.05 |

## Notes

- The sampling notebook draws points without a fixed random seed, so re-running it produces a new
  sample. The samples used for the assessment are the ones in `data/`.

## Reference

Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E. & Wulder, M. A. Good practices
for estimating area and assessing accuracy of land change. *Remote Sensing of Environment* 148,
42–57 (2014). https://doi.org/10.1016/j.rse.2014.02.015
