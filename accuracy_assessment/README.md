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

## Notes

- The sampling notebook draws points without a fixed random seed, so re-running it produces a new
  sample. The samples used for the assessment are the ones in `data/`.

## Reference

Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E. & Wulder, M. A. Good practices
for estimating area and assessing accuracy of land change. *Remote Sensing of Environment* 148,
42–57 (2014). https://doi.org/10.1016/j.rse.2014.02.015
