# Preparing a dataset

A dataset is three files plus one YAML config:

```
experiments/data/raster/
├── 2024NAIP_CojoTerrace_RGBIR.tif          # imagery
├── 2024NAIP_CojoTerrace_RGBIR_labels.shp   # hand drawn class polygons (+ .dbf/.shx/.prj)
└── 2024NAIP_CojoTerrace_RGBIR_grid.shp     # tiling of the scene (+ .dbf/.shx/.prj)

config/naip_2024.yaml                       # ties the three together
```

Nothing under `experiments/` is tracked by git; it is the working directory for
imagery, labels and training runs. The paths inside a config are resolved
relative to the directory you launch `train.py` from, i.e. the repository root.

## 1. Imagery

* Format: GeoTIFF readable by `rasterio`.
* Data type: `uint8`. Pixel values are divided by 255 and normalized with
  ImageNet statistics, so 16-bit imagery has to be rendered/stretched to 8 bit
  first (`gdal_translate -ot Byte -scale ...`).
* Bands: any number. The config selects which ones to feed the model, e.g. the
  four RGBIR bands of a NAIP scene that also carries an alpha band.
* CRS: a projected CRS in metres (the reference datasets use EPSG:26910,
  UTM 10N). The scene must be north-up (no rotation in the geotransform).

Check a scene with:

```bash
gdalinfo experiments/data/raster/2024NAIP_CojoTerrace_RGBIR.tif | head -20
```

The reference scene is 5019 x 5919 px, 5 bands (R, G, B, NIR, alpha), `uint8`,
0.6 m resolution, EPSG:26910.

## 2. Label shapefile

Polygons drawn over the imagery, typically in QGIS.

| requirement | detail |
| --- | --- |
| geometry | `Polygon` or `MultiPolygon` |
| attribute | a text field named **`class`** |
| CRS | any — it is reprojected to the raster CRS automatically |

The values of `class` are free text; the dataset config maps them to integer
class ids. In the reference labels the values are `Iceplant`,
`Mixed Iceplant`, `Other Vegetation`, `Tree` and `Others`.

Labels do **not** have to cover the whole scene. Every pixel that no polygon
touches falls back to class id 0, which is why class 0 is the "background"
class of the project (`ovg`, other vegetation). Concretely, in a grid cell that
you have marked as labeled, anything you did not draw is treated as class 0 —
so only mark a cell as labeled once everything that is *not* class 0 inside it
has been drawn.

Polygons are burned into a label raster with `all_touched=True`, i.e. every
pixel the polygon boundary touches gets the class, which slightly grows thin
features rather than dropping them.

## 3. Grid shapefile

The grid tiles the scene into rectangular cells and drives two things: which
parts of the scene are used at all, and which cells are held out for
validation.

| field | type | required | meaning |
| --- | --- | --- | --- |
| `Index` | integer | yes | cell id, referenced by `val_grid_ids` in the config |
| `Labeled` | integer | yes | `1` = the cell is fully labeled and may be used, anything else = ignored |
| `Group` | integer | no | free grouping tag, carried through as sample metadata |
| `row_index`, `col_index` | integer | no | cell position, carried through as sample metadata |

Requirements:

* Cells are axis-aligned rectangles (only their bounding box is used).
* The grid must be in **the same CRS as the raster** — unlike the labels it is
  not reprojected.
* `Index` does not have to be unique or contiguous. If two cells share an
  `Index`, both follow that id into the train or the validation split.

A regular grid is easiest to create with QGIS (*Vector ▸ Research Tools ▸
Create Grid*, rectangle type, extent of the raster, cell size a few hundred
metres), after which you add the `Index` and `Labeled` fields and fill them in
while labeling. The reference 2024 NAIP grid has 13 labeled cells of about
541 x 535 px (325 x 321 m) each.

## 4. Dataset config

```yaml
# config/naip_2024.yaml
name: naip-rgbir-2024      # free-form label, for your own reference only
band:                      # 1-based band indices to read from the raster
  - 1
  - 2
  - 3
  - 4
data:                      # one [raster, labels, grid] triple per scene
  - [
      "experiments/data/raster/2024NAIP_CojoTerrace_RGBIR.tif",
      "experiments/data/raster/2024NAIP_CojoTerrace_RGBIR_labels.shp",
      "experiments/data/raster/2024NAIP_CojoTerrace_RGBIR_grid.shp",
    ]
class_mapping:             # label `class` value -> class id
  Iceplant: 1
  Mixed Iceplant: 1
  Other Vegetation: 0
  Tree: 0
  Others: 2
patch_size: 128            # training patch size in pixels
stride: 32                 # training patch stride in pixels
val_grid_ids: [49, 56]     # `Index` values held out for validation
```

Field by field:

* **`band`** — the number of entries is the number of input channels of the
  model. `[1, 2, 3, 4]` is RGB+NIR, `[1, 2, 3]` RGB only, `[1, 2, 4]`
  red/green/NIR. A checkpoint can only be used on imagery with the same number
  of channels.
* **`data`** — several triples can be listed in one config; their patches are
  concatenated. Use this for scenes that share the same class mapping and band
  layout.
* **`class_mapping`** — maps the text in the shapefile `class` field to an
  integer id. Ids must start at 0 and be contiguous; any value not listed here
  (and every unlabeled pixel) becomes 0. The three ids used by the reference
  configs correspond to `--class "ovg,iceplant,others"` on the training command
  line: 0 = other vegetation, 1 = iceplant, 2 = everything else.
* **`patch_size`** — 128 for all reference configs. It has to be a multiple of
  32 for the U-Net decoder to line up with the encoder feature maps.
* **`stride`** — how densely training patches are sampled inside a cell.
  Smaller stride = more, more strongly overlapping patches. Validation patches
  always use a stride equal to `patch_size`, so they never overlap.
* **`val_grid_ids`** — cells held out for validation; every other labeled cell
  is used for training. Hold out cells that are representative of the scene,
  not a corner of it.

## 5. How patches are produced

For every labeled cell, the extraction walks a `stride`-spaced lattice starting
at the cell's top-left corner and keeps every `patch_size` window that lies
fully inside the raster. Windows near the right/bottom edge of a cell may
extend into the neighbouring cell — the cell defines where sampling *starts*,
not a hard clip. Patch counts therefore follow
`ceil(cell_size / stride)` per axis.

With the reference 2024 NAIP config: 13 labeled cells, 3 of them held out
(`Index` 49 appears twice in that grid), cells of 541 x 535 px:

* train: 10 cells x 17 x 17 patches (stride 32) = **2890 patches**
* val: 3 cells x 5 x 5 patches (stride 128) = **75 patches**

`train.py` prints exactly these numbers when it loads a config, which is the
quickest way to check a new dataset:

```
Loaded dataset from config/naip_2024.yaml
  Number of training samples: 2890
  Number of validation samples: 75
```

Zero samples means the grid and the raster do not overlap, no cell has
`Labeled == 1`, or `val_grid_ids` matches no `Index`.

## 6. Adding a new scene

1. Put the GeoTIFF, the label shapefile and the grid shapefile in
   `experiments/data/raster/`.
2. Copy an existing config, point `data` at the three new files, and adjust
   `band`, `class_mapping` and `val_grid_ids`.
3. Sanity check the config:

   ```bash
   python train.py --exp-path check --dataset config/my_scene.yaml \
       --class "ovg,iceplant,others" --backbone unet-resnet18 \
       --batch-size 4 --workers 0 --debug
   rm -rf experiments/models/check
   ```

4. Train for real, see [training](training.md).
