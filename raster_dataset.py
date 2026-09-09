import yaml
import numpy as np
import torch
import rasterio
import geopandas as gpd
from rasterio.windows import Window
from rasterio.features import rasterize
from torch.utils.data import Dataset


class RasterDataset(Dataset):
    def __init__(self, config_path, split="train", transforms=None, cache=True):
        """
        Initialize the dataset.

        Args:
            config_path (str): Path to the dataset configuration YAML file.
            split (str): Dataset split, either 'train' or 'val'.
            transforms (callable, optional): Transforms to apply to the data samples.
            cache (bool): If True, load all raster data into memory during initialization.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.cache = cache
        self.cached_rasters = {}

        self.split = split
        self.patch_size = self.config["patch_size"]
        self.stride = (
            self.config["stride"] if split == "train" else self.config["patch_size"]
        )
        self.class_mapping = self.config["class_mapping"]
        self.val_grid_ids = self.config["val_grid_ids"]
        self.bands = self.config["band"]
        self.transforms = transforms

        # Prepare data samples
        self.samples = self._prepare_samples()

    def _get_class_id(self, class_name):
        """Maps a class name string to its corresponding integer ID."""
        return self.class_mapping.get(class_name, 0)  # Default to Other Vegetation (0)

    def _rasterize_labels(self, image_path, shapefile_path):
        """
        Rasterize the vector shapefile to match the source image grid.
        """
        with rasterio.open(image_path) as src_image:
            labels = gpd.read_file(shapefile_path)

            # Ensure label CRS matches image CRS
            if labels.crs != src_image.crs:
                labels = labels.to_crs(src_image.crs)

            # Prepare shapes for rasterization
            shapes = [
                (geom, self._get_class_id(class_name))
                for geom, class_name in zip(labels.geometry, labels["class"])
            ]

            # Rasterize shapes
            rasterized_data = rasterize(
                shapes=shapes,
                out_shape=(src_image.height, src_image.width),
                transform=src_image.transform,
                fill=0,  # Default value is 0 (Other Vegetation)
                all_touched=True,
                dtype=rasterio.uint8,
            )

        return rasterized_data

    def _extract_patches_from_grid(self, image_path, label_path, grid_path):
        """
        Extract image-label patch coordinates based on grid.
        """
        # Open image and rasterize labels
        with rasterio.open(image_path) as image:
            rasterized_labels = self._rasterize_labels(image_path, label_path)

            # Read grid
            grid = gpd.read_file(grid_path)

            # Filter grid based on split
            if self.split == "train":
                grid = grid[~grid["Index"].isin(self.val_grid_ids)]
            else:  # val
                grid = grid[grid["Index"].isin(self.val_grid_ids)]

            # Filter grid to only include labeled cells (Labeled == 1)
            grid = grid[grid["Labeled"] == 1]

            patches = []
            # Process each cell in the filtered grid
            for _, row in grid.iterrows():
                minx, miny, maxx, maxy = row.geometry.bounds
                group = int(row["Group"]) if "Group" in row else 0
                row_idx = int(row["row_index"]) if "row_index" in row else 0
                col_idx = int(row["col_index"]) if "col_index" in row else 0
                grid_index = int(row["Index"])

                # Define the starting point in pixel coordinates for this grid cell
                start_col = int((minx - image.bounds.left) / image.transform.a)
                start_row = int((image.bounds.top - maxy) / abs(image.transform.e))

                # Calculate width and height of the grid cell in pixels
                cell_width = int((maxx - minx) / image.transform.a)
                cell_height = int((maxy - miny) / abs(image.transform.e))

                # Iterate through the grid cell with the given stride to extract patches
                patch_num = 1
                for y_offset in range(0, cell_height, self.stride):
                    for x_offset in range(0, cell_width, self.stride):
                        # Calculate window origin
                        col_off = start_col + x_offset
                        row_off = start_row + y_offset

                        # Create the patch window
                        window = Window(
                            col_off, row_off, self.patch_size, self.patch_size
                        )

                        # Ensure the window is within the image bounds
                        if (
                            window.col_off < 0
                            or window.row_off < 0
                            or window.col_off + window.width > image.width
                            or window.row_off + window.height > image.height
                        ):
                            continue

                        patches.append(
                            {
                                "image_path": image_path,
                                "window": window,
                                "label_data": rasterized_labels[
                                    window.row_off : window.row_off + window.height,
                                    window.col_off : window.col_off + window.width,
                                ],
                                "group": group,
                                "row_idx": row_idx,
                                "col_idx": col_idx,
                                "patch_num": patch_num,
                                "grid_index": grid_index,
                            }
                        )

                        patch_num += 1

        return patches

    def _prepare_samples(self):
        """
        Prepare all samples for the dataset.
        """
        all_patches = []
        for data_entry in self.config["data"]:
            raster_path, label_path, grid_path = data_entry

            # Cache the raster data if enabled
            if self.cache:
                with rasterio.open(raster_path) as image:
                    # Read only specified bands for the entire raster
                    self.cached_rasters[raster_path] = {
                        "data": image.read(self.bands),
                        "transform": image.transform,
                        "width": image.width,
                        "height": image.height,
                    }

            patches = self._extract_patches_from_grid(
                raster_path, label_path, grid_path
            )
            all_patches.extend(patches)

        return all_patches

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image data, label data, and metadata.
        """
        sample = self.samples[idx]

        # Get image data for the window
        if self.cache:
            # Use cached data
            cached_raster = self.cached_rasters[sample["image_path"]]
            window = sample["window"]
            img_data = cached_raster["data"][
                :,
                window.row_off : window.row_off + window.height,
                window.col_off : window.col_off + window.width,
            ]
            img_data = np.moveaxis(img_data, 0, -1)
        else:
            # Read from disk
            with rasterio.open(sample["image_path"]) as image:
                # Read only specified bands
                img_data = image.read(self.bands, window=sample["window"])
                img_data = np.moveaxis(img_data, 0, -1)

        mask = sample["label_data"]

        # Apply transforms if they exist
        if self.transforms:
            transformed = self.transforms(image=img_data, mask=mask)
            img_data = transformed["image"]
            mask = transformed["mask"]
        else:
            img_data = torch.from_numpy(np.moveaxis(img_data, -1, 0).copy())
            mask = torch.from_numpy(mask.copy())

        # Prepare the sample dictionary
        output = {
            "image": img_data,
            "mask": mask.long(),
            "metadata": {
                "group": sample["group"],
                "row_idx": sample["row_idx"],
                "col_idx": sample["col_idx"],
                "patch_num": sample["patch_num"],
                "grid_index": sample["grid_index"],
            },
        }

        return output
