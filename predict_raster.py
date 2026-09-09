import os
import argparse
import numpy as np
import torch
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from tqdm import tqdm
from model import SegmModel
from train import normalize_imgnet


def parse_options():
    parser = argparse.ArgumentParser(description="Predict on a raster image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input raster file.",
    )
    parser.add_argument(
        "--bands",
        type=str,
        help="Comma-separated list of band indices to use (1-based indexing, e.g., '1,2,3')",
        default=None,
    )
    parser.add_argument(
        "--exp-path",
        type=str,
        required=True,
        help="Path to the PyTorch Lightning experiment folder (e.g., xxx/version_0).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint file to use inside <exp-path>/checkpoints.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=128, help="Size of the image patches."
    )
    parser.add_argument(
        "--stride", type=int, default=64, help="Stride for patch extraction."
    )
    return parser.parse_args()


def _window_offsets(length, patch_size, stride):
    """Sliding window origins covering [0, length), including the trailing edge."""
    if length < patch_size:
        raise ValueError(
            f"Raster side of {length} px is smaller than the patch size of {patch_size} px"
        )
    offsets = list(range(0, length - patch_size + 1, stride))
    if offsets[-1] != length - patch_size:
        offsets.append(length - patch_size)
    return offsets


def predict_raster(image_path, exp_path, patch_size, stride, checkpoint="last.ckpt",
                   bands=None):
    checkpoint_path = os.path.join(exp_path, "checkpoints", checkpoint)
    print("Using checkpoint:", checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegmModel.load_from_checkpoint(checkpoint_path, map_location=device)
    class_names = model.class_names
    model.eval()
    model.to(device)

    # Open the raster file
    with rasterio.open(image_path) as src:
        if bands is not None:
            band_indices = [int(b) for b in bands.split(",")]
            raster = src.read(band_indices)
            print(f"Using bands: {[i for i in band_indices]}")
        else:
            raster = src.read()
            print(f"Using all available bands: {list(range(1, src.count + 1))}")

        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    # Create placeholders for the full prediction and a count map for averaging
    prediction = np.zeros((len(class_names), height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    # Create a Gaussian window for smooth blending
    gaussian_window = np.exp(
        -((np.arange(patch_size) - patch_size // 2) ** 2 / (2 * (patch_size / 4) ** 2))
    )
    gaussian_window = np.outer(gaussian_window, gaussian_window)
    gaussian_window = np.tile(gaussian_window, (len(class_names), 1, 1))

    # Slide over the raster and predict on patches
    for y in tqdm(_window_offsets(height, patch_size, stride), desc="Processing patches"):
        for x in _window_offsets(width, patch_size, stride):
            patch = raster[:, y : y + patch_size, x : x + patch_size]
            # The model is trained on HWC patches, so normalize in that layout
            patch = normalize_imgnet(np.moveaxis(patch, 0, -1))
            patch = np.moveaxis(patch, -1, 0)
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                patch_pred = model(patch_tensor)
                patch_pred = torch.softmax(patch_pred, dim=1).squeeze(0).cpu().numpy()

            # Add the prediction to the main prediction array, weighted by the Gaussian window
            prediction[:, y : y + patch_size, x : x + patch_size] += (
                patch_pred * gaussian_window
            )
            counts[y : y + patch_size, x : x + patch_size] += np.squeeze(
                gaussian_window[0, :, :]
            )

    # Avoid division by zero
    counts[counts == 0] = 1
    # Normalize the predictions by the counts to average the overlapping areas
    prediction /= np.expand_dims(counts, axis=0)

    # Get the final class prediction
    final_prediction = np.argmax(prediction, axis=0).astype(np.uint8)

    return final_prediction, transform, crs, class_names


def vectorize_prediction(prediction, transform, crs, output_path, class_names):
    # Get shapes of predicted polygons
    results = [
        {"properties": {"class_id": v}, "geometry": s}
        for s, v in shapes(prediction, transform=transform)
    ]

    if not results:
        print("No features found in the prediction.")
        return

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(results)
    gdf.crs = crs

    # Add class names
    gdf["class"] = gdf["class_id"].map(
        lambda idx: class_names[int(idx)] if idx < len(class_names) else "Unknown"
    )

    # Save to shapefile
    gdf.to_file(output_path, driver="ESRI Shapefile")
    print(f"Shapefile saved to {output_path}")


def main():
    args = parse_options()

    image_basename = os.path.splitext(os.path.basename(args.image))[0]
    output_dir = os.path.join(args.exp_path, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    prediction, transform, crs, class_names = predict_raster(
        args.image,
        args.exp_path,
        args.patch_size,
        args.stride,
        args.checkpoint,
        args.bands,
    )

    # Save the raster prediction
    output_raster_path = os.path.join(output_dir, f"{image_basename}_prediction.tif")
    with rasterio.open(
        output_raster_path,
        "w",
        driver="GTiff",
        height=prediction.shape[0],
        width=prediction.shape[1],
        count=1,
        dtype=prediction.dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(prediction, 1)
    print(f"Predicted raster saved to {output_raster_path}")

    # Vectorize the prediction
    output_shapefile_path = os.path.join(output_dir, f"{image_basename}_prediction.shp")
    vectorize_prediction(prediction, transform, crs, output_shapefile_path, class_names)


if __name__ == "__main__":
    main()
