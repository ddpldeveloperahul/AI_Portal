import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import shape, Polygon
import geopandas as gpd
import pandas as pd
import torch
from transformers import SegformerForSemanticSegmentation
from django.conf import settings

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TILE_SIZE = 1024
PREVIEW_MAX_SIZE = 1800
NUM_CLASSES = 3

MODEL_PATH = settings.BASE_DIR / "ai_models" / "best_segformer_b5_new1.pth"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    model.to(_device)
    model.eval()

    _model = model
    return _model


def _report_progress(callback, percent, message=""):
    if callback:
        callback(percent, message)


def _safe_output_dir(prediction_id):
    output_dir = settings.MEDIA_ROOT / "results_tree" / str(prediction_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _to_uint8_rgb(tile):
    if tile.shape[0] == 1:
        tile = np.repeat(tile, 3, axis=0)

    tile = np.transpose(tile[:3], (1, 2, 0))
    if tile.dtype == np.uint8:
        return tile

    return cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _predict_tile(model, tile):
    h, w = tile.shape[:2]

    padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
    padded[:h, :w] = tile.astype(np.float32) / 255.0

    tensor = (
        torch.tensor(padded.transpose(2, 0, 1), dtype=torch.float32)
        .unsqueeze(0)
        .to(_device)
    )

    with torch.no_grad():
        outputs = model(pixel_values=tensor)
        logits = outputs.logits
        pred = torch.nn.functional.interpolate(
            logits,
            size=(TILE_SIZE, TILE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        pred = pred.argmax(1)
        pred = pred.squeeze().cpu().numpy().astype(np.uint8)

    return pred[:h, :w]


def _create_overlay(tile, mask):
    color_mask = np.zeros_like(tile)
    # Tree (class 1) = Green [0, 255, 0]
    color_mask[mask == 1] = [0, 255, 0]
    # Vegetation (class 2) = Yellow [255, 255, 0]
    color_mask[mask == 2] = [255, 255, 0]

    overlay = cv2.addWeighted(tile, 0.7, color_mask, 0.3, 0)
    return overlay


def _mask_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if len(contour) >= 3]


def _contour_to_polygon(contour, x_offset=0, y_offset=0, transform=None):
    points = contour.reshape(-1, 2)
    coords = []

    for col, row in points:
        col = int(col) + x_offset
        row = int(row) + y_offset
        if transform is not None:
            x_coord, y_coord = transform * (col, row)
        else:
            x_coord, y_coord = float(col), float(row)
        coords.append((x_coord, y_coord))

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    polygon = Polygon(coords)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def _write_outputs(polygons_records, crs, output_dir, basename):
    if not polygons_records:
        gdf = gpd.GeoDataFrame(columns=["geometry", "class_id", "class_name"], crs=crs)
    else:
        gdf = gpd.GeoDataFrame(polygons_records, crs=crs)

    if not gdf.empty:
        gdf["area"] = gdf.geometry.area
        gdf = gdf[gdf["area"] > 10]

    shape_dir = output_dir / f"{basename}_shapefile"
    shape_dir.mkdir(parents=True, exist_ok=True)
    shp_path = shape_dir / f"{basename}.shp"

    gdf.to_file(shp_path)

    zip_path = output_dir / f"{basename}_shapefile.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in shape_dir.iterdir():
            archive.write(file_path, file_path.name)

    csv_path = output_dir / f"{basename}_area_stats.csv"
    if not gdf.empty:
        stats = gdf.groupby("class_name")["area"].sum().reset_index()
        stats.to_csv(csv_path, index=False)
    else:
        pd.DataFrame(columns=["class_name", "area"]).to_csv(csv_path, index=False)

    return zip_path, csv_path, len(gdf)


def _read_preview(src):
    scale = min(PREVIEW_MAX_SIZE / src.width, PREVIEW_MAX_SIZE / src.height, 1)
    preview_width = max(1, int(src.width * scale))
    preview_height = max(1, int(src.height * scale))
    indexes = [1, 2, 3] if src.count >= 3 else [1]
    data = src.read(indexes, out_shape=(len(indexes), preview_height, preview_width))
    preview = _to_uint8_rgb(data)
    return preview, scale


def _predict_standard_image(
    model, input_path, output_dir, basename, progress_callback=None
):
    _report_progress(progress_callback, 10, "Reading image")
    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {input_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    final_mask = np.zeros((height, width), dtype=np.uint8)
    polygons_records = []

    total_tiles = (
        ((height + TILE_SIZE - 1) // TILE_SIZE)
        * ((width + TILE_SIZE - 1) // TILE_SIZE)
    )
    processed_tiles = 0

    _report_progress(progress_callback, 20, "Running segmentation")
    for y in range(0, height, TILE_SIZE):
        for x in range(0, width, TILE_SIZE):
            tile = image_rgb[y : y + TILE_SIZE, x : x + TILE_SIZE]
            mask = _predict_tile(model, tile)

            h, w = tile.shape[:2]
            final_mask[y : y + h, x : x + w] = mask

            # Extract Tree polygons
            mask_tree = (mask == 1).astype(np.uint8)
            for contour in _mask_contours(mask_tree):
                polygon = _contour_to_polygon(contour, x_offset=x, y_offset=y)
                if polygon is not None:
                    polygons_records.append(
                        {"geometry": polygon, "class_id": 1, "class_name": "Tree"}
                    )

            # Extract Vegetation polygons
            mask_veg = (mask == 2).astype(np.uint8)
            for contour in _mask_contours(mask_veg):
                polygon = _contour_to_polygon(contour, x_offset=x, y_offset=y)
                if polygon is not None:
                    polygons_records.append(
                        {
                            "geometry": polygon,
                            "class_id": 2,
                            "class_name": "Vegetation",
                        }
                    )

            processed_tiles += 1
            tile_percent = 20 + (processed_tiles / total_tiles) * 60
            _report_progress(
                progress_callback,
                tile_percent,
                f"Processed {processed_tiles}/{total_tiles} tiles",
            )

    _report_progress(progress_callback, 82, "Creating overlay result")
    overlay = _create_overlay(image_rgb, final_mask)

    preview_path = output_dir / f"{basename}_result.png"
    cv2.imwrite(str(preview_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    _report_progress(progress_callback, 90, "Writing shapefile and stats")
    zip_path, csv_path, count = _write_outputs(
        polygons_records,
        crs=None,
        output_dir=output_dir,
        basename=basename,
    )

    return {
        "preview_path": preview_path,
        "raster_path": None,
        "shapefile_zip_path": zip_path,
        "stats_csv_path": csv_path,
        "footprint_count": count,
    }


def _predict_tiff(model, input_path, output_dir, basename, progress_callback=None):
    polygons_records = []
    raster_path = output_dir / f"{basename}_result.tif"
    preview_path = output_dir / f"{basename}_result.png"

    _report_progress(progress_callback, 10, "Reading TIFF")
    with rasterio.open(input_path) as src:
        _report_progress(progress_callback, 15, "Creating preview")
        preview_rgb, preview_scale = _read_preview(src)
        preview_result = preview_rgb.copy()

        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=3, compress="lzw")

        total_tiles = (
            ((src.height + TILE_SIZE - 1) // TILE_SIZE)
            * ((src.width + TILE_SIZE - 1) // TILE_SIZE)
        )
        processed_tiles = 0

        with rasterio.open(raster_path, "w", **profile) as dst:
            for y in range(0, src.height, TILE_SIZE):
                for x in range(0, src.width, TILE_SIZE):
                    window = Window(
                        x,
                        y,
                        min(TILE_SIZE, src.width - x),
                        min(TILE_SIZE, src.height - y),
                    )

                    indexes = [1, 2, 3] if src.count >= 3 else [1]
                    tile_data = src.read(indexes, window=window)
                    tile = _to_uint8_rgb(tile_data)

                    mask = _predict_tile(model, tile)
                    overlay = _create_overlay(tile, mask)
                    dst.write(np.transpose(overlay, (2, 0, 1)), window=window)

                    # Extract Tree polygons
                    mask_tree = (mask == 1).astype(np.uint8)
                    for contour in _mask_contours(mask_tree):
                        polygon = _contour_to_polygon(
                            contour,
                            x_offset=x,
                            y_offset=y,
                            transform=src.transform,
                        )
                        if polygon is not None:
                            polygons_records.append(
                                {
                                    "geometry": polygon,
                                    "class_id": 1,
                                    "class_name": "Tree",
                                }
                            )

                    # Extract Vegetation polygons
                    mask_veg = (mask == 2).astype(np.uint8)
                    for contour in _mask_contours(mask_veg):
                        polygon = _contour_to_polygon(
                            contour,
                            x_offset=x,
                            y_offset=y,
                            transform=src.transform,
                        )
                        if polygon is not None:
                            polygons_records.append(
                                {
                                    "geometry": polygon,
                                    "class_id": 2,
                                    "class_name": "Vegetation",
                                }
                            )

                    # Update preview
                    preview_mask = cv2.resize(
                        mask,
                        (
                            max(1, int(mask.shape[1] * preview_scale)),
                            max(1, int(mask.shape[0] * preview_scale)),
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    px = int(x * preview_scale)
                    py = int(y * preview_scale)
                    ph, pw = preview_mask.shape[:2]
                    target = preview_result[py : py + ph, px : px + pw]
                    if target.size:
                        preview_result[py : py + ph, px : px + pw] = (
                            _create_overlay(
                                target,
                                preview_mask[: target.shape[0], : target.shape[1]],
                            )
                        )

                    processed_tiles += 1
                    tile_percent = 20 + (processed_tiles / total_tiles) * 60
                    _report_progress(
                        progress_callback,
                        tile_percent,
                        f"Processed {processed_tiles}/{total_tiles} tiles",
                    )

        _report_progress(progress_callback, 82, "Saving preview")
        cv2.imwrite(str(preview_path), cv2.cvtColor(preview_result, cv2.COLOR_RGB2BGR))

        _report_progress(progress_callback, 90, "Writing shapefile and stats")
        zip_path, csv_path, count = _write_outputs(
            polygons_records,
            crs=src.crs,
            output_dir=output_dir,
            basename=basename,
        )

    return {
        "preview_path": preview_path,
        "raster_path": raster_path,
        "shapefile_zip_path": zip_path,
        "stats_csv_path": csv_path,
        "footprint_count": count,
    }


def predict_tree_vegetation(input_path, prediction_id, progress_callback=None):
    input_path = Path(input_path)
    ext = input_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported input file extension: {ext}. "
            "Use a .jpg, .jpeg, .png, .tif, or .tiff file."
        )

    _report_progress(progress_callback, 5, "Loading tree model")
    model = _load_model()
    output_dir = _safe_output_dir(prediction_id)
    basename = f"prediction_{prediction_id}"

    if ext in {".tif", ".tiff"}:
        return _predict_tiff(model, input_path, output_dir, basename, progress_callback)

    return _predict_standard_image(
        model, input_path, output_dir, basename, progress_callback
    )
