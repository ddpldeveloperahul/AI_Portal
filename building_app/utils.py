import os
import zipfile
from pathlib import Path

import cv2
import fiona
import numpy as np
import rasterio
import torch
from django.conf import settings
from rasterio.windows import Window
from shapely.geometry import Polygon, mapping
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TILE_SIZE = 2048
CONFIDENCE = 0.15
PREVIEW_MAX_SIZE = 1800

MODEL_PATH = settings.BASE_DIR / "ai_models" / "building_maskrcnn_trained.pth"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def _load_model():
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = maskrcnn_resnet50_fpn(weights=None)
    num_classes = 2

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        num_classes,
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=_device, weights_only=True)
    )
    model.to(_device)
    model.eval()
    _model = model
    return _model


def _predict_tile(model, rgb_tile):
    tensor = transforms.ToTensor()(rgb_tile).to(_device)
    with torch.no_grad():
        prediction = model([tensor])[0]
    return prediction["masks"], prediction["scores"]


def _mask_contours(mask):
    binary = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if len(contour) >= 3]


def _draw_contours(rgb_tile, contours):
    overlay = rgb_tile.copy()
    cv2.fillPoly(overlay, contours, (255, 0, 0))
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return cv2.addWeighted(overlay, 0.4, rgb_tile, 0.6, 0)


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


def _write_shapefile_zip(polygons, output_dir, basename, crs=None):
    shape_dir = output_dir / f"{basename}_shapefile"
    shape_dir.mkdir(parents=True, exist_ok=True)
    shp_path = shape_dir / f"{basename}.shp"

    schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "int",
            "score": "float",
        },
    }

    crs_wkt = crs.to_wkt() if crs else None
    with fiona.open(
        shp_path,
        "w",
        driver="ESRI Shapefile",
        schema=schema,
        crs_wkt=crs_wkt,
    ) as layer:
        for index, (polygon, score) in enumerate(polygons, start=1):
            layer.write(
                {
                    "geometry": mapping(polygon),
                    "properties": {"id": index, "score": float(score)},
                }
            )

    zip_path = output_dir / f"{basename}_shapefile.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in shape_dir.iterdir():
            archive.write(file_path, file_path.name)

    return zip_path


def _safe_output_dir(prediction_id):
    output_dir = settings.MEDIA_ROOT / "results" / str(prediction_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _report_progress(callback, percent, message=""):
    if callback:
        callback(percent, message)


def predict_building_footprints(input_path, prediction_id, progress_callback=None):
    input_path = Path(input_path)
    ext = input_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported input file extension: {ext}. "
            "Use a .jpg, .jpeg, .png, .tif, or .tiff file."
        )

    _report_progress(progress_callback, 5, "Loading model")
    model = _load_model()
    output_dir = _safe_output_dir(prediction_id)
    basename = f"prediction_{prediction_id}"

    if ext in {".jpg", ".jpeg", ".png"}:
        return _predict_standard_image(
            model,
            input_path,
            output_dir,
            basename,
            progress_callback,
        )

    return _predict_tiff(model, input_path, output_dir, basename, progress_callback)


def _predict_standard_image(model, input_path, output_dir, basename, progress_callback=None):
    _report_progress(progress_callback, 10, "Reading image")
    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {input_path}")

    _report_progress(progress_callback, 20, "Preparing image")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    _report_progress(progress_callback, 35, "Running model")
    masks, scores = _predict_tile(model, image_rgb)

    _report_progress(progress_callback, 65, "Extracting footprints")
    all_contours = []
    polygons = []
    for mask, score in zip(masks, scores):
        if float(score) < CONFIDENCE:
            continue

        contours = _mask_contours(mask)
        all_contours.extend(contours)
        for contour in contours:
            polygon = _contour_to_polygon(contour)
            if polygon is not None:
                polygons.append((polygon, float(score)))

    result_rgb = _draw_contours(image_rgb, all_contours) if all_contours else image_rgb
    preview_path = output_dir / f"{basename}_result.png"
    _report_progress(progress_callback, 82, "Saving preview")
    cv2.imwrite(str(preview_path), cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))

    _report_progress(progress_callback, 92, "Writing shapefile")
    zip_path = _write_shapefile_zip(polygons, output_dir, basename)
    return {
        "preview_path": preview_path,
        "raster_path": None,
        "shapefile_zip_path": zip_path,
        "footprint_count": len(polygons),
    }


def _read_preview(src):
    scale = min(PREVIEW_MAX_SIZE / src.width, PREVIEW_MAX_SIZE / src.height, 1)
    preview_width = max(1, int(src.width * scale))
    preview_height = max(1, int(src.height * scale))
    indexes = [1, 2, 3] if src.count >= 3 else [1]
    data = src.read(indexes, out_shape=(len(indexes), preview_height, preview_width))

    if len(indexes) == 1:
        data = np.repeat(data, 3, axis=0)

    preview = np.transpose(data[:3], (1, 2, 0))
    preview = cv2.normalize(preview, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return preview, scale


def _predict_tiff(model, input_path, output_dir, basename, progress_callback=None):
    polygons = []
    preview_contours = []
    raster_path = output_dir / f"{basename}_result.tif"
    preview_path = output_dir / f"{basename}_result.png"

    _report_progress(progress_callback, 10, "Reading TIFF")
    with rasterio.open(input_path) as src:
        _report_progress(progress_callback, 15, "Creating preview")
        preview_rgb, preview_scale = _read_preview(src)
        profile = {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": 3,
            "dtype": "uint8",
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "transform": src.transform,
            "crs": src.crs,
        }

        total_tiles = (
            ((src.width + TILE_SIZE - 1) // TILE_SIZE)
            * ((src.height + TILE_SIZE - 1) // TILE_SIZE)
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

                    try:
                        indexes = [1, 2, 3] if src.count >= 3 else [1]
                        tile = src.read(indexes, window=window)
                    except Exception:
                        processed_tiles += 1
                        tile_percent = 20 + (processed_tiles / total_tiles) * 65
                        _report_progress(
                            progress_callback,
                            tile_percent,
                            f"Processed {processed_tiles}/{total_tiles} tiles",
                        )
                        continue

                    if tile.shape[0] == 1:
                        tile = np.repeat(tile, 3, axis=0)

                    tile = np.transpose(tile[:3], (1, 2, 0))
                    tile = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX)
                    tile = tile.astype(np.uint8)

                    masks, scores = _predict_tile(model, tile)
                    tile_contours = []

                    for mask, score in zip(masks, scores):
                        if float(score) < CONFIDENCE:
                            continue

                        contours = _mask_contours(mask)
                        tile_contours.extend(contours)

                        for contour in contours:
                            polygon = _contour_to_polygon(
                                contour,
                                x_offset=x,
                                y_offset=y,
                                transform=src.transform,
                            )
                            if polygon is not None:
                                polygons.append((polygon, float(score)))

                            scaled = contour.copy().astype(np.float32)
                            scaled[:, 0, 0] = (scaled[:, 0, 0] + x) * preview_scale
                            scaled[:, 0, 1] = (scaled[:, 0, 1] + y) * preview_scale
                            preview_contours.append(scaled.astype(np.int32))

                    result = _draw_contours(tile, tile_contours) if tile_contours else tile
                    dst.write(np.transpose(result, (2, 0, 1)), window=window)
                    processed_tiles += 1
                    tile_percent = 20 + (processed_tiles / total_tiles) * 65
                    _report_progress(
                        progress_callback,
                        tile_percent,
                        f"Processed {processed_tiles}/{total_tiles} tiles",
                    )

        _report_progress(progress_callback, 88, "Saving preview")
        preview_result = (
            _draw_contours(preview_rgb, preview_contours)
            if preview_contours
            else preview_rgb
        )
        cv2.imwrite(str(preview_path), cv2.cvtColor(preview_result, cv2.COLOR_RGB2BGR))
        _report_progress(progress_callback, 94, "Writing shapefile")
        zip_path = _write_shapefile_zip(polygons, output_dir, basename, crs=src.crs)

    return {
        "preview_path": preview_path,
        "raster_path": raster_path,
        "shapefile_zip_path": zip_path,
        "footprint_count": len(polygons),
    }
