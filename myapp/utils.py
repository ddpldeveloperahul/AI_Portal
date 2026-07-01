
import os
from datetime import datetime

# Set GDAL environment variable BEFORE importing fiona
os.environ["SHAPE_RESTORE_SHX"] = "YES"

import geopandas as gpd
import pandas as pd
import cv2
import fiona
import numpy as np
import rasterio
from contextlib import nullcontext
from rasterio.coords import disjoint_bounds
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import shape
import zipfile


ROW_CHUNK = 64
HALO = 16
MIN_BUILDING_PIXELS = 96
MIN_ROAD_PIXELS = 72
MIN_BUILDING_DELTA = 32.0
MIN_ROAD_DELTA = 18.0
MIN_BUILDING_SCORE = 0.60
MIN_ROAD_SCORE = 0.54
BUILDING_SCORE_MARGIN = 0.18
ROAD_SCORE_MARGIN = 0.14
SHADOW_BRIGHTNESS_DROP = 24.0
SHADOW_MAX_BRIGHTNESS = 160.0
SHADOW_MAX_SATURATION = 72.0
OLD_BUILDING_IGNORE_MARGIN = 0.08
MIN_BUILDING_WIDTH = 6
MIN_BUILDING_HEIGHT = 6
MAX_BUILDING_ASPECT = 6.0
MIN_BUILDING_FILL_RATIO = 0.28

RESCUE_BUILDING_SCORE = 0.56
RESCUE_ROAD_SCORE = 0.50
RESCUE_BUILDING_DELTA = 26.0
RESCUE_ROAD_DELTA = 16.0
RESCUE_BUILDING_MARGIN = 0.10
RESCUE_ROAD_MARGIN = 0.08
RESCUE_MIN_BUILDING_PIXELS = 60
RESCUE_MIN_ROAD_PIXELS = 54
STRONG_BUILDING_SCORE = 0.72
STRONG_ROAD_SCORE = 0.66
MIN_FOOTPRINT_PIXELS = 64
MIN_FOOTPRINT_WIDTH = 5
MIN_FOOTPRINT_HEIGHT = 5
MAX_FOOTPRINT_ASPECT = 10.0
MIN_FOOTPRINT_FILL_RATIO = 0.08
MIN_EXPORT_POLYGON_AREA = 10.0
OLD_FOOTPRINT_TOLERANCE_PIXELS = 3
FOOTPRINT_CLOSE_KERNEL = 3


def color_stats(rgb):
    r, g, b = [int(x) for x in rgb]
    vmax = max(r, g, b)
    vmin = min(r, g, b)
    sat = 0.0 if vmax == 0 else ((vmax - vmin) / vmax) * 255.0
    green_dom = g - max(r, b)
    blue_dom = b - max(r, g)
    spread = max(abs(r - g), abs(g - b), abs(r - b))
    return vmax, sat, green_dom, blue_dom, spread


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def clamp01_array(arr):
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def building_score(rgb):
    vmax, sat, green_dom, blue_dom, spread = color_stats(rgb)
    r, g, b = [int(x) for x in rgb]

    brightness = clamp01((vmax - 150.0) / 85.0)
    neutrality = max(clamp01((120.0 - sat) / 120.0), clamp01((58.0 - spread) / 58.0))
    warm_rg = clamp01((18.0 - (g - r)) / 18.0)
    warm_gb = clamp01((40.0 - (b - g)) / 40.0)
    warmth = min(warm_rg, warm_gb)
    white_bonus = 1.0 if vmax >= 224 and sat <= 30 else 0.0

    score = max(white_bonus, (0.48 * brightness) + (0.34 * neutrality) + (0.18 * warmth))
    if green_dom > 10 or blue_dom > 10:
        score *= 0.25
    if vmax < 145:
        score *= 0.35
    return clamp01(score)


def road_score(rgb):
    vmax, sat, green_dom, blue_dom, spread = color_stats(rgb)
    r, g, b = [int(x) for x in rgb]

    mid_brightness = clamp01(1.0 - (abs(vmax - 118.0) / 55.0))
    neutrality = clamp01((78.0 - sat) / 78.0)
    spread_score = clamp01((26.0 - spread) / 26.0)
    balance = clamp01((24.0 - max(abs(r - g), abs(g - b), abs(r - b))) / 24.0)

    score = (0.38 * mid_brightness) + (0.24 * neutrality) + (0.20 * spread_score) + (0.18 * balance)
    if green_dom > 8 or blue_dom > 8:
        score *= 0.25
    if vmax < 72 or vmax > 170:
        score *= 0.40
    return clamp01(score)


def ensure_uint8_rgb(data):
    if data.dtype == np.uint8:
        return data

    data = data.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    if max_val <= min_val + 1e-9:
        return np.zeros(data.shape, dtype=np.uint8)
    scaled = (data - min_val) / (max_val - min_val)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def palette_to_rgb(index_data, cmap):
    lut = np.zeros((256, 3), dtype=np.uint8)
    for value in range(256):
        lut[value] = cmap.get(value, (value, value, value, 255))[:3]
    return lut[index_data]


def read_rgb_window(src, window):
    if src.count >= 3:
        rgb = src.read([1, 2, 3], window=window, resampling=Resampling.nearest)
        return ensure_uint8_rgb(np.moveaxis(rgb, 0, 2))

    band = src.read(1, window=window, resampling=Resampling.nearest)
    if src.colorinterp and src.colorinterp[0].name.lower() == "palette":
        return palette_to_rgb(band, src.colormap(1))

    band = ensure_uint8_rgb(band)
    return np.repeat(band[:, :, None], 3, axis=2)


def building_score_array(rgb):
    rgb = rgb.astype(np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    vmax = np.maximum.reduce([r, g, b])
    vmin = np.minimum.reduce([r, g, b])
    sat = np.where(vmax <= 0.0, 0.0, ((vmax - vmin) / np.maximum(vmax, 1.0)) * 255.0)
    green_dom = g - np.maximum(r, b)
    blue_dom = b - np.maximum(r, g)
    spread = np.maximum.reduce([np.abs(r - g), np.abs(g - b), np.abs(r - b)])

    brightness = clamp01_array((vmax - 150.0) / 85.0)
    neutrality = np.maximum(clamp01_array((120.0 - sat) / 120.0), clamp01_array((58.0 - spread) / 58.0))
    warm_rg = clamp01_array((18.0 - (g - r)) / 18.0)
    warm_gb = clamp01_array((40.0 - (b - g)) / 40.0)
    warmth = np.minimum(warm_rg, warm_gb)
    white_bonus = ((vmax >= 224.0) & (sat <= 30.0)).astype(np.float32)

    score = np.maximum(white_bonus, (0.48 * brightness) + (0.34 * neutrality) + (0.18 * warmth))
    score = np.where((green_dom > 10.0) | (blue_dom > 10.0), score * 0.25, score)
    score = np.where(vmax < 145.0, score * 0.35, score)
    return clamp01_array(score)


def road_score_array(rgb):
    rgb = rgb.astype(np.float32)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    vmax = np.maximum.reduce([r, g, b])
    vmin = np.minimum.reduce([r, g, b])
    sat = np.where(vmax <= 0.0, 0.0, ((vmax - vmin) / np.maximum(vmax, 1.0)) * 255.0)
    green_dom = g - np.maximum(r, b)
    blue_dom = b - np.maximum(r, g)
    spread = np.maximum.reduce([np.abs(r - g), np.abs(g - b), np.abs(r - b)])
    balance = np.maximum.reduce([np.abs(r - g), np.abs(g - b), np.abs(r - b)])

    mid_brightness = clamp01_array(1.0 - (np.abs(vmax - 118.0) / 55.0))
    neutrality = clamp01_array((78.0 - sat) / 78.0)
    spread_score = clamp01_array((26.0 - spread) / 26.0)
    balance_score = clamp01_array((24.0 - balance) / 24.0)

    score = (0.38 * mid_brightness) + (0.24 * neutrality) + (0.20 * spread_score) + (0.18 * balance_score)
    score = np.where((green_dom > 8.0) | (blue_dom > 8.0), score * 0.25, score)
    score = np.where((vmax < 72.0) | (vmax > 170.0), score * 0.40, score)
    return clamp01_array(score)


def brightness_array(rgb):
    rgb = rgb.astype(np.float32)
    return np.maximum.reduce([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])


def saturation_array(rgb):
    rgb = rgb.astype(np.float32)
    vmax = np.maximum.reduce([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])
    vmin = np.minimum.reduce([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])
    return np.where(vmax <= 0.0, 0.0, ((vmax - vmin) / np.maximum(vmax, 1.0)) * 255.0)


def shadow_mask_array(old_rgb, new_rgb):
    old_brightness = brightness_array(old_rgb)
    new_brightness = brightness_array(new_rgb)
    new_saturation = saturation_array(new_rgb)
    brightness_drop = old_brightness - new_brightness
    return (
        (brightness_drop >= SHADOW_BRIGHTNESS_DROP)
        & (new_brightness <= SHADOW_MAX_BRIGHTNESS)
        & (new_saturation <= SHADOW_MAX_SATURATION)
    )


def compute_change_metrics(old_rgb, new_rgb):
    old_build_scores = building_score_array(old_rgb)
    new_build_scores = building_score_array(new_rgb)
    old_road_scores = road_score_array(old_rgb)
    new_road_scores = road_score_array(new_rgb)
    shadow_mask = shadow_mask_array(old_rgb, new_rgb)
    delta = np.sqrt(((new_rgb.astype(np.float32) - old_rgb.astype(np.float32)) ** 2).sum(axis=2))
    old_building_mask = (
        (old_build_scores >= (MIN_BUILDING_SCORE - OLD_BUILDING_IGNORE_MARGIN))
        & (old_build_scores >= old_road_scores)
    )
    strong_new_building = (new_build_scores >= STRONG_BUILDING_SCORE) & (delta >= (MIN_BUILDING_DELTA + 8.0))
    strong_new_road = (new_road_scores >= STRONG_ROAD_SCORE) & (delta >= (MIN_ROAD_DELTA + 6.0))

    return {
        "old_build_scores": old_build_scores,
        "new_build_scores": new_build_scores,
        "old_road_scores": old_road_scores,
        "new_road_scores": new_road_scores,
        "shadow_mask": shadow_mask,
        "delta": delta,
        "old_building_mask": old_building_mask,
        "strong_new_building": strong_new_building,
        "strong_new_road": strong_new_road,
    }


def classify_primary_changes(metrics):
    classes = np.zeros(metrics["delta"].shape, dtype=np.uint8)

    building_mask = (
        (metrics["new_build_scores"] >= MIN_BUILDING_SCORE)
        & (metrics["new_build_scores"] >= metrics["new_road_scores"] + 0.12)
        & ((metrics["new_build_scores"] - metrics["old_build_scores"]) >= BUILDING_SCORE_MARGIN)
        & (metrics["delta"] >= MIN_BUILDING_DELTA)
        & (~metrics["old_building_mask"])
        & (~metrics["shadow_mask"])
    )
    road_mask = (
        (metrics["new_road_scores"] >= MIN_ROAD_SCORE)
        & (metrics["new_road_scores"] >= metrics["new_build_scores"] + 0.08)
        & ((metrics["new_road_scores"] - metrics["old_road_scores"]) >= ROAD_SCORE_MARGIN)
        & (metrics["delta"] >= MIN_ROAD_DELTA)
        & (~metrics["shadow_mask"])
    )

    classes[road_mask] = 2
    classes[building_mask] = 1
    return classes


def classify_rescue_changes(metrics):
    classes = np.zeros(metrics["delta"].shape, dtype=np.uint8)
    old_strong_building = metrics["old_build_scores"] >= (MIN_BUILDING_SCORE + 0.06)
    shadow_exception = metrics["shadow_mask"] & (
        metrics["strong_new_building"] | metrics["strong_new_road"]
    )

    building_mask = (
        (metrics["new_build_scores"] >= RESCUE_BUILDING_SCORE)
        & (metrics["new_build_scores"] >= metrics["new_road_scores"] + 0.06)
        & ((metrics["new_build_scores"] - metrics["old_build_scores"]) >= RESCUE_BUILDING_MARGIN)
        & (metrics["delta"] >= RESCUE_BUILDING_DELTA)
        & (~old_strong_building | metrics["strong_new_building"])
        & ((~metrics["shadow_mask"]) | shadow_exception)
    )
    road_mask = (
        (metrics["new_road_scores"] >= RESCUE_ROAD_SCORE)
        & (metrics["new_road_scores"] >= metrics["new_build_scores"] + 0.05)
        & ((metrics["new_road_scores"] - metrics["old_road_scores"]) >= RESCUE_ROAD_MARGIN)
        & (metrics["delta"] >= RESCUE_ROAD_DELTA)
        & ((~metrics["shadow_mask"]) | shadow_exception)
    )

    classes[road_mask] = 2
    classes[building_mask] = 1
    return classes


def iter_row_windows(src):
    for row in range(0, src.height, ROW_CHUNK):
        row_start = max(0, row - HALO)
        row_stop = min(src.height, row + ROW_CHUNK + HALO)
        core_height = min(ROW_CHUNK, src.height - row)
        read_window = Window(0, row_start, src.width, row_stop - row_start)
        core_top = row - row_start
        core_bottom = core_top + core_height
        yield row, read_window, core_top, core_bottom


def filter_building_components(building_mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(building_mask, connectivity=8)
    filtered = np.zeros_like(building_mask)

    for label in range(1, num_labels):
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_area:
            continue
        if width < MIN_BUILDING_WIDTH or height < MIN_BUILDING_HEIGHT:
            continue

        long_side = max(width, height)
        short_side = max(1, min(width, height))
        if (long_side / short_side) > MAX_BUILDING_ASPECT:
            continue

        fill_ratio = area / float(width * height)
        if fill_ratio < MIN_BUILDING_FILL_RATIO:
            continue

        filtered[labels == label] = 1

    return filtered


def filter_road_components(road_mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask, connectivity=8)
    filtered = np.zeros_like(road_mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if area < min_area:
            continue
        if max(width, height) < 8:
            continue
        filtered[labels == label] = 1

    return filtered


def clean_primary_classes(added_classes):
    building = (added_classes == 1).astype(np.uint8)
    road = (added_classes == 2).astype(np.uint8)

    building = cv2.medianBlur(building, 5)
    building = cv2.morphologyEx(building, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    building = cv2.morphologyEx(building, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    building = filter_building_components(building, MIN_BUILDING_PIXELS)

    road = cv2.medianBlur(road, 5)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    road = filter_road_components(road, MIN_ROAD_PIXELS)

    cleaned = np.zeros_like(added_classes, dtype=np.uint8)
    cleaned[road == 1] = 2
    cleaned[building == 1] = 1
    return cleaned


def clean_rescue_classes(added_classes):
    building = (added_classes == 1).astype(np.uint8)
    road = (added_classes == 2).astype(np.uint8)

    building = cv2.medianBlur(building, 5)
    building = cv2.morphologyEx(building, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    building = cv2.morphologyEx(building, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    building = filter_building_components(building, RESCUE_MIN_BUILDING_PIXELS)

    road = cv2.medianBlur(road, 5)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    road = filter_road_components(road, RESCUE_MIN_ROAD_PIXELS)

    cleaned = np.zeros_like(added_classes, dtype=np.uint8)
    cleaned[road == 1] = 2
    cleaned[building == 1] = 1
    return cleaned


def merge_cleaned_classes(primary, rescue):
    merged = primary.copy()
    rescue_building = (rescue == 1) & (primary == 0)
    rescue_road = (rescue == 2) & (primary == 0)
    merged[rescue_road] = 2
    merged[rescue_building] = 1
    return merged


def make_preview_rgb(class_arr):
    rgb = np.full((class_arr.shape[0], class_arr.shape[1], 3), 255, dtype=np.uint8)
    rgb[class_arr == 1] = (255, 0, 0)
    rgb[class_arr == 2] = (0, 0, 255)
    return rgb


def red_footprint_mask(rgb):
    rgb = rgb.astype(np.int16)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    return (
        (r >= 110)
        & ((r - g) >= 35)
        & ((r - b) >= 25)
        & (g <= 170)
        & (b <= 170)
    ).astype(np.uint8)


def clean_new_footprint_mask(mask):
    mask = mask.astype(np.uint8)
    if FOOTPRINT_CLOSE_KERNEL > 1:
        kernel = np.ones((FOOTPRINT_CLOSE_KERNEL, FOOTPRINT_CLOSE_KERNEL), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        if area < MIN_FOOTPRINT_PIXELS:
            continue
        if width < MIN_FOOTPRINT_WIDTH or height < MIN_FOOTPRINT_HEIGHT:
            continue

        long_side = max(width, height)
        short_side = max(1, min(width, height))
        if (long_side / short_side) > MAX_FOOTPRINT_ASPECT:
            continue

        fill_ratio = area / float(width * height)
        if fill_ratio < MIN_FOOTPRINT_FILL_RATIO:
            continue

        cleaned[labels == label] = 1

    return cleaned


def export_shapefile(class_raster, shp_path):
    schema = {
        "geometry": "Polygon",
        "properties": {"id": "int", "class": "str:12", "value": "int"},
    }

    with rasterio.open(class_raster) as src:
        with fiona.open(
            shp_path,
            "w",
            driver="ESRI Shapefile",
            crs=src.crs.to_wkt() if src.crs else None,
            schema=schema,
        ) as sink:
            feature_id = 1
            for geom, value in shapes(rasterio.band(src, 1), transform=src.transform, connectivity=8):
                value = int(value)
                if value == 0:
                    continue
                polygon = shape(geom)
                if polygon.area < MIN_EXPORT_POLYGON_AREA:
                    continue
                sink.write(
                    {
                        "geometry": geom,
                        "properties": {
                            "id": feature_id,
                            "class": "building" if value == 1 else "road",
                            "value": value,
                        },
                    }
                )
                feature_id += 1


def open_aligned_new_source(old_src, new_src):
    # Cropped old TIFF ke extent ke bahar comparison avoid karne ke liye
    # pehle ensure karo ki full new TIFF us area ko overlap karta ho.
    if old_src.crs and new_src.crs:
        new_bounds_in_old_crs = transform_bounds(
            new_src.crs,
            old_src.crs,
            *new_src.bounds,
            densify_pts=21,
        )
        if disjoint_bounds(old_src.bounds, new_bounds_in_old_crs):
            raise ValueError("Old TIFF extent does not overlap the new TIFF extent.")
    elif old_src.crs != new_src.crs:
        raise ValueError("Both TIFF files need valid CRS metadata to align cropped and full images.")

    requires_alignment = any(
        (
            old_src.width != new_src.width,
            old_src.height != new_src.height,
            old_src.transform != new_src.transform,
            old_src.crs != new_src.crs,
        )
    )

    if not requires_alignment:
        return nullcontext(new_src)

    if not old_src.crs or not new_src.crs:
        raise ValueError("Cropped/full TIFF alignment requires CRS metadata in both files.")

    # Full new TIFF ko old cropped TIFF ki exact grid par warp/crop karo,
    # taaki shapefile sirf old image ke covered area ke liye bane.
    vrt_options = {
        "crs": old_src.crs,
        "transform": old_src.transform,
        "width": old_src.width,
        "height": old_src.height,
        "resampling": Resampling.nearest,
    }

    if new_src.nodata is not None:
        vrt_options["src_nodata"] = new_src.nodata
        vrt_options["nodata"] = new_src.nodata

    print("Aligning new TIFF to old TIFF extent, resolution, and georeferencing...")
    return WarpedVRT(new_src, **vrt_options)


def process_change_detection(old_tif, new_tif, class_output, preview_output):
    # Change detection is based only on the trained AI footprint model:
    # old TIFF -> AI mask, new TIFF -> AI mask, then new mask minus old mask.
    from .footprint_detection import generate_building_footprint_mask

    output_dir = os.path.dirname(class_output)
    old_mask_tif = os.path.join(output_dir, "ai_old_building_mask.tif")
    new_mask_tif = os.path.join(output_dir, "ai_new_building_mask.tif")

    print("Running AI model on old TIFF...")
    generate_building_footprint_mask(old_tif, old_mask_tif)

    print("Running AI model on new TIFF...")
    generate_building_footprint_mask(new_tif, new_mask_tif)

    with rasterio.open(old_mask_tif) as old_src, rasterio.open(new_mask_tif) as new_src_raw, open_aligned_new_source(
        old_src, new_src_raw
    ) as new_src:

        class_profile = old_src.profile.copy()
        class_profile.update(
            driver="GTiff",
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            photometric="minisblack",
        )

        preview_profile = old_src.profile.copy()
        preview_profile.update(
            driver="GTiff",
            dtype=rasterio.uint8,
            count=3,
            nodata=None,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            photometric="rgb",
        )

        building_pixels = 0
        road_pixels = 0

        with rasterio.open(class_output, "w", **class_profile) as class_dst, rasterio.open(
            preview_output, "w", **preview_profile
        ) as preview_dst:
            for row, read_window, core_top, core_bottom in iter_row_windows(old_src):
                old_mask = old_src.read(1, window=read_window, resampling=Resampling.nearest)
                new_mask = new_src.read(1, window=read_window, resampling=Resampling.nearest)

                old_mask = (old_mask > 0).astype(np.uint8)
                new_mask = (new_mask > 0).astype(np.uint8)
                if OLD_FOOTPRINT_TOLERANCE_PIXELS > 0:
                    kernel_size = (OLD_FOOTPRINT_TOLERANCE_PIXELS * 2) + 1
                    old_kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    old_mask = cv2.dilate(old_mask, old_kernel, iterations=1)

                added_classes = ((new_mask == 1) & (old_mask == 0)).astype(np.uint8)
                added_classes = clean_new_footprint_mask(added_classes)

                core = added_classes[core_top:core_bottom]
                preview = make_preview_rgb(core)

                core_window = Window(0, row, old_src.width, core.shape[0])
                class_dst.write(core, 1, window=core_window)
                preview_dst.write(np.moveaxis(preview, 2, 0), window=core_window)

                building_pixels += int((core == 1).sum())
                print(f"Processed rows {row} to {row + core.shape[0]} / {old_src.height}")

    return building_pixels, road_pixels


def main(old_tif, new_tif, output_dir):
    print("Checking input files...")
    if not os.path.exists(old_tif) or not os.path.exists(new_tif):
        raise FileNotFoundError("Input TIFF file not found.")

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    class_output = os.path.join(run_dir, "hybrid_building_road_classes.tif")
    preview_output = os.path.join(run_dir, "hybrid_building_road_preview.tif")
    shp_output = os.path.join(run_dir, "hybrid_building_road_changes.shp")

    print("Running merged RGB change detection...")
    building_pixels, road_pixels = process_change_detection(old_tif, new_tif, class_output, preview_output)

    print("Creating shapefile...")
    export_shapefile(class_output, shp_output)

    print(f"Class raster saved at: {class_output}")
    print(f"White/red/blue preview saved at: {preview_output}")
    print(f"Shapefile saved at: {shp_output}")
    print(f"New building pixels: {building_pixels}")
    print(f"New road pixels: {road_pixels}")

from datetime import datetime
def process_change(old_tif, new_tif, output_dir):
     # ✅ START TIME
    start_time = datetime.now()
    print("Checking input files...")
    if not os.path.exists(old_tif) or not os.path.exists(new_tif):
        raise FileNotFoundError("Input TIFF file not found.")

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    class_output = os.path.join(run_dir, "hybrid_building_road_classes.tif")
    preview_tif = os.path.join(run_dir, "hybrid_building_road_preview.tif")
    preview_png = os.path.join(run_dir, "hybrid_building_road_preview.png")
    shp_output = os.path.join(run_dir, "hybrid_building_road_changes.shp")
    zip_output = os.path.join(run_dir, "hybrid_building_road_changes.zip")

    print("Running merged RGB change detection...")
    process_change_detection(old_tif, new_tif, class_output, preview_tif)

    print("Creating shapefile...")
    export_shapefile(class_output, shp_output)

    with rasterio.open(preview_tif) as src:
        # Downsample to reasonable size to avoid memory issues
        MAX_PREVIEW_SIZE = 1024
        scale = max(src.width / MAX_PREVIEW_SIZE, src.height / MAX_PREVIEW_SIZE, 1)
        out_height = int(src.height / scale)
        out_width = int(src.width / scale)
        
        preview = src.read([1, 2, 3], out_shape=(3, out_height, out_width), resampling=Resampling.bilinear)
        preview = np.moveaxis(preview, 0, 2)
        preview = ensure_uint8_rgb(preview)
        preview = np.ascontiguousarray(preview)
        cv2.imwrite(preview_png, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

    print("Reading shapefile...")
    try:
        with fiona.open(shp_output) as collection:
            # for crop code 1416 - 1424
            if len(collection) == 0:
                print("No change polygons detected; exporting an empty shapefile package.")

        with zipfile.ZipFile(zip_output, "w") as archive:
            base, _ = os.path.splitext(shp_output)
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                part = base + ext
                if os.path.exists(part):
                    archive.write(part, os.path.basename(part))
    except Exception as e:
        print(f"Error opening shapefile: {e}")
        import traceback
        traceback.print_exc()
        # Check if shapefile files exist
        base, _ = os.path.splitext(shp_output)
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            part = base + ext
            print(f"File {ext}: exists={os.path.exists(part)}")
        raise
    # ✅ END TIME
    end_time = datetime.now()
    print(f"Total processing time: {end_time - start_time}")
    return preview_png, class_output, zip_output
    return preview_png, class_output, zip_output


if __name__ == "__main__":
    OLD_TIF = r"C:\Users\Faiz Ansari\Desktop\band3\phase1.tif"
    NEW_TIF = r"C:\Users\Faiz Ansari\Desktop\band3\phase2.tif"
    OUTPUT_DIR = r"C:\Users\Faiz Ansari\Desktop\band3\outputs"

    main(OLD_TIF, NEW_TIF, OUTPUT_DIR)


import geopandas as gpd

def process_spatial_join(main_path, change_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    shp_output = os.path.join(output_dir, "joined_output.shp")
    excel_output = os.path.join(output_dir, "joined_output.xlsx")

    # =========================
    # HELPER: Load shapefile with multi-layer support
    # =========================
    def load_shapefile(shp_path):
        """Load shapefile, handling multiple layers/nested directories by merging them"""
        try:
            # If it's a directory, find all .shp files and merge them
            if os.path.isdir(shp_path):
                # Find all .shp files recursively (handles nested directories)
                shp_files = []
                for root, _, files in os.walk(shp_path):
                    for file in files:
                        if file.lower().endswith('.shp'):
                            shp_files.append(os.path.join(root, file))
                
                if not shp_files:
                    raise Exception(f"No .shp files found in {shp_path}")
                
                if len(shp_files) == 1:
                    print(f"✅ Loaded single shapefile: {os.path.basename(shp_files[0])}")
                    return gpd.read_file(shp_files[0])
                
                # Merge multiple shapefiles
                print(f"📦 Found {len(shp_files)} shapefiles, merging...")
                gdfs = []
                for shp in shp_files:
                    try:
                        gdf = gpd.read_file(shp)
                        gdfs.append(gdf)
                        print(f"  ✅ Loaded: {os.path.basename(shp)} ({len(gdf)} features)")
                    except Exception as e:
                        print(f"  ⚠️ Skipped {os.path.basename(shp)}: {str(e)}")
                
                if not gdfs:
                    raise Exception(f"Could not load any valid shapefiles from {shp_path}")
                
                merged = pd.concat(gdfs, ignore_index=True)
                merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=gdfs[0].crs)
                print(f"✅ Merged {len(gdfs)} shapefiles into {len(merged)} total features")
                return merged
            else:
                print(f"✅ Loaded shapefile: {os.path.basename(shp_path)}")
                return gpd.read_file(shp_path)
        except Exception as e:
            raise Exception(f"Error loading shapefile {shp_path}: {str(e)}")

    # =========================
    # LOAD FILES
    # =========================
    main = load_shapefile(main_path)
    change_original = load_shapefile(change_path)

    # =========================
    # CRS FIX
    # =========================
    change = change_original
    if main.crs != change.crs:
        change = change.to_crs(main.crs)

    # =========================
    # FIND ID COLUMN (MAIN)
    # =========================
    possible_cols = ['Id', 'id', 'ID', 'fid', 'FID', 'objectid', 'OBJECTID']

    main_id_col = None
    for col in possible_cols:
        if col in main.columns:
            main_id_col = col
            break

    if main_id_col is None:
        raise Exception(f"No ID column found in MAIN. Columns: {list(main.columns)}")

    # =========================
    # REMOVE SMALL NOISE (IMPORTANT)
    # =========================
    change = change[change.geometry.area > 10].copy()   # threshold adjust kar sakte ho

    # =========================
    # SPATIAL JOIN
    # =========================
    joined = gpd.sjoin(
        main,
        change[['geometry']],
        how='left',
        predicate='intersects'
    )

    # =========================
    # CALCULATE INTERSECTION AREA
    # =========================
    joined['intersection_area'] = 0.0

    valid = joined['index_right'].notna()

    joined.loc[valid, 'intersection_area'] = joined[valid].apply(
        lambda row: row.geometry.intersection(
            change.loc[int(row['index_right'])].geometry
        ).area,
        axis=1
    )

    # =========================
    # BUILDING AREA
    # =========================
    joined['building_area'] = joined.geometry.area

    # =========================
    # % CHANGE
    # =========================
    joined['change_percent'] = joined['intersection_area'] / joined['building_area']

    # =========================
    # THRESHOLD (IMPORTANT)
    # =========================
    THRESHOLD = 0.05   # 5% change (adjust kar sakte ho)

    joined['changed_flag'] = joined['change_percent'] > THRESHOLD

    # =========================
    # REMOVE DUPLICATES
    # =========================
    final = joined.groupby(joined.index).agg({
        main_id_col: 'first',
        'geometry': 'first',
        'changed_flag': 'max',
        'change_percent': 'max'
    }).reset_index(drop=True)

    # =========================
    # FIX GEODATAFRAME ERROR
    # =========================
    final = gpd.GeoDataFrame(final, geometry='geometry', crs=main.crs)

    # =========================
    # ARCGIS-FRIENDLY FIELDS
    # Shapefile/DBF is limited: keep names <= 10 chars and avoid bool fields.
    # =========================
    final['CHANGED'] = final['changed_flag'].map({
        True: 'YES',
        False: 'NO'
    }).astype(str)
    final['CHG_FLAG'] = final['changed_flag'].astype(int)
    final['CHG_PCT'] = final['change_percent'].fillna(0).astype(float)

    # =========================
    # DOWNLOAD SHAPEFILE
    # Keep the change-detection shapefile geometry/attributes, and attach
    # the intersecting land ID so the downloaded ZIP matches the first page
    # change layer with parcel/building ID information added.
    # =========================
    change_export = change_original.copy()
    change_for_id = change_export
    if main.crs != change_export.crs:
        change_for_id = change_export.to_crs(main.crs)

    change_for_id = change_for_id.copy()
    change_for_id['_chg_idx'] = change_for_id.index

    matches = gpd.sjoin(
        change_for_id[['_chg_idx', 'geometry']],
        main[[main_id_col, 'geometry']],
        how='left',
        predicate='intersects'
    )

    land_ids = {}
    valid_matches = matches[matches['index_right'].notna()].copy()
    if not valid_matches.empty:
        valid_matches['intersection_area'] = valid_matches.apply(
            lambda row: row.geometry.intersection(
                main.loc[int(row['index_right'])].geometry
            ).area,
            axis=1
        )
        best_matches = valid_matches.sort_values('intersection_area').drop_duplicates('_chg_idx', keep='last')
        land_ids = best_matches.set_index('_chg_idx')[main_id_col].to_dict()

    change_export['LAND_ID'] = change_export.index.map(land_ids)
    change_export['CHANGED'] = 'YES'

    export_cols = [col for col in ['id', 'class', 'value'] if col in change_export.columns]
    final_export = change_export[export_cols + ['LAND_ID', 'CHANGED', 'geometry']].copy()

    # =========================
    # SAVE SHAPEFILE
    # =========================
    final_export.to_file(shp_output, driver='ESRI Shapefile', encoding='UTF-8')

    # =========================
    # SAVE EXCEL
    # =========================
    excel_df = final.copy()

    excel_df['geometry'] = excel_df['geometry'].apply(
        lambda g: g.wkt if g else None
    )
    excel_df['changed'] = final['CHANGED']

    with pd.ExcelWriter(excel_output) as writer:
        excel_df.to_excel(writer, sheet_name='All Data', index=False)
        excel_df[excel_df['changed'] == 'YES'].to_excel(writer, sheet_name='Changed', index=False)
        excel_df[excel_df['changed'] == 'NO'].to_excel(writer, sheet_name='Unchanged', index=False)

    # =========================
    # STATS
    # =========================
    total = len(final)
    changed = int(final['changed_flag'].sum())
    unchanged = total - changed

    return {
        "total": total,
        "changed": changed,
        "unchanged": unchanged,
        "shapefile": shp_output,
        "excel": excel_output
    }


def load_vector_layers(vector_path):
    """Load one vector file or every supported vector layer inside a folder."""
    supported_exts = (".shp", ".geojson", ".json", ".gpkg")

    if os.path.isdir(vector_path):
        vector_files = []
        for root, _, files in os.walk(vector_path):
            for file_name in files:
                if file_name.lower().endswith(supported_exts):
                    vector_files.append(os.path.join(root, file_name))

        if not vector_files:
            raise Exception(f"No vector layers found in {vector_path}")
    else:
        vector_files = [vector_path]

    gdfs = []
    for file_path in vector_files:
        try:
            gdf = gpd.read_file(file_path)
            if not gdf.empty:
                gdfs.append(gdf)
                print(f"Loaded vector layer: {os.path.basename(file_path)} ({len(gdf)} features)")
        except Exception as exc:
            print(f"Skipped vector layer {file_path}: {exc}")

    if not gdfs:
        raise Exception(f"Could not load any non-empty vector layer from {vector_path}")

    base_crs = gdfs[0].crs
    normalized = []
    for gdf in gdfs:
        if base_crs and gdf.crs and gdf.crs != base_crs:
            gdf = gdf.to_crs(base_crs)
        normalized.append(gdf)

    merged = pd.concat(normalized, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=base_crs)


def clean_footprint_geometries(gdf):
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        return gdf

    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def area_crs_for(gdf):
    if not gdf.crs:
        return None

    try:
        if gdf.crs.is_projected:
            return gdf.crs
        estimated = gdf.estimate_utm_crs()
        return estimated or gdf.crs
    except Exception:
        return gdf.crs


def process_new_footprint_detection(old_path, new_path, output_dir, existing_overlap_threshold=0.20):
    """
    Compare old and new building footprint layers and export only new buildings.

    A new footprint is treated as already existing when at least
    existing_overlap_threshold of its area overlaps old footprints.
    """
    os.makedirs(output_dir, exist_ok=True)

    shp_output = os.path.join(output_dir, "new_building_footprints.shp")
    excel_output = os.path.join(output_dir, "new_building_footprints.xlsx")

    old_gdf = clean_footprint_geometries(load_vector_layers(old_path))
    new_gdf = clean_footprint_geometries(load_vector_layers(new_path))

    if new_gdf.empty:
        raise Exception("New footprint layer has no valid building geometries.")

    if old_gdf.empty:
        old_gdf = gpd.GeoDataFrame(geometry=[], crs=new_gdf.crs)

    if old_gdf.crs and new_gdf.crs and old_gdf.crs != new_gdf.crs:
        old_gdf = old_gdf.to_crs(new_gdf.crs)
    elif old_gdf.crs is None and new_gdf.crs is not None:
        old_gdf = old_gdf.set_crs(new_gdf.crs, allow_override=True)

    working_crs = area_crs_for(new_gdf)
    if working_crs:
        new_work = new_gdf.to_crs(working_crs)
        old_work = old_gdf.to_crs(working_crs) if old_gdf.crs else old_gdf
    else:
        new_work = new_gdf.copy()
        old_work = old_gdf.copy()

    new_work = new_work.reset_index(drop=True)
    old_work = old_work.reset_index(drop=True)
    new_work["_new_idx"] = new_work.index
    new_work["_new_area"] = new_work.geometry.area

    overlap_ratio = pd.Series(0.0, index=new_work.index)

    if not old_work.empty:
        matches = gpd.sjoin(
            new_work[["_new_idx", "_new_area", "geometry"]],
            old_work[["geometry"]],
            how="left",
            predicate="intersects",
        )
        valid_matches = matches[matches["index_right"].notna()].copy()

        if not valid_matches.empty:
            valid_matches["_intersection_area"] = valid_matches.apply(
                lambda row: row.geometry.intersection(
                    old_work.loc[int(row["index_right"])].geometry
                ).area,
                axis=1,
            )
            total_overlap = valid_matches.groupby("_new_idx")["_intersection_area"].sum()
            overlap_ratio = (total_overlap / new_work["_new_area"]).clip(upper=1.0)
            overlap_ratio = overlap_ratio.reindex(new_work.index, fill_value=0.0)

    is_new = overlap_ratio < existing_overlap_threshold
    new_buildings = new_gdf.reset_index(drop=True).loc[is_new].copy()
    new_buildings["NEW_BLDG"] = "YES"
    new_buildings["OLD_OVLP"] = overlap_ratio.loc[is_new].round(4).values

    final_export = new_buildings[["NEW_BLDG", "OLD_OVLP", "geometry"]].copy()
    final_export = gpd.GeoDataFrame(final_export, geometry="geometry", crs=new_gdf.crs)
    final_export.to_file(shp_output, driver="ESRI Shapefile", encoding="UTF-8")

    excel_df = new_gdf.reset_index(drop=True).copy()
    excel_df["old_overlap_ratio"] = overlap_ratio.round(4).values
    excel_df["is_new_building"] = is_new.map({True: "YES", False: "NO"}).values
    excel_df["geometry"] = excel_df.geometry.apply(lambda geom: geom.wkt if geom else None)

    with pd.ExcelWriter(excel_output) as writer:
        excel_df.to_excel(writer, sheet_name="All Data", index=False)
        excel_df[excel_df["is_new_building"] == "YES"].to_excel(writer, sheet_name="New Buildings", index=False)
        excel_df[excel_df["is_new_building"] == "NO"].to_excel(writer, sheet_name="Existing", index=False)

    total = len(new_gdf)
    changed = int(is_new.sum())

    return {
        "total": total,
        "changed": changed,
        "unchanged": total - changed,
        "shapefile": shp_output,
        "excel": excel_output,
    }
