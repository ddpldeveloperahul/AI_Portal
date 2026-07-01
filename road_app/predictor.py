import os
import tempfile
import zipfile
import cv2
import fiona
import torch
import rasterio
import numpy as np
import segmentation_models_pytorch as smp

from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape, mapping

MODEL_PATH = r"D:\AI_Portal\drdesingaiportal\ai_models\best_deeplabv3plus_road_new_detection.pth"

TILE_SIZE = 1024

THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

_MODEL = None


def load_model_once():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    _MODEL = model
    return _MODEL


def predict_tile(tile):
    model = load_model_once()

    h, w = tile.shape[:2]

    padded = np.zeros(
        (TILE_SIZE, TILE_SIZE, 3),
        dtype=np.uint8
    )

    padded[:h, :w] = tile

    x = padded.astype(np.float32) / 255.0

    x = np.transpose(x, (2, 0, 1))

    x = torch.tensor(
        x,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        pred = model(x)

        pred = torch.sigmoid(pred)

        pred = pred.squeeze().cpu().numpy()

    mask = (pred > THRESHOLD).astype(np.uint8)

    return mask[:h, :w]


def create_overlay(tile, mask):

    result = tile.copy()

    green = np.zeros_like(tile)

    green[:, :, 1] = 255

    road = mask == 1

    result[road] = (
        tile[road] * 0.6 +
        green[road] * 0.4
    ).astype(np.uint8)

    return result


def save_png_preview(tiff_path, png_path):
    with rasterio.open(tiff_path) as src:
        max_size = 2000
        scale = min(max_size / src.width, max_size / src.height, 1)
        out_width = max(1, int(src.width * scale))
        out_height = max(1, int(src.height * scale))
        band_count = min(src.count, 3)

        data = src.read(
            list(range(1, band_count + 1)),
            out_shape=(band_count, out_height, out_width),
            resampling=Resampling.bilinear
        )

    if band_count == 1:
        image = np.repeat(data[0][:, :, None], 3, axis=2)
    else:
        image = np.transpose(data, (1, 2, 0))

    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(png_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def export_mask_shapefile_zip(mask_path, zip_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        shp_path = os.path.join(temp_dir, "road_detection.shp")

        with rasterio.open(mask_path) as src:
            schema = {
                "geometry": "Polygon",
                "properties": {"class_id": "int"},
            }
            crs = src.crs.to_wkt() if src.crs else None

            open_kwargs = {
                "driver": "ESRI Shapefile",
                "schema": schema,
            }
            if crs:
                open_kwargs["crs_wkt"] = crs

            mask_data = src.read(1)

            with fiona.open(shp_path, "w", **open_kwargs) as dst:
                for geom, value in shapes(
                    mask_data,
                    mask=mask_data == 1,
                    transform=src.transform,
                    connectivity=8,
                ):
                    if int(value) != 1:
                        continue

                    polygon = shape(geom)
                    if polygon.is_empty:
                        continue

                    dst.write(
                        {
                            "geometry": mapping(polygon),
                            "properties": {"class_id": 1},
                        }
                    )

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            base, _ = os.path.splitext(shp_path)
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                file_path = base + ext
                if os.path.exists(file_path):
                    archive.write(file_path, os.path.basename(file_path))


def convert_to_temp_tiff(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    temp_dir = tempfile.gettempdir()
    temp_tiff_path = os.path.join(temp_dir, os.path.basename(image_path) + "_temp.tif")
    
    h, w, c = image.shape
    with rasterio.open(
        temp_tiff_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=rasterio.uint8,
    ) as dst:
        dst.write(np.transpose(image, (2, 0, 1)))
        
    return temp_tiff_path


def process_image(input_path, output_path, progress_callback=None, png_output_path=None, shapefile_zip_path=None):
    ext = os.path.splitext(input_path)[1].lower()
    is_temp = False
    
    if ext in [".jpg", ".jpeg", ".png"]:
        input_path = convert_to_temp_tiff(input_path)
        is_temp = True
        ext = ".tif"
        
    if ext in [".tif", ".tiff"]:
        mask_path = os.path.splitext(output_path)[0] + "_mask.tif"

        with rasterio.open(input_path) as src:
            if src.count < 3:
                raise ValueError("Input TIFF must have at least 3 bands.")

            width = src.width

            height = src.height

            profile = src.profile.copy()
            mask_profile = src.profile.copy()

            profile.update(
                driver="GTiff",
                dtype=rasterio.uint8,
                count=3,
                photometric="RGB"
            )
            profile.pop("nodata", None)

            for key in ("photometric", "interleave"):
                mask_profile.pop(key, None)

            mask_profile.update(
                driver="GTiff",
                dtype=rasterio.uint8,
                count=1,
                nodata=0
            )
            total_tiles = (
                ((height + TILE_SIZE - 1)//TILE_SIZE)
                *
                ((width + TILE_SIZE - 1)//TILE_SIZE)
            )

            done = 0

            with rasterio.open(
                output_path,
                "w",
                **profile
            ) as dst, rasterio.open(
                mask_path,
                "w",
                **mask_profile
            ) as mask_dst:

                for y in range(
                    0,
                    height,
                    TILE_SIZE
                ):

                    for x in range(
                        0,
                        width,
                        TILE_SIZE
                    ):

                        w = min(
                            TILE_SIZE,
                            width - x
                        )

                        h = min(
                            TILE_SIZE,
                            height - y
                        )

                        window = Window(
                            x,
                            y,
                            w,
                            h
                        )

                        tile = src.read(
                            [1, 2, 3],
                            window=window
                        )

                        tile = np.transpose(
                            tile,
                            (1, 2, 0)
                        )

                        mask = predict_tile(tile)

                        result = create_overlay(
                            tile,
                            mask
                        )

                        dst.write(
                            np.transpose(
                                result,
                                (2, 0, 1)
                            ),
                            window=window
                        )

                        mask_dst.write(
                            mask,
                            1,
                            window=window
                        )

                        done += 1

                        if progress_callback:

                            progress = int(
                                done / total_tiles * 100
                            )

                            progress_callback(progress)

        if png_output_path:
            save_png_preview(output_path, png_output_path)

        if shapefile_zip_path:
            export_mask_shapefile_zip(mask_path, shapefile_zip_path)

        if is_temp:
            try:
                os.remove(input_path)
            except Exception:
                pass

    return output_path


