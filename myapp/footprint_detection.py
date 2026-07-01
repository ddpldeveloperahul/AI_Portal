import os
import cv2
import torch
import rasterio
import numpy as np

from rasterio.windows import Window

from torchvision import transforms

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from django.conf import settings


# =====================================================
# SETTINGS
# =====================================================

# Dynamic model path (works on any environment)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Try multiple possible paths
POSSIBLE_PATHS = [
    os.path.join(PROJECT_DIR, 'ai_models', 'building_maskrcnn_trained.pth'),
    os.path.join(BASE_DIR, 'ai_models', 'building_maskrcnn_trained.pth'),
    r"D:\AI_Portal\drdesingaiportal\ai_models\building_maskrcnn_trained.pth",
]

MODEL_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        print(f"Model found at: {MODEL_PATH}")
        break

if MODEL_PATH is None:
    print("WARNING: Model not found at any expected location!")
    print("Tried:")
    for path in POSSIBLE_PATHS:
        print(f"  - {path}")
    MODEL_PATH = POSSIBLE_PATHS[0]  # Use first path as fallback

CONFIDENCE = 0.30
TILE_SIZE = 2048

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

_MODEL = None


# =====================================================
# LOAD MODEL ONLY ONCE
# =====================================================

def load_model_once():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    try:
        print("Loading building footprint model...")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        model = maskrcnn_resnet50_fpn(weights=None)

        num_classes = 2

        # box predictor
        in_features = (
            model.roi_heads.box_predictor.cls_score.in_features
        )

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes
        )

        # mask predictor
        in_features_mask = (
            model.roi_heads.mask_predictor.conv5_mask.in_channels
        )

        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            256,
            num_classes
        )

        print(f"Loading model weights from: {MODEL_PATH}")
        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=DEVICE,
                weights_only=False
            )
        )

        model.to(DEVICE)
        model.eval()

        print("Building Footprint Model Loaded Successfully")

        _MODEL = model

        return _MODEL
        
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        print(f"   Device: {DEVICE}")
        print(f"   Model Path: {MODEL_PATH}")
        print(f"   File exists: {os.path.exists(MODEL_PATH)}")
        raise


# =====================================================
# MASK DRAWING
# =====================================================

def draw_predictions(image, prediction):

    overlay = image.copy()

    masks = prediction[0]["masks"]
    scores = prediction[0]["scores"]

    for mask, score in zip(masks, scores):

        if score < CONFIDENCE:
            continue

        mask = (
            mask[0]
            .cpu()
            .numpy()
            > 0.5
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.fillPoly(
            overlay,
            contours,
            (255, 0, 0)
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            (0, 255, 255),
            2
        )

    result = cv2.addWeighted(
        overlay,
        0.4,
        image,
        0.6,
        0
    )

    return result


def prediction_to_binary_mask(prediction, height, width):
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    masks = prediction[0]["masks"]
    scores = prediction[0]["scores"]

    for mask, score in zip(masks, scores):
        if score < CONFIDENCE:
            continue

        mask = (
            mask[0]
            .cpu()
            .numpy()
            > 0.5
        ).astype(np.uint8)

        combined_mask[mask == 1] = 1

    return combined_mask


def generate_building_footprint_mask(input_file, output_file, progress_callback=None):
    """
    Run the trained Mask R-CNN model and save a 1-band building mask TIFF.
    Pixel/RGB scoring is not used here; pixels are only model input.
    """

    model = load_model_once()
    ext = os.path.splitext(input_file)[1].lower()

    if progress_callback:
        progress_callback(5)

    if ext in [".jpg", ".jpeg", ".png"]:
        img = cv2.imread(input_file)
        if img is None:
            raise Exception(f"Unable to read image: {input_file}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transforms.ToTensor()(img_rgb).to(DEVICE)

        with torch.no_grad():
            prediction = model([tensor])

        mask = prediction_to_binary_mask(prediction, img_rgb.shape[0], img_rgb.shape[1])
        cv2.imwrite(output_file, (mask * 255).astype(np.uint8))

        if progress_callback:
            progress_callback(100)

        return output_file

    if ext not in [".tif", ".tiff"]:
        raise Exception(f"Unsupported format: {ext}")

    with rasterio.open(input_file) as src:
        width = src.width
        height = src.height

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "transform": src.transform,
            "crs": src.crs,
            "photometric": "minisblack",
        }

        total_tiles = (
            ((width + TILE_SIZE - 1) // TILE_SIZE)
            *
            ((height + TILE_SIZE - 1) // TILE_SIZE)
        )
        processed_tiles = 0

        with rasterio.open(output_file, "w", **profile) as dst:
            for y in range(0, height, TILE_SIZE):
                for x in range(0, width, TILE_SIZE):
                    window = Window(
                        x,
                        y,
                        min(TILE_SIZE, width - x),
                        min(TILE_SIZE, height - y)
                    )

                    try:
                        tile = src.read([1, 2, 3], window=window)
                    except Exception:
                        continue

                    tile = np.transpose(tile, (1, 2, 0))
                    tile = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX)
                    tile = tile.astype(np.uint8)

                    tensor = transforms.ToTensor()(tile).to(DEVICE)

                    with torch.no_grad():
                        prediction = model([tensor])

                    mask = prediction_to_binary_mask(
                        prediction,
                        tile.shape[0],
                        tile.shape[1]
                    )

                    dst.write(mask, 1, window=window)

                    processed_tiles += 1
                    if progress_callback:
                        progress = int((processed_tiles / total_tiles) * 100)
                        progress_callback(min(progress, 100))

    return output_file


# =====================================================
# MAIN FUNCTION
# =====================================================

def generate_building_footprint(
    input_file,
    output_file,
    progress_callback=None):
    """
    Run building footprint detection.

    Parameters:
        input_file : str
        output_file : str
        progress_callback : function(int)

    Returns:
        output_file
    """

    model = load_model_once()

    ext = os.path.splitext(
        input_file
    )[1].lower()

    if progress_callback:
        progress_callback(5)

    # =================================================
    # JPG / PNG
    # =================================================

    if ext in [".jpg", ".jpeg", ".png"]:

        img = cv2.imread(input_file)

        if img is None:
            raise Exception(
                f"Unable to read image: {input_file}"
            )

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        tensor = transforms.ToTensor()(
            img_rgb
        ).to(DEVICE)

        if progress_callback:
            progress_callback(40)

        with torch.no_grad():

            prediction = model([tensor])

        if progress_callback:
            progress_callback(80)

        result = draw_predictions(
            img_rgb,
            prediction
        )

        cv2.imwrite(
            output_file,
            cv2.cvtColor(
                result,
                cv2.COLOR_RGB2BGR
            )
        )

        if progress_callback:
            progress_callback(100)

        return output_file

    # =================================================
    # TIFF
    # =================================================

    elif ext in [".tif", ".tiff"]:

        with rasterio.open(input_file) as src:

            width = src.width
            height = src.height

            transform = src.transform
            crs = src.crs

            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 3,
                "dtype": "uint8",
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "transform": transform,
                "crs": crs
            }

            total_tiles = (
                ((width // TILE_SIZE) + 1)
                *
                ((height // TILE_SIZE) + 1)
            )

            processed_tiles = 0

            with rasterio.open(
                output_file,
                "w",
                **profile
            ) as dst:

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

                        window = Window(
                            x,
                            y,
                            min(
                                TILE_SIZE,
                                width - x
                            ),
                            min(
                                TILE_SIZE,
                                height - y
                            )
                        )

                        try:

                            tile = src.read(
                                [1, 2, 3],
                                window=window
                            )

                        except Exception:
                            continue

                        tile = np.transpose(
                            tile,
                            (1, 2, 0)
                        )

                        tile = cv2.normalize(
                            tile,
                            None,
                            0,
                            255,
                            cv2.NORM_MINMAX
                        )

                        tile = tile.astype(
                            np.uint8
                        )

                        tensor = transforms.ToTensor()(
                            tile
                        ).to(DEVICE)

                        with torch.no_grad():
                            prediction = model(
                                [tensor]
                            )

                        result = draw_predictions(
                            tile,
                            prediction
                        )

                        result = np.transpose(
                            result,
                            (2, 0, 1)
                        )

                        dst.write(
                            result,
                            window=window
                        )

                        processed_tiles += 1

                        if progress_callback:

                            progress = int(
                                (
                                    processed_tiles
                                    /
                                    total_tiles
                                ) * 100
                            )

                            progress_callback(
                                min(progress, 100)
                            )

        return output_file

    else:

        raise Exception(
            f"Unsupported format: {ext}"
        )
