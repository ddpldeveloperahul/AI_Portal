# =====================================================
# MASK R-CNN BUILDING FOOTPRINT TRAINING
# =====================================================

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms

from torchvision.models.detection import maskrcnn_resnet50_fpn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using:",device)

# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform=transforms.ToTensor()

# =====================================================
# CUSTOM DATASET
# =====================================================

class CustomCocoDataset(
    CocoDetection
):

    def __getitem__(self,idx):

        img,annotations=super().__getitem__(idx)

        img=transform(img)

        boxes=[]
        labels=[]
        masks=[]

        width=img.shape[2]
        height=img.shape[1]

        for ann in annotations:

            x,y,w,h=ann["bbox"]

            if w<=0 or h<=0:
                continue

            boxes.append(
                [x,y,x+w,y+h]
            )

            labels.append(1)

            # ==================================
            # BUILD MASK FROM SEGMENTATION
            # ==================================

            mask=torch.zeros(
                (height,width),
                dtype=torch.uint8
            )

            if "segmentation" in ann:

                import cv2
                import numpy as np

                for seg in ann["segmentation"]:

                    points=np.array(
                        seg
                    ).reshape(
                        -1,
                        2
                    )

                    points=np.int32(
                        points
                    )

                    cv2.fillPoly(
                        mask.numpy(),
                        [points],
                        1
                    )

            masks.append(mask)

        # ======================================
        # EMPTY IMAGE CASE
        # ======================================

        if len(boxes)==0:

            boxes=torch.zeros(
                (0,4),
                dtype=torch.float32
            )

            labels=torch.zeros(
                (0,),
                dtype=torch.int64
            )

            masks=torch.zeros(
                (0,height,width),
                dtype=torch.uint8
            )

        else:

            boxes=torch.tensor(
                boxes,
                dtype=torch.float32
            )

            labels=torch.tensor(
                labels,
                dtype=torch.int64
            )

            masks=torch.stack(
                masks
            )

        target={

            "boxes":boxes,

            "labels":labels,

            "masks":masks,

            "image_id":
            torch.tensor(
                [idx]
            ),

            "area":
            (
                boxes[:,3]
                -
                boxes[:,1]
            )
            *
            (
                boxes[:,2]
                -
                boxes[:,0]
            )
            if len(boxes)>0
            else torch.tensor([]),

            "iscrowd":
            torch.zeros(
                (
                    len(labels),
                ),
                dtype=torch.int64
            )
        }

        return img,target

# =====================================================
# DATASET
# =====================================================

dataset=CustomCocoDataset(

root=
"/content/building_detection_data/train",

annFile=
"/content/building_detection_data/train/_annotations.coco.json"

)

# =====================================================
# COLLATE
# =====================================================

def collate_fn(batch):

    return tuple(
        zip(*batch)
    )

loader=DataLoader(

dataset,

batch_size=1,

shuffle=True,

collate_fn=collate_fn
)

# =====================================================
# LOAD MASK R-CNN
# =====================================================

model=maskrcnn_resnet50_fpn(
weights="DEFAULT"
)

num_classes=2

# =====================================
# BOX PREDICTOR
# =====================================

in_features=model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor=FastRCNNPredictor(

in_features,

num_classes
)

# =====================================
# MASK PREDICTOR
# =====================================

in_features_mask=(

model.roi_heads
.mask_predictor
.conv5_mask
.in_channels

)

hidden_layer=256

model.roi_heads.mask_predictor=(

MaskRCNNPredictor(

in_features_mask,

hidden_layer,

num_classes
)

)

model.to(
device
)

# =====================================================
# OPTIMIZER
# =====================================================

optimizer=torch.optim.Adam(

model.parameters(),

lr=0.0001
)

# =====================================================
# TRAINING
# =====================================================

model.train()

EPOCHS=10

for epoch in range(
EPOCHS
):

    total_loss=0

    for i,(images,targets) in enumerate(
    loader
    ):

        images=[

        img.to(
        device
        )

        for img in images
        ]

        targets=[

        {

        k:v.to(
        device
        )

        for k,v in t.items()

        }

        for t in targets
        ]

        loss_dict=model(

        images,

        targets
        )

        losses=sum(

        loss

        for loss in loss_dict.values()

        )

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

        total_loss+=losses.item()

        print(

        f"Epoch={epoch+1}"

        f" Batch={i}"

        f" Loss={losses.item():.4f}"

        )

    print(

    f"\nEpoch {epoch+1}"

    f" Total Loss={total_loss:.4f}"

    )

# =====================================================
# SAVE MODEL
# =====================================================

torch.save(

model.state_dict(),

"/content/building_maskrcnn_trained.pth"

)

print(
"\nTraining completed"
)

print(
"Saved:"
"/content/building_maskrcnn_trained.pth"
)


# ------Train.py------



import torch
import cv2
import numpy as np
import os
import argparse
import rasterio
from rasterio.windows import Window

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # type: ignore
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor # type: ignore

from torchvision import transforms # type: ignore

# ====================================================
# SETTINGS
# ====================================================

DEFAULT_INPUT_FILE=r"D:\image_data\ortho1\tiff1.tif"

MODEL_PATH="building_maskrcnn_trained.pth"


OUTPUT_FILE="building_result_phase2.tif"

TILE_SIZE=2048

CONFIDENCE=0.30

device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using:",device)

parser=argparse.ArgumentParser(
    description="Run building detection on a JPG, PNG, or TIFF image."
)

parser.add_argument(
    "input_file",
    nargs="?",
    default=DEFAULT_INPUT_FILE,
    help="Path to the image file to process."
)

args=parser.parse_args()

INPUT_FILE=args.input_file

if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Run the script with a local image path, for example:\n"
        'python predict.py "D:\\path\\to\\image.jpg"'
    )

# ====================================================
# LOAD MODEL
# ====================================================

model=maskrcnn_resnet50_fpn(
    weights=None
)

num_classes=2

# Box predictor
in_features=(
model.roi_heads.box_predictor
.cls_score.in_features
)

model.roi_heads.box_predictor=(
FastRCNNPredictor(
in_features,
num_classes
)
)

# Mask predictor
in_features_mask=(
model.roi_heads
.mask_predictor
.conv5_mask
.in_channels
)

model.roi_heads.mask_predictor=(
MaskRCNNPredictor(
in_features_mask,
256,
num_classes
)
)

model.load_state_dict(

torch.load(
MODEL_PATH,
map_location=device,
weights_only=True
)

)

model.to(device)

model.eval()

print("Model Loaded")

# ====================================================
# GET FILE EXTENSION
# ====================================================

ext=os.path.splitext(
INPUT_FILE
)[1].lower()

# ====================================================
# JPG / PNG
# ====================================================

if ext in [".jpg",".jpeg",".png"]:

    print("Processing image...")

    img=cv2.imread(
    INPUT_FILE
    )

    if img is None:
        raise ValueError(
            f"OpenCV could not read this image: {INPUT_FILE}\n"
            "Check that the file is a valid JPG, JPEG, or PNG."
        )

    img_rgb=cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
    )

    tensor=transforms.ToTensor()(
    img_rgb
    ).to(device)

    with torch.no_grad():

        prediction=model(
        [tensor]
        )

    overlay=img_rgb.copy()

    masks=prediction[0]["masks"]

    scores=prediction[0]["scores"]

    for mask,score in zip(
    masks,
    scores
    ):

        if score<CONFIDENCE:
            continue

        mask=(

        mask[0]
        .cpu()
        .numpy()

        >0.5

        ).astype(
        np.uint8
        )

        contours,_=cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

        )

        cv2.fillPoly(
        overlay,
        contours,
        (255,0,0)
        )

        cv2.drawContours(
        overlay,
        contours,
        -1,
        (0,255,255),
        2
        )

    result=cv2.addWeighted(

    overlay,
    0.4,
    img_rgb,
    0.6,
    0

    )

    cv2.imwrite(

    "building_result.png",

    cv2.cvtColor(
    result,
    cv2.COLOR_RGB2BGR
    )

    )

    print(
    "Saved: building_result.png"
    )


# ====================================================
# TIFF LARGE FILE
# ====================================================

elif ext in [".tif",".tiff"]:

    print(
    "Processing TIFF..."
    )

    with rasterio.open(
    INPUT_FILE
    ) as src:

        width=src.width

        height=src.height

        transform=src.transform

        crs=src.crs

        profile={

            "driver":"GTiff",

            "height":height,

            "width":width,

            "count":3,

            "dtype":"uint8",

            "compress":"lzw",

            "tiled":True,

            "blockxsize":256,

            "blockysize":256,

            "transform":transform,

            "crs":crs
        }

        with rasterio.open(

        OUTPUT_FILE,

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

                    print(
                    f"Processing:{x},{y}"
                    )

                    window=Window(

                    x,

                    y,

                    min(
                    TILE_SIZE,
                    width-x
                    ),

                    min(
                    TILE_SIZE,
                    height-y
                    )
                    )

                    try:

                        tile=src.read(
                        [1,2,3],
                        window=window
                        )

                    except:

                        continue

                    tile=np.transpose(
                    tile,
                    (1,2,0)
                    )

                    tile=cv2.normalize(

                    tile,

                    None,

                    0,

                    255,

                    cv2.NORM_MINMAX
                    )

                    tile=tile.astype(
                    np.uint8
                    )

                    tensor=transforms.ToTensor()(
                    tile
                    ).to(device)

                    with torch.no_grad():

                        prediction=model(
                        [tensor]
                        )

                    overlay=tile.copy()

                    masks=prediction[0]["masks"]

                    scores=prediction[0]["scores"]

                    for mask,score in zip(

                    masks,
                    scores
                    ):

                        if score<CONFIDENCE:

                            continue

                        mask=(

                        mask[0]

                        .cpu()

                        .numpy()

                        >0.5

                        ).astype(
                        np.uint8
                        )

                        contours,_=cv2.findContours(

                        mask,

                        cv2.RETR_EXTERNAL,

                        cv2.CHAIN_APPROX_SIMPLE

                        )

                        cv2.fillPoly(

                        overlay,

                        contours,

                        (255,0,0)

                        )

                        cv2.drawContours(

                        overlay,

                        contours,

                        -1,

                        (0,255,255),

                        2
                        )

                    result=cv2.addWeighted(

                    overlay,

                    0.4,

                    tile,

                    0.6,

                    0

                    )

                    result=np.transpose(
                    result,
                    (2,0,1)
                    )

                    dst.write(
                    result,
                    window=window
                    )

else:
    raise ValueError(
        f"Unsupported input file extension: {ext}\n"
        "Use a .jpg, .jpeg, .png, .tif, or .tiff file."
    )

print(
"\nFinished Successfully"
)


# ----predict.py------