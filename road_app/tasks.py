import os
import time
import logging
from celery import shared_task
from django.conf import settings
from .models import RoadDetection
from .utils import predict_road_footprints

logger = logging.getLogger(__name__)


def _media_name(path):
    relative = os.path.relpath(path, settings.MEDIA_ROOT)
    return relative.replace("\\", "/")


def _set_progress(prediction_id, percent, message=""):
    RoadDetection.objects.filter(id=prediction_id).update(
        progress_percent=max(0, min(100, int(percent))),
        progress_message=message,
    )


@shared_task
def run_prediction(prediction_id):
    pred = None
    for _ in range(5):
        try:
            pred = RoadDetection.objects.get(id=prediction_id)
            break
        except RoadDetection.DoesNotExist:
            time.sleep(1)

    if pred is None:
        known_ids = list(
            RoadDetection.objects.order_by("-id").values_list("id", flat=True)[:10]
        )
        logger.warning(
            "Ignoring stale prediction task for id %s. Database: %s. Known ids: %s",
            prediction_id,
            settings.DATABASES["default"]["NAME"],
            known_ids,
        )
        return False

    pred.status = RoadDetection.STATUS_PROCESSING
    pred.error_message = ""
    pred.progress_percent = 1
    pred.progress_message = "Starting analysis"
    pred.save(
        update_fields=[
            "status",
            "error_message",
            "progress_percent",
            "progress_message",
        ]
    )

    try:
        outputs = predict_road_footprints(
            pred.image.path,
            pred.id,
            progress_callback=lambda percent, message="": _set_progress(
                prediction_id,
                percent,
                message,
            ),
        )
        pred.result.name = _media_name(outputs["preview_path"])
        pred.shapefile_zip.name = _media_name(outputs["shapefile_zip_path"])
        pred.footprint_count = outputs["footprint_count"]

        if outputs["raster_path"]:
            pred.result_raster.name = _media_name(outputs["raster_path"])

        pred.status = RoadDetection.STATUS_DONE
        pred.progress_percent = 100
        pred.progress_message = "Analysis complete"
        pred.save(
            update_fields=[
                "result",
                "result_raster",
                "shapefile_zip",
                "footprint_count",
                "status",
                "progress_percent",
                "progress_message",
            ]
        )
        return True
    except Exception as exc:
        pred.status = RoadDetection.STATUS_FAILED
        pred.error_message = str(exc)
        pred.progress_message = "Analysis failed"
        pred.save(update_fields=["status", "error_message", "progress_message"])
        raise
