from django.http import FileResponse, Http404, JsonResponse
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import TreeDetection
from .tasks import run_prediction
from .utils import SUPPORTED_EXTENSIONS


@require_http_methods(["GET", "POST"])
def upload_page(request):
    if request.method == "GET":
        return render(request, "tree/tree.html")

    image = request.FILES.get("image")
    if not image:
        return render(request, "tree/tree.html", {"error": "Please choose an image file."})

    extension = "." + image.name.rsplit(".", 1)[-1].lower() if "." in image.name else ""
    if extension not in SUPPORTED_EXTENSIONS:
        return render(
            request,
            "tree/tree.html",
            {"error": "Use a .jpg, .jpeg, .png, .tif, or .tiff file."},
        )

    pred = TreeDetection.objects.create(image=image)
    transaction.on_commit(lambda: run_prediction.delay(pred.id))
    return redirect("result_tree", pk=pred.id)


def result_page(request, pk):
    pred = get_object_or_404(TreeDetection, id=pk)
    return render(request, "tree/result_tree.html", {"prediction": pred})


def download_shapefile(request, pk):
    pred = get_object_or_404(TreeDetection, id=pk)
    if not pred.shapefile_zip:
        raise Http404("Shapefile is not ready yet.")

    return FileResponse(
        pred.shapefile_zip.open("rb"),
        as_attachment=True,
        filename=f"prediction_{pred.id}_shapefile.zip",
    )


def prediction_status_json(request, pk):
    pred = get_object_or_404(TreeDetection, id=pk)
    return JsonResponse(
        {
            "id": pred.id,
            "status": pred.status,
            "result": pred.result.url if pred.result else None,
            "result_raster": pred.result_raster.url if pred.result_raster else None,
            "shapefile": pred.shapefile_zip.url if pred.shapefile_zip else None,
            "stats_file": pred.stats_csv.url if pred.stats_csv else None,
            "progress_percent": pred.progress_percent,
            "progress_message": pred.progress_message,
            "footprint_count": pred.footprint_count,
            "error": pred.error_message,
        }
    )


@api_view(["POST"])
def upload_image(request):
    image = request.FILES.get("image")
    if not image:
        return Response({"error": "Missing image file."}, status=400)

    extension = "." + image.name.rsplit(".", 1)[-1].lower() if "." in image.name else ""
    if extension not in SUPPORTED_EXTENSIONS:
        return Response(
            {"error": "Use a .jpg, .jpeg, .png, .tif, or .tiff file."},
            status=400,
        )

    pred = TreeDetection.objects.create(image=image)
    transaction.on_commit(lambda: run_prediction.delay(pred.id))

    return Response(
        {
            "prediction_id": pred.id,
            "status": pred.status,
            "progress_percent": pred.progress_percent,
            "progress_message": pred.progress_message,
            "status_url": reverse("tree_prediction_status", args=[pred.id]),
            "result_url": reverse("result_tree", args=[pred.id]),
        }
    )


@api_view(["GET"])
def check_status(request, pk):
    pred = get_object_or_404(TreeDetection, id=pk)
    return Response(
        {
            "id": pred.id,
            "status": pred.status,
            "result": pred.result.url if pred.result else None,
            "result_raster": pred.result_raster.url if pred.result_raster else None,
            "shapefile": pred.shapefile_zip.url if pred.shapefile_zip else None,
            "stats_file": pred.stats_csv.url if pred.stats_csv else None,
            "progress_percent": pred.progress_percent,
            "progress_message": pred.progress_message,
            "footprint_count": pred.footprint_count,
            "error": pred.error_message,
        }
    )
