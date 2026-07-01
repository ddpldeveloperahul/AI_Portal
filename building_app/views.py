from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth.decorators import login_required
from .models import BuildingDetection
from .tasks import run_prediction
from .utils import SUPPORTED_EXTENSIONS

@login_required
@require_http_methods(["GET", "POST"])
def upload_page(request):
    if request.method == "GET":
        return render(request, "building_app/building.html")

    image = request.FILES.get("image")
    if not image:
        return render(request, "building_app/building.html", {"error": "Please choose an image file."})

    extension = "." + image.name.rsplit(".", 1)[-1].lower() if "." in image.name else ""
    if extension not in SUPPORTED_EXTENSIONS:
        return render(
            request,
            "building_app/building.html",
            {"error": "Use a .jpg, .jpeg, .png, .tif, or .tiff file."},
        )

    pred = BuildingDetection.objects.create(image=image)
    run_prediction.delay(pred.id)
    return redirect("result", pk=pred.id)


def result_page(request, pk):
    pred = get_object_or_404(BuildingDetection, id=pk)
    return render(request, "building_app/result.html", {"prediction": pred})


def download_shapefile(request, pk):
    pred = get_object_or_404(BuildingDetection, id=pk)
    if not pred.shapefile_zip:
        raise Http404("Shapefile is not ready yet.")

    return FileResponse(
        pred.shapefile_zip.open("rb"),
        as_attachment=True,
        filename=f"prediction_{pred.id}_shapefile.zip",
    )


def prediction_status_json(request, pk):
    pred = get_object_or_404(BuildingDetection, id=pk)
    return JsonResponse(
        {
            "id": pred.id,
            "status": pred.status,
            "result": pred.result.url if pred.result else None,
            "shapefile": pred.shapefile_zip.url if pred.shapefile_zip else None,
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

    pred = BuildingDetection.objects.create(image=image)
    run_prediction.delay(pred.id)

    return Response(
        {
            "prediction_id": pred.id,
            "status": pred.status,
            "progress_percent": pred.progress_percent,
            "progress_message": pred.progress_message,
            "status_url": reverse("building_prediction_status", args=[pred.id]),
            "result_url": reverse("result", args=[pred.id]),
        }
    )


@api_view(["GET"])
def check_status(request, pk):
    pred = get_object_or_404(BuildingDetection, id=pk)
    return Response(
        {
            "id": pred.id,
            "status": pred.status,
            "result": pred.result.url if pred.result else None,
            "result_raster": pred.result_raster.url if pred.result_raster else None,
            "shapefile": pred.shapefile_zip.url if pred.shapefile_zip else None,
            "progress_percent": pred.progress_percent,
            "progress_message": pred.progress_message,
            "footprint_count": pred.footprint_count,
            "error": pred.error_message,
        }
    )
