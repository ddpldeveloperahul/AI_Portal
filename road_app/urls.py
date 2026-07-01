from django.urls import path
from .views import (
    check_status,
    download_shapefile,
    prediction_status_json,
    result_page,
    upload_image,
    upload_page,
)

urlpatterns = [
    path("", upload_page, name="road"),
    path("result_road/<int:pk>/", result_page, name="result_road"),
    path("result_road/<int:pk>/status/", prediction_status_json, name="road_prediction_status"),
    path("result_road/<int:pk>/download/", download_shapefile, name="road_download_shapefile"),
    path("api/predict/", upload_image, name="road_api_predict"),
    path("api/status/<int:pk>/", check_status, name="road_api_status"),
]
