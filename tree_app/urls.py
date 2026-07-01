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
    path("", upload_page, name="tree_home"),
    path("result_tree/<int:pk>/", result_page, name="result_tree"),
    path("result_tree/<int:pk>/status/", prediction_status_json, name="tree_prediction_status"),
    path("result_tree/<int:pk>/download/", download_shapefile, name="tree_download_shapefile"),
    path("api/predict/", upload_image, name="tree_api_predict"),
    path("api/status/<int:pk>/", check_status, name="tree_api_status"),
]
