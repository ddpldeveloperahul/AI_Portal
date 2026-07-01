from django.urls import path

from .views import *

urlpatterns = [
    
    
    path("", upload_page, name="building_home"),
    path("result/<int:pk>/", result_page, name="result"),
    path("result/<int:pk>/status/", prediction_status_json, name="building_prediction_status"),
    path("result/<int:pk>/download/", download_shapefile, name="building_download_shapefile"),
    path("api/predict/", upload_image, name="building_api_predict"),
    path("api/status/<int:pk>/", check_status, name="building_api_status"),
]

