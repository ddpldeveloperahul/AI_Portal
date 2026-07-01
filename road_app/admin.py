from django.contrib import admin
from django.utils.html import format_html

from .models import RoadDetection
from .tasks import run_prediction


@admin.register(RoadDetection)
class RoadDetectionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "status",
        "footprint_count",
        "image_link",
        "shapefile_link",
        "created_at",
    )
    list_filter = ("status", "created_at")
    search_fields = ("id", "image", "shapefile_zip")
    readonly_fields = (
        "result",
        "result_raster",
        "shapefile_zip",
        "footprint_count",
        "progress_percent",
        "progress_message",
        "status",
        "error_message",
        "created_at",
        "image_link",
        "preview_link",
        "raster_link",
        "shapefile_link",
    )
    fields = (
        "image",
        "image_link",
        "status",
        "progress_percent",
        "progress_message",
        "footprint_count",
        "result",
        "preview_link",
        "result_raster",
        "raster_link",
        "shapefile_zip",
        "shapefile_link",
        "error_message",
        "created_at",
    )
    ordering = ("-created_at",)

    def save_model(self, request, obj, form, change):
        is_new = obj.pk is None
        super().save_model(request, obj, form, change)

        if is_new:
            run_prediction.delay(obj.id)

    @admin.display(description="Uploaded file")
    def image_link(self, obj):
        return self._file_link(obj.image, "Open upload")

    @admin.display(description="Preview")
    def preview_link(self, obj):
        return self._file_link(obj.result, "Open preview")

    @admin.display(description="Result TIFF")
    def raster_link(self, obj):
        return self._file_link(obj.result_raster, "Download TIFF")

    @admin.display(description="Shapefile")
    def shapefile_link(self, obj):
        return self._file_link(obj.shapefile_zip, "Download shapefile")

    def _file_link(self, file_field, label):
        if not file_field:
            return "-"
        return format_html('<a href="{}" target="_blank">{}</a>', file_field.url, label)

