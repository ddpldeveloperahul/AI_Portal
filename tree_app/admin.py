from django.contrib import admin
from .models import TreeDetection


@admin.register(TreeDetection)
class TreeDetectionAdmin(admin.ModelAdmin):
    list_display = ("id", "image", "status", "footprint_count", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("id",)