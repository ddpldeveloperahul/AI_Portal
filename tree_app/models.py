from django.db import models


class TreeDetection(models.Model):

    STATUS_PENDING = "PENDING"
    STATUS_PROCESSING = "PROCESSING"
    STATUS_DONE = "DONE"
    STATUS_FAILED = "FAILED"

    image = models.FileField(upload_to='uploads_tree/', null=True, blank=True)

    result = models.ImageField(
        upload_to='results_tree/',
        null=True,
        blank=True
    )

    result_raster = models.FileField(
        upload_to='results_tree/',
        null=True,
        blank=True
    )

    shapefile_zip = models.FileField(
        upload_to='results_tree/',
        null=True,
        blank=True
    )

    stats_csv = models.FileField(
        upload_to='results_tree/',
        null=True,
        blank=True
    )

    footprint_count = models.PositiveIntegerField(default=0)

    progress_percent = models.PositiveSmallIntegerField(default=0)

    progress_message = models.CharField(
        max_length=120,
        blank=True,
        default=""
    )

    status = models.CharField(
        max_length=20,
        default=STATUS_PENDING
    )

    error_message = models.TextField(
        blank=True,
        default=""
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"Tree Detection {self.id}"
