from django.db import models


# class RoadDetection(models.Model):
#     STATUS_CHOICES = (
#         ('PENDING', 'PENDING'),
#         ('PROCESSING', 'PROCESSING'),
#         ('COMPLETED', 'COMPLETED'),
#         ('FAILED', 'FAILED'),
#     )

#     input_file = models.FileField(upload_to='upload_road/')
#     output_file = models.FileField(upload_to='outputs_road/', blank=True, null=True)
#     status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
#     progress = models.IntegerField(default=0)
#     error = models.TextField(blank=True, null=True)
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)

#     class Meta:
#         db_table = 'road_app_predictiontask'

#     def __str__(self):
#         return f"Road Detection {self.id}"

class RoadDetection(models.Model):

    STATUS_PENDING = "PENDING"
    STATUS_PROCESSING = "PROCESSING"
    STATUS_DONE = "DONE"
    STATUS_FAILED = "FAILED"

    image = models.FileField(upload_to='uploads_road/', null=True, blank=True)

    result = models.ImageField(
        upload_to='results_road/',
        null=True,
        blank=True
    )

    result_raster = models.FileField(
        upload_to='results_road/',
        null=True,
        blank=True
    )

    shapefile_zip = models.FileField(
        upload_to='results_road/',
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
