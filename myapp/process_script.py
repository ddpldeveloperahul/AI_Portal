import sys
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "m_gis_image.settings")
django.setup()

from myapp.models import ChangeResult
from myapp.utils import process_change

job_id = sys.argv[1]

job = ChangeResult.objects.get(id=job_id)

try:
    job.status = "processing"
    job.save()

    png, tif, shp = process_change(
        job.uploaded_2023.path,
        job.uploaded_2025.path,
        "/var/www/m_gis_image/media/outputs"
    )

    job.result_png.name = png.replace("/var/www/m_gis_image/media/", "")
    job.result_tif.name = tif.replace("/var/www/m_gis_image/media/", "")

    if shp:
        job.result_shp.name = shp.replace("/var/www/m_gis_image/media/", "")

    job.status = "done"
    job.save()

except Exception as e:
    job.status = "failed"
    job.save()
    print(e)