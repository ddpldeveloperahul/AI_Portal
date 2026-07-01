# pyrefly: ignore [missing-import]
from celery import shared_task
from django.conf import settings
from django.core.files import File
from django.contrib.auth import get_user_model
import os, zipfile, shutil

from .utils import process_change, process_spatial_join, process_new_footprint_detection
# Defer torch import to avoid DLL issues at startup
from .models import change_detection_with_building_footprint, SpatialJoinResult

User = get_user_model()

@shared_task(bind=True)
def run_change_detection(self, img23_path, img25_path, user_id, job_id):

    from .views import build_preview_path, save_tiff_preview_png
    from .utils import process_change

    job = change_detection_with_building_footprint.objects.get(id=job_id)

    try:
        # 🔄 START PROCESS
        job.status = "processing"
        job.save(update_fields=["status"])

        output_path = os.path.join(settings.MEDIA_ROOT, 'outputs')
        os.makedirs(output_path, exist_ok=True)

        media_root = os.path.abspath(settings.MEDIA_ROOT)
        for field_name, file_path in (
            ("uploaded_2023", img23_path),
            ("uploaded_2025", img25_path),
        ):
            abs_path = os.path.abspath(file_path)
            if os.path.commonpath([media_root, abs_path]) == media_root:
                getattr(job, field_name).name = os.path.relpath(abs_path, media_root).replace("\\", "/")

        # =========================
        # 🖼 PREVIEW GENERATION
        # =========================
        img23_png = build_preview_path(img23_path)
        img25_png = build_preview_path(img25_path)

        save_tiff_preview_png(img23_path, img23_png)
        save_tiff_preview_png(img25_path, img25_png)

        # =========================
        # 🔥 MAIN PROCESS
        # =========================
        png, tif, zip_file = process_change(img23_path, img25_path, output_path)

        # =========================
        # 💾 SAVE OUTPUT FILES
        # =========================
        with open(png, 'rb') as f:
            job.result_png.save(os.path.basename(png), File(f), save=False)

        with open(tif, 'rb') as f:
            job.result_tif.save(os.path.basename(tif), File(f), save=False)

        if zip_file and os.path.exists(zip_file):
            with open(zip_file, 'rb') as f:
                job.result_shp.save(os.path.basename(zip_file), File(f), save=False)

        # =========================
        # ✅ COMPLETE (UNLOCK)
        # =========================
        job.status = "done"
        job.save()

        return {"status": "done", "id": job.id, "job_id": job.id}

    except Exception as e:
        import traceback
        print(traceback.format_exc())

        # =========================
        # ❌ FAIL (UNLOCK)
        # =========================
        job.status = "failed"
        job.save()

        raise e
    
# =========================
# 🔥 SPATIAL JOIN TASK
# =========================
@shared_task
def run_spatial_join(main_zip_path, change_zip_path, user_id):

    base_dir = settings.MEDIA_ROOT
    work_dir = os.path.join(base_dir, 'spatial_work')
    main_dir = os.path.join(work_dir, 'main_extract')
    change_dir = os.path.join(work_dir, 'change_extract')
    output_dir = os.path.join(base_dir, 'spatial_output')

    if not os.path.exists(main_zip_path):
        raise FileNotFoundError(f"Old shapefile ZIP not found: {main_zip_path}")

    if not os.path.exists(change_zip_path):
        raise FileNotFoundError(f"Change shapefile ZIP not found: {change_zip_path}")

    # clean extraction folders only; do not delete uploaded ZIP folders
    # Use safer cleanup to avoid permission errors
    def safe_remove_tree(path):
        """Safely remove directory tree, handling permission issues"""
        try:
            if os.path.exists(path):
                import stat
                def handle_remove_readonly(func, path, exc):
                    if not os.access(path, os.W_OK):
                        os.chmod(path, stat.S_IWUSR | stat.S_IREAD)
                        func(path)
                    else:
                        raise
                shutil.rmtree(path, onerror=handle_remove_readonly)
        except Exception as e:
            print(f"Warning: Could not fully clean {path}: {e}")
    
    for d in [main_dir, change_dir]:
        safe_remove_tree(d)
        os.makedirs(d, exist_ok=True)

    os.makedirs(output_dir, exist_ok=True)

    # unzip
    zipfile.ZipFile(main_zip_path).extractall(main_dir)
    zipfile.ZipFile(change_zip_path).extractall(change_dir)

    # find all shapefiles (to support multiple layers)
    def find_all_shp(folder):
        """Find all .shp files in folder (handles nested and wrapped ZIP structures)"""
        shp_files = []
        
        # Look for .shp files recursively
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith('.shp'):
                    shp_files.append(os.path.join(root, file))
        
        # If no .shp files found and folder has only one subfolder, search in that subfolder
        if not shp_files:
            contents = os.listdir(folder)
            subdirs = [d for d in contents if os.path.isdir(os.path.join(folder, d))]
            if len(subdirs) == 1:
                nested_dir = os.path.join(folder, subdirs[0])
                for root, _, files in os.walk(nested_dir):
                    for file in files:
                        if file.lower().endswith('.shp'):
                            shp_files.append(os.path.join(root, file))
        
        return shp_files if shp_files else None

    main_shp_list = find_all_shp(main_dir)
    change_shp_list = find_all_shp(change_dir)

    if not main_shp_list:
        main_contents = os.listdir(main_dir) if os.path.exists(main_dir) else []
        return {"error": f"SHP not found in main shapefile. Found: {main_contents}"}
    
    if not change_shp_list:
        change_contents = os.listdir(change_dir) if os.path.exists(change_dir) else []
        return {"error": f"SHP not found in change shapefile. Found: {change_contents}"}
    
    # Use directories for multi-layer support
    # process_spatial_join will automatically merge all shapefiles in the directory
    # process
    result = process_spatial_join(main_dir, change_dir, output_dir)
    shp_zip_path = os.path.splitext(result['shapefile'])[0] + ".zip"
    with zipfile.ZipFile(shp_zip_path, "w") as archive:
        base, _ = os.path.splitext(result['shapefile'])
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            part = base + ext
            if os.path.exists(part):
                archive.write(part, os.path.basename(part))

    user = User.objects.get(id=user_id)
    obj = SpatialJoinResult.objects.create(user=user)

    obj.main_shapefile.save(os.path.basename(main_zip_path), File(open(main_zip_path, 'rb')))
    obj.change_shapefile.save(os.path.basename(change_zip_path), File(open(change_zip_path, 'rb')))

    obj.result_shapefile.save(os.path.basename(shp_zip_path), File(open(shp_zip_path, 'rb')))
    obj.result_excel.save(os.path.basename(result['excel']), File(open(result['excel'], 'rb')))

    obj.save()

    return {
        "id": obj.id,
        "total": result.get("total", 0),
        "changed": result.get("changed", 0),
        "unchanged": result.get("unchanged", 0),
    }


def extract_vector_upload(input_path, output_dir):
    """Prepare uploaded ZIP/vector file for GeoPandas processing."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if input_path.lower().endswith(".zip"):
        with zipfile.ZipFile(input_path) as archive:
            archive.extractall(output_dir)
        return output_dir

    shutil.copy2(input_path, os.path.join(output_dir, os.path.basename(input_path)))
    return output_dir


@shared_task
def run_new_footprint_detection(old_zip_path, new_zip_path, user_id):
    base_dir = settings.MEDIA_ROOT
    work_dir = os.path.join(base_dir, "footprint_work")
    old_dir = os.path.join(work_dir, "old_extract")
    new_dir = os.path.join(work_dir, "new_extract")
    output_dir = os.path.join(base_dir, "footprint_output")

    if not os.path.exists(old_zip_path):
        raise FileNotFoundError(f"Old footprint file not found: {old_zip_path}")

    if not os.path.exists(new_zip_path):
        raise FileNotFoundError(f"New footprint file not found: {new_zip_path}")

    os.makedirs(output_dir, exist_ok=True)
    old_input = extract_vector_upload(old_zip_path, old_dir)
    new_input = extract_vector_upload(new_zip_path, new_dir)

    result = process_new_footprint_detection(old_input, new_input, output_dir)

    shp_zip_path = os.path.splitext(result["shapefile"])[0] + ".zip"
    with zipfile.ZipFile(shp_zip_path, "w") as archive:
        base, _ = os.path.splitext(result["shapefile"])
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            part = base + ext
            if os.path.exists(part):
                archive.write(part, os.path.basename(part))

    user = User.objects.get(id=user_id)
    obj = SpatialJoinResult.objects.create(user=user)

    with open(old_zip_path, "rb") as old_file:
        obj.main_shapefile.save(os.path.basename(old_zip_path), File(old_file), save=False)
    with open(new_zip_path, "rb") as new_file:
        obj.change_shapefile.save(os.path.basename(new_zip_path), File(new_file), save=False)
    with open(shp_zip_path, "rb") as shp_file:
        obj.result_shapefile.save(os.path.basename(shp_zip_path), File(shp_file), save=False)
    with open(result["excel"], "rb") as excel_file:
        obj.result_excel.save(os.path.basename(result["excel"]), File(excel_file), save=False)

    obj.save()

    return {
        "id": obj.id,
        "total": result.get("total", 0),
        "changed": result.get("changed", 0),
        "unchanged": result.get("unchanged", 0),
    }



@shared_task
def process_building_footprints(change_result_id):
    from .footprint_detection import generate_building_footprint

    obj = change_detection_with_building_footprint.objects.get(id=change_result_id)

    try:

        # ------------------------
        # OLD IMAGE
        # ------------------------

        obj.footprint_old_status = "processing"
        obj.save()

        old_output = os.path.join(
            settings.MEDIA_ROOT,
            "footprints",
            f"old_fp_{obj.id}.tif"
        )

        generate_building_footprint(
            obj.uploaded_2023.path,
            old_output
        )

        obj.footprint_old.name = (
            f"footprints/old_fp_{obj.id}.tif"
        )

        obj.footprint_old_status = "completed"
        obj.save()

        # ------------------------
        # NEW IMAGE
        # ------------------------

        obj.footprint_new_status = "processing"
        obj.save()

        new_output = os.path.join(
            settings.MEDIA_ROOT,
            "footprints",
            f"new_fp_{obj.id}.tif"
        )

        generate_building_footprint(
            obj.uploaded_2025.path,
            new_output
        )

        obj.footprint_new.name = (
            f"footprints/new_fp_{obj.id}.tif"
        )

        obj.footprint_new_status = "completed"

        obj.save()

    except Exception as e:

        obj.footprint_old_status = "failed"
        obj.footprint_new_status = "failed"
        obj.save()

        raise e


@shared_task(bind=True)
def process_single_footprint(self, file_path, job_id, image_type):
    from .models import change_detection_with_building_footprint
    from django.core.files import File
    from .footprint_detection import generate_building_footprint

    job = change_detection_with_building_footprint.objects.get(id=job_id)

    # Update database status to processing and progress to 0
    if image_type == 'old':
        job.footprint_old_status = 'processing'
        job.footprint_old_progress = 0
    else:
        job.footprint_new_status = 'processing'
        job.footprint_new_progress = 0
    job.save()

    def progress_callback(progress_val):
        # Update progress dynamically in the database
        try:
            current_job = change_detection_with_building_footprint.objects.get(id=job_id)
            if image_type == 'old':
                current_job.footprint_old_progress = progress_val
                current_job.save(update_fields=['footprint_old_progress'])
            else:
                current_job.footprint_new_progress = progress_val
                current_job.save(update_fields=['footprint_new_progress'])
        except Exception as e:
            print(f"Error in progress callback: {e}")

    try:
        output_dir = os.path.join(settings.MEDIA_ROOT, 'footprints')
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{image_type}_fp_{job_id}{os.path.splitext(file_path)[1]}"
        output_path = os.path.join(output_dir, filename)

        # Run model footprint prediction
        generate_building_footprint(
            file_path,
            output_path,
            progress_callback=progress_callback
        )

        # Save footprint to database model
        job = change_detection_with_building_footprint.objects.get(id=job_id)
        with open(output_path, 'rb') as f:
            if image_type == 'old':
                job.footprint_old.save(filename, File(f), save=False)
                job.footprint_old_status = 'completed'
                job.footprint_old_progress = 100
            else:
                job.footprint_new.save(filename, File(f), save=False)
                job.footprint_new_status = 'completed'
                job.footprint_new_progress = 100
        job.save()

        return {"status": "success", "image_type": image_type, "job_id": job_id}
    except Exception as e:
        job = change_detection_with_building_footprint.objects.get(id=job_id)
        if image_type == 'old':
            job.footprint_old_status = 'failed'
        else:
            job.footprint_new_status = 'failed'
        job.save()
        raise e
