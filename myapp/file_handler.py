import os
import uuid
from django.conf import settings
from django.utils.text import get_valid_filename

def save_large_file(file_obj, folder="uploads", user=None):
    user_part = f"user_{user.id}" if user and getattr(user, "id", None) else "anonymous"
    upload_dir = os.path.join(settings.MEDIA_ROOT, folder, user_part)
    os.makedirs(upload_dir, exist_ok=True)

    original_name = get_valid_filename(os.path.basename(file_obj.name))
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    file_path = os.path.join(upload_dir, unique_name)

    with open(file_path, 'wb+') as destination:
        for chunk in file_obj.chunks():
            destination.write(chunk)

    return file_path
