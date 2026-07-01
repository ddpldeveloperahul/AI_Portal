from django.db import models
from django.contrib.auth.models import AbstractUser
import os  
from django.core.exceptions import ValidationError
from .managers import CustomUserManager

def validate_file_extension(value):

    allowed_extensions = [
        '.tif',
        '.tiff',
        '.png',
        '.jpg',
        '.jpeg'
    ]

    ext = os.path.splitext(
        value.name
    )[1]

    if ext.lower() not in allowed_extensions:

        raise ValidationError(
            'Unsupported file format.'
        )
class CustomUser(AbstractUser):
    username = models.CharField(max_length=100,unique=True)
    email = models.EmailField(unique=True)
    profile_image=models.ImageField(upload_to='profile/',blank=True,null=True)
    
    ROLE_CHOICES = [
        ('admin','Admin'),
        ('user','User')
    ]
    
    role=models.CharField(max_length=20,choices=ROLE_CHOICES,default='user')
    USERNAME_FIELD='email'
    REQUIRED_FIELDS=['username']
    objects=CustomUserManager()
    
    def __str__(self):
        return self.email

        
class change_detection_with_building_footprint(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)

    uploaded_2023 = models.FileField(upload_to='uploads/', validators=[validate_file_extension])
    uploaded_2025 = models.FileField(upload_to='uploads/', validators=[validate_file_extension])

    result_png = models.FileField(upload_to='images_upload/', validators=[validate_file_extension])
    result_tif = models.FileField(upload_to='images_upload/')
    result_shp = models.FileField(upload_to='images_upload/')

    status = models.CharField(
        max_length=20,
        default='pending'
    )

    # ====================================
    # Building Footprint Detection
    # ====================================

    footprint_old_status = models.CharField(
        max_length=20,
        default='pending'
    )

    footprint_new_status = models.CharField(
        max_length=20,
        default='pending'
    )

    # NEW FIELDS
    footprint_old_progress = models.IntegerField(
        default=0
    )

    footprint_new_progress = models.IntegerField(
        default=0
    )

    footprint_old = models.FileField(
        upload_to='footprints/',
        null=True,
        blank=True
    )

    footprint_new = models.FileField(
        upload_to='footprints/',
        null=True,
        blank=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"Change Result - {self.user.username}"

    @property
    def footprints_ready(self):
        return (
            self.footprint_old_status == "completed"
            and
            self.footprint_new_status == "completed"
        )



class SpatialJoinResult(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)

    main_shapefile = models.FileField(upload_to='shapefiles/')
    change_shapefile = models.FileField(upload_to='shapefiles/')

    result_shapefile = models.FileField(upload_to='output/')
    result_excel = models.FileField(upload_to='output/', validators=[validate_file_extension]) # ✅ ADD
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Spatial Join Result - {self.user.username}"


class ContactMessage(models.Model):
    name = models.CharField(max_length=150)
    email = models.EmailField()
    phone = models.CharField(max_length=20, blank=True, null=True)
    subject = models.CharField(max_length=200)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.subject} - {self.email}"

    
