from django.contrib import admin
from myapp.models import *

@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('email','username','role','is_staff')
    search_fields = ('email','username')
    ordering = ('email',)

    
    
@admin.register(change_detection_with_building_footprint)
class BuildingFootprintAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'status', 'footprint_old_status', 'footprint_new_status', 'created_at')
    list_filter = ('status', 'footprint_old_status', 'footprint_new_status')
    search_fields = ('id', 'user__username')    


@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('subject', 'name', 'email', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('name', 'email', 'subject', 'message')


    
    
