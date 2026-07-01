from django.conf import settings
from django.conf.urls.static import static
from myapp import views
from django.urls import path
from django.contrib.auth import views as auth_views
from myapp import views
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView # type: ignore

urlpatterns = [
    path('', views.base, name='base'),
    path('login/', views.login_page, name='login'),
    path('signup/', views.signup_page, name='signup'),
    path('logout/', views.logout_page, name='logout'),
    # Forgot password
    path('password-reset/',auth_views.PasswordResetView.as_view(template_name='authentication/forgot_password.html',email_template_name='authentication/password_reset_email.html',success_url='/password-reset-done/'),name='password_reset'),
    path('password-reset-done/',auth_views.PasswordResetDoneView.as_view(template_name='authentication/password_reset_done.html'),name='password_reset_done'),
    path('reset/<uidb64>/<token>/',auth_views.PasswordResetConfirmView.as_view(template_name='authentication/password_reset_confirm.html',success_url='/password-reset-complete/'),name='password_reset_confirm'),
    path('password-reset-complete/',auth_views.PasswordResetCompleteView.as_view(template_name='authentication/password_reset_complete.html'),name='password_reset_complete'),
    # Change password
    path('change-password/',auth_views.PasswordChangeView.as_view(template_name='authentication/change_password.html',success_url='/password-changed/'),name='change_password'),
    path('password-changed/',auth_views.PasswordChangeDoneView.as_view(template_name='authentication/password_changed.html'),name='password_change_done'
    ),
    path('tools/', views.list_tools, name='tools'),
    path('contact/', views.contact_view, name='contact'),
    path('about/', views.about_page, name='about'),
    path('status/<int:task_id>/',views.task_status,name='task_status'),
    
    path('home/', views.home, name='home'),
    #api endpoints
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    path('upload/', views.upload_images, name='upload'),
    path('result/', views.result_view, name='change_result'),
    path('run-spatial-join/', views.spatial_join_view, name='result'),
    path('download-excel/', views.download_excel, name='download_excel'),
    path('download-shapefile/', views.download_shapefile, name='download_shapefile'),
    
    # Chunked upload for large files
    path('processing-availability/', views.processing_availability, name='processing_availability'),
    path('upload-chunk/', views.upload_chunk, name='upload_chunk'),
    
    # Building Footprint Detection
    path('start-footprint-detection/', views.start_footprint_detection, name='start_footprint_detection'),
    path('footprint-status/<int:job_id>/', views.footprint_status, name='footprint_status'),
    path('get-footprint-image/', views.get_footprint_image, name='get_footprint_image'),
    
    path('api/signup/', views.signup_api),
    path('api/login/', views.login_api),
    path('api/logout/', views.logout_api),
    path('api/excel-files/', views.list_excel_files, name='list_excel_files'),
    path('spatial-join/', views.spatial_join_view, name='spatial_join'),
    path('start-processing/', views.start_processing, name='start_processing'),
    path('task-status/<str:task_id>/', views.task_status, name='task_status'),

    
    
    path('', views.login_page),
    # path('login/', views.login_page, name='login'),
    # path('signup/', views.signup_page),
    path('footprint-status/<int:pk>/',views.footprint_status,name='footprint-status'),
    
    
    
] +  static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)   
