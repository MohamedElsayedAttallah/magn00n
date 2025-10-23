# signal_viewer_project/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # Optional: Django Admin interface
    path('admin/', admin.site.urls),

    # Main application routing:
    # This includes all paths defined in signal_viewer_app/urls.py (like /, /ecg/, /api/convert_eeg/, etc.)
    path('', include('signal_viewer_app.urls')), 
]