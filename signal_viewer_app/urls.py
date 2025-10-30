# signal_viewer_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # --- Page Rendering Paths ---
    path('', views.index, name='index'),
    path('ecg/', views.ecg_view, name='ecg'),
    path('eeg/', views.eeg_view, name='eeg'),
    path('sar/', views.sar_view, name='sar'),
    path('detect/', views.detect_view, name='detect'),
    path('doppler/', views.doppler_view, name='doppler'),
    path('detect-cars/', views.detect_cars_view, name='detect_cars'),
    path('detect-voices/', views.detect_voices_view, name='detect_voices'),

    # --- API Paths ---
    # Path for ECG (already working)
    path('api/convert_ecg/', views.convert_ecg_dat_to_json, name='api_convert_ecg'),

    # Path for EEG
    path('api/convert_eeg/', views.convert_eeg_set_to_json, name='api_convert_eeg'),

    # Path for SAR
    path('api/process_sar/', views.process_sar_grd, name='api_process_sar'),

    # Path for Detect (Drone/Bird)
    path('api/analyze_audio/', views.analyze_audio_detect_bird_and_drone, name='api_analyze_audio'),

    # Path for Doppler Generator
    path('api/generate_doppler/', views.generate_doppler_audio, name='api_generate_doppler'),

    # Path for Detect Cars
    path('api/analyze_cars/', views.analyze_cars_audio, name='api_analyze_cars'),

    # Path for Detect Voices (Gender)
    path('api/analyze_voices/', views.analyze_voices_gender, name='api_analyze_voices'),

    # Path for Anti-Aliasing
    path('api/apply_anti_aliasing/', views.apply_anti_aliasing, name='api_apply_anti_aliasing'),

    # NEW: Path for Speed Prediction
    path('api/predict_speed/', views.predict_car_speed, name='api_predict_speed'),
    path('api/detect_ecg/', views.detect_ecg_abnormality, name='api_detect_ecg'),
]