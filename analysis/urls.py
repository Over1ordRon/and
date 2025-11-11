# analysis/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='upload_file'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('analyze/', views.analyze_data, name='analyze_data'),
    path('generate-graph/', views.generate_graph, name='generate_graph'),
    path('clear-session/', views.clear_session, name='clear_session'),
]