from django.urls import path
from .views import post

urlpatterns = [
    path('', post, name='object_detection'),
]
