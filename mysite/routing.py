from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/object_detection/', consumers.ObjectDetectionConsumer.as_asgi()),
]