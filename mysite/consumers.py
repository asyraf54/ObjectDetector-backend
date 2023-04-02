import asyncio
import json
import mimetypes
from django.test import RequestFactory
import pika
from channels.generic.websocket import AsyncWebsocketConsumer
from django.core.files.base import ContentFile
from django.http import HttpRequest, HttpResponse, QueryDict
from objectDetect.forms import ImageForm
from objectDetect.views import post
from objectDetect.object_detector import ObjectDetector
import base64
import base64
import io
import json
import requests
from PIL import Image
from channels.generic.websocket import AsyncWebsocketConsumer
import tempfile
import os


class ObjectDetectionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        print("\n\ntes1\n\n")
        # Set up connection to RabbitMQ server
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.channel = self.connection.channel()

        # Declare a queue for this consumer
        self.channel.queue_declare(queue='object_detection')

        # Start consuming messages from the queue
        self.channel.basic_consume(
            queue='object_detection',
            on_message_callback=self.handle_message,
            auto_ack=True
        )

    async def receive(self, text_data):
        message = json.loads(text_data)
            
        if message['type'] == 'process_images':
            image_files = message['data']
            results = await self.process_images(image_files)
            response = {'type': 'result', 'data': results}
            await self.send(text_data=json.dumps(response))


    async def process_images(self, image_files):
        if image_files is None:
            return
        results = []
        for image_file in image_files:
            try:
                # Decode the base64 encoded image data
                image_data = base64.b64decode(image_file.split(",")[1])
                
                # Convert the image data to a PIL image object
                img = Image.open(io.BytesIO(image_data))
                # Save the image to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    img.save(f, format='JPEG')
                    image_path = f.name
                detector = ObjectDetector('yolo/yolov3.cfg', 'yolo/yolov3.weights', 0.5)
                detection_image = detector.detect_objects(image_path)
                # Get object detection results and format as JSON response
                detections = detector.get_detections()
                with open("output.jpg", 'rb') as f:
                    file_data = f.read()
                # Encode the image data as base64 and send to client
                encoded_data = base64.b64encode(file_data).decode('utf-8')
                results.append(encoded_data)
               
            except Exception as e:
                print(f"Error processing image file: {e}")

        return results

    def handle_message(self, ch, method, properties, body):
        print("\n\ntes5\n\n")
        form_data = {'image': body}
        form = ImageForm(form_data)
        if form.is_valid():
            request = HttpRequest()
            request.method = 'POST'
            request.POST['image'] = form.cleaned_data['image']
            response = post(request)
            response_dict = json.loads(response.content)
            self.send(text_data=json.dumps(response_dict))
        else:
            self.send(text_data=json.dumps({'error': 'Invalid form'}))