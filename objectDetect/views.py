import mimetypes
import os

from django.shortcuts import render
from django.views import View
from django.http import HttpResponse, JsonResponse
from .forms import ImageForm
from .object_detector import ObjectDetector
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def post(request):
    form = ImageForm(request.POST, request.FILES)
    if form.is_valid():
        # images = request.FILES.getlist('image')
        image = form.cleaned_data['image']
        # print(images)
        # for image in images:

     
        directory = 'media/uploads'
        if not os.path.exists(directory):
            os.makedirs(directory)

        image_path = os.path.join(directory, image.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        detector = ObjectDetector('yolo/yolov3.cfg', 'yolo/yolov3.weights', 0.5)
        detection_image = detector.detect_objects(image_path)

        # Get object detection results and format as JSON response
        detections = detector.get_detections()
        with open("output.jpg", 'rb') as f:
            file_data = f.read()
        response = HttpResponse(content=file_data, content_type=mimetypes.guess_type("output.jpg")[0])
        response['Content-Disposition'] = 'attachment; filename="output.jpg"'
        return response
    else:
        return JsonResponse({'error': 'Invalid form'})
