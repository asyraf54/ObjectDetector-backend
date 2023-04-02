import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path, config_path, confidence_threshold):
        self.net = cv2.dnn.readNetFromDarknet("yolo/yolov3.cfg", "yolo/yolov3.weights")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.confidence_threshold = confidence_threshold
        self.classes = None
        with open("yolo/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.detections = []
        
    def detect_objects(self, image):
        # Load image
        img = cv2.imread(image)
        
        # Get image dimensions
        (h, w) = img.shape[:2]
        
        # Detect objects in the image
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        print(self.net.getUnconnectedOutLayers())
        output_layers = [layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        
        # Process detected objects
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-maximum suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        
        # Draw bounding boxes and labels on image
        self.detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = round(confidences[i], 2)
                color = self.colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f"{label} {confidence}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                self.detections.append({
                    'class_name': label,
                    'confidence': confidence,
                    'bounding_box': [x, y, x+w, y+h]
                })
        
        # Save image with bounding boxes and labels
        cv2.imwrite("output.jpg", img)
        return "output.jpg"
    
    def get_detections(self):
        return self.detections
