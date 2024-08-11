import cv2
import numpy as np
import json

## Fotoğraf versiyonu

def average(values):
    return sum(values) / len(values)

def actual_width(target_class):
    width = next((item['actual_width'] for item in data if item['class'] == target_class), None)
    return width

def compute_distance(width_pixels, focal_length, object_real_width_cm):
    """Kameradan uzaklığı hesaplar"""
    if width_pixels > 0:
        return (object_real_width_cm * focal_length) / width_pixels
    return 0

# Sabit nesne boyutları ve odak uzunluğu (örnek olarak)
focal_length = 800  # Örnek odak uzunluğu (piksel cinsinden)

# YOLO modelini yükle
model = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# COCO dataset etiketlerini yükle
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
with open("configuration.json", "r") as file:
    data = json.load(file)

# Fotoğrafı yükle
image_path = 'indir.jpeg'
frame = cv2.imread(image_path)

# Model için giriş verisi hazırla
blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
model.setInput(blob)

# YOLO modelini çalıştır
outs = model.forward(model.getUnconnectedOutLayersNames())

# Tespit edilen nesneleri işleme
boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * frame.shape[1])
            center_y = int(detection[1] * frame.shape[0])
            w = int(detection[2] * frame.shape[1])
            h = int(detection[3] * frame.shape[0])

            # Sınırlayıcı kutuların koordinatları
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Listelere bilgiler eklenir
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# NMS uygulaması
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Tespit edilen nesneleri işaretleme
distance_values = {}
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)

        # Nesnenin uzaklığını hesapla
        distance_cm = compute_distance(w, focal_length, actual_width(label))

        # Sonuçları görüntüle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Dist: {distance_cm:.2f} cm", (x + 150, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Nesneye ait mesafe değerilerini kaydet
        if label not in distance_values:
            distance_values[label] = []
        distance_values[label].append(distance_cm)

# Ortalama mesafenin hesaplanması ve json'a aktarılması
averages = [{"object": label, "average_distance_cm": average(distance_values[label])} for label in distance_values]
with open('distance.json', 'w') as f:
    json.dump(averages, f, indent=4)

# Sonuç görüntüsünü kaydet
output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, frame)

print(f"İşlem tamamlandı. Sonuçlar {output_image_path} dosyasına kaydedildi.")
