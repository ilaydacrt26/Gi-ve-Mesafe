import cv2
import numpy as np
import time
import json
from ultralytics import YOLO  # YOLOv8 kütüphanesi

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

# YOLOv8 modelini yükle
model = YOLO("best.pt")

# COCO dataset etiketlerini yükle
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

with open("configuration.json", "r") as file:
    data = json.load(file)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Ortalamayı hesaplamak için değişkenler
distance_values = {}
time_period = 3
start_time = time.time()

while True:
    # Kameradan frame al
    ret, frame = cap.read()

    # YOLOv8 modelini çalıştır
    results = model(frame)

    # Tespit edilen nesneleri işleme
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Sınırlayıcı kutular
    confidences = results[0].boxes.conf.cpu().numpy()  # Güven skoru
    class_ids = results[0].boxes.cls.cpu().numpy()  # Sınıf id'leri

    for i in range(len(boxes)):
        x, y, x2, y2 = boxes[i].astype(int)
        w = x2 - x
        h = y2 - y
        label = str(classes[int(class_ids[i])])  # Nesne sınıfı
        confidence = confidences[i]

        # Nesnenin uzaklığını hesapla
        distance_cm = compute_distance(w, focal_length, actual_width(label))

        # Sonuçları görüntüle
        color = (0, 255, 0)  # Yeşil
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Dist: {distance_cm:.2f} cm", (x + 150, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Nesneye ait mesafe değerilerini kaydet
        if label not in distance_values:
            distance_values[label] = []
        distance_values[label].append(distance_cm)

    # 3 saniyede bir ortalama mesafenin hesaplanması ve json'a aktarılması
    if time.time() - start_time >= time_period:
        averages = [{"object": label, "average_distance_cm": average(distance_values[label])} for label in distance_values]
        with open('distance.json', 'w') as f:
            json.dump(averages, f, indent=4)
        start_time = time.time()
        distance_values = {}  # Yeni ölçümler için sıfırlama

    # Sonuçları göster
    cv2.imshow("Object Detection", frame)

    # 'Esc' tuşuna basarak çıkış yap
    key = cv2.waitKey(1)
    if key == 27:
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
