import cv2
import numpy as np
import time
import json

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

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Ortalamayı hesaplamak için değişkenler
distance_values = {}
time_period = 3
start_time = time.time()

while True:
    # Kameradan frame al
    ret, frame = cap.read()

    # Model için giriş verisi hazırla
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # frame --> giriş görüntüsü
    # 0.00392 --> ölçek faktörüdür. piksel değerlerinin 0 ile 1 arasında normalize edilmesini sağlar.
    # (416, 416) --> görüntü boyutlarını yeniden boyutlandırmak için kullanılır
    # (0, 0, 0) --> ortalama renk değerleri çıkarıldı.
    # True --> Kanal sıralamasıdır. BGR kanal sıralamasını işaret eder.OpenCv nin varsayılan renk sıralaması)
    # crop=False --> Frame alınırken görüntünün kesilmesini engeller
    model.setInput(blob)

    # YOLO modelini çalıştır
    outs = model.forward(model.getUnconnectedOutLayersNames())
    # outs --> tahmin çıktıları

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
    # aynı nesnenin birden fazla kutuyla tespit edilmesini önlemek için kullanılır
    # 0.5 --> güven skoru
    # 0.4 --> örtüşme oranı - kutucukların hangi oranda örtüşeceklerini belirler - IoU oranı da denir

    # Tespit edilen nesneleri işaretleme
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i] # o anki indekse ait konum bilgisi alındı
            label = str(classes[class_ids[i]]) # o anki indekse ait sınıf id'si alınıp sınıf adı tespit edildi
            confidence = confidences[i] # o anki indekse ait güven skoru alındı
            color = (0, 255, 0) # yeşil - sınırlayıcı kutu rengi

            # Nesnenin uzaklığını hesapla
            distance_cm = compute_distance(w, focal_length, actual_width(label))

            # Sonuçları görüntüle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # kutu çizimi
            # 2 --> kenar kalınlığı
            
            # yazı eklemesi
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Dist: {distance_cm:.2f} cm", (x + 150, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Nesneye ait mesafe değerilerini kaydet
            if label not in distance_values:
                distance_values[label] = []
            distance_values[label].append(distance_cm)
            
    # 3 sn de bir ortalama mesafenin hesaplanması ve json a aktarılması
    if time.time() - start_time >= time_period:
        averages = [{"object": label, "average_distance_cm": average(distance_values[label])} for label in distance_values]
        with open('distance.json', 'w') as f:
            json.dump(averages, f, indent=4)
        start_time = time.time()
        distance_values = {} # yeni ölçümler için resetleme

    # Sonuçları göster
    cv2.imshow("Object Detection", frame)

    # 'Esc' tuşuna basarak çıkış yap
    key = cv2.waitKey(1)
    if key == 27:
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
