import cv2
import numpy as np
import mediapipe as mp
import math


def findAngle(img, p1, p2, p3, lmList, draw=True):
    # İlgili noktaların x, y koordinatlarını al
    x1, y1 = lmList[p1][1:]  # p1 koordinatları
    x2, y2 = lmList[p2][1:]  # p2 koordinatları
    x3, y3 = lmList[p3][1:]  # p3 koordinatları

    # Açı hesaplama: atan2 ile iki vektör arasındaki açıyı hesaplar
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360  # Açıyı pozitif yap

    if draw:
        # Görüntü üzerinde çizimler yap (açıyı gösteren çizgiler ve noktalar)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 1. çizgi
        cv2.line(img, (x3, y3), (x2, y2), (0, 0, 255), 3)  # 2. çizgi

        # Noktalara işaretçi (çember) ekle
        cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)  # p1 noktası
        cv2.circle(img, (x2, y2), 10, (0, 255, 255), cv2.FILLED)  # p2 noktası
        cv2.circle(img, (x3, y3), 10, (0, 255, 255), cv2.FILLED)  # p3 noktası

        # Noktalara çevresel işaretçi (dış çember) ekle
        cv2.circle(img, (x1, y1), 15, (0, 255, 255))  # p1 çevresi
        cv2.circle(img, (x2, y2), 15, (0, 255, 255))  # p2 çevresi
        cv2.circle(img, (x3, y3), 15, (0, 255, 255))  # p3 çevresi

        # Açı bilgisini ekrana yazdır
        cv2.putText(img, str(int(angle)), (x2 - 40, y2 + 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    return angle


# Video kaynağını aç
cap = cv2.VideoCapture("video1.mp4")

# Mediapipe modülünü başlat
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

dir = 0  # Hareket yönünü takip etmek için değişken (1: yukarı, 0: aşağı)
count = 0  # Şınav sayısı
while True:
    success, img = cap.read()  # Videoyu oku
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Renk formatını RGB'ye çevir

    results = pose.process(imgRGB)  # Poz tespiti işlemi

    lmList = []  # Landmark listesi
    if results.pose_landmarks:  # Eğer poz tespiti yapılmışsa
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  # Poz bağlantılarını çiz

        # Poz işaretçilerini listeye ekle
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, _ = img.shape  # Görüntü boyutlarını al
            cx, cy = int(lm.x * w), int(lm.y * h)  # X, Y koordinatlarını elde et
            lmList.append([id, cx, cy])  # Landmark id, x ve y koordinatlarını lmList'e ekle

    # Landmark'lar bulunduğunda
    if len(lmList) != 0:
        # # Şınav tespiti için açı hesaplama
        angle = findAngle(img, 11, 13, 15, lmList)  # Kollar arasındaki açıyı bul
        per = np.interp(angle, (185, 245), (0, 100))  # Açıyı 0-100 arasında normalize et
        print(angle)  # Açıyı ekrana yazdır

        # Şınav sayma
        if per == 100:
            if dir == 0:  # Eğer tam üstteyse
                count += 0.5  # Sayacı artır
                dir = 1  # Yukarıya hareket etti
        if per == 0:
            if dir == 1:  # Eğer tam alttaysa
                count += 0.5  # Sayacı artır
                dir = 0  # Aşağıya hareket etti

        print(count)  # Şınav sayısını yazdır

        # Şınav sayısını görüntüye yaz
        cv2.putText(img, str(int(count)), (45, 125), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)

    # Görüntüyü ekranda göster
    cv2.imshow("image", img)
    cv2.waitKey(40)  # 40ms bekle, bu da 25 fps'ye denk gelir
