# OpenCV ve Mediapipe kütüphanelerini import ediyoruz
import cv2
import mediapipe as mp

# Video dosyasını açmak için VideoCapture kullanıyoruz. Burada "video3.mp4" adlı dosya okunuyor.
cap = cv2.VideoCapture(0)

# Mediapipe'in yüz algılama modülünü initialize ediyoruz.
# FaceDetection sınıfını çağırıyoruz ve min_detection_confidence parametresiyle algılama hassasiyetini belirtiyoruz (0.20).
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.20)

# Çizim işlemleri için Mediapipe'in drawing_utils modülünü kullanıyoruz.
mpDraw = mp.solutions.drawing_utils

# Video karesini okumak ve işlemek için sonsuz bir döngü başlatıyoruz.
while True:
    # Videodan bir kare okuyup başarı durumunu (success) ve görüntüyü (img) alıyoruz.
    success, img = cap.read()

    # Eğer video sona ererse veya kare okunamazsa döngüyü sonlandırabiliriz (bu kodda kontrol edilmemiş).
    if not success:
        break

    # OpenCV'de görüntü formatı BGR'dir. Mediapipe ise RGB formatını kullanır.
    # Bu nedenle görüntüyü BGR'den RGB'ye dönüştürüyoruz.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # FaceDetection sınıfı ile görüntüde yüz algılama işlemini gerçekleştiriyoruz.
    results = faceDetection.process(imgRGB)

    if results.detections:  # eğer en az bir yüz algılandıysa bir liste döner.
        # Algılanan her yüz için döngü başlatıyoruz.
        for id, detection in enumerate(results.detections):
            # Her bir yüzün konum bilgilerini alıyoruz (bounding box bilgileri).
            bboxC = detection.location_data.relative_bounding_box

            # Görüntünün genişlik (w) ve yükseklik (h) bilgilerini alıyoruz.
            h, w, _ = img.shape
            # Bağıl koordinatları piksel değerlerine dönüştürüyoruz:
            # bboxC.xmin: Yüzün sol üst köşesinin yatay orantılı başlangıç noktası (0-1 arasında)
            # bboxC.ymin: Yüzün sol üst köşesinin dikey orantılı başlangıç noktası (0-1 arasında)
            # bboxC.width: Yüzün orantılı genişliği
            # bboxC.height: Yüzün orantılı yüksekliği
            bbox = (
                int(bboxC.xmin * w),  # Oran x genişlik → Sol üst köşenin X piksel değeri
                int(bboxC.ymin * h),  # Oran x yükseklik → Sol üst köşenin Y piksel değeri
                int(bboxC.width * w),  # Oran x genişlik → Yüzün piksel genişliği
                int(bboxC.height * h)  # Oran x yükseklik → Yüzün piksel yüksekliği
            )
            # Algılanan yüzün etrafına dikdörtgen çiziyoruz.
            # Bbox koordinatları kullanılarak, sarı (0, 255, 255) renkte ve kalınlığı 2 olan bir çerçeve çiziliyor.
            cv2.rectangle(img, bbox, (0, 255, 255), 2)


    # İşlenen görüntüyü bir pencere içinde gösteriyoruz.
    cv2.imshow("img", img)

    # Her kareyi göstermek için 10 ms bekliyoruz. Ayrıca, 'q' tuşuna basıldığında döngü sonlandırılabilir.
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Video işleme tamamlandıktan sonra kaynakları serbest bırakıyoruz.
cap.release()
cv2.destroyAllWindows()
