import cv2
import mediapipe as mp
import numpy as np

# Kamera açma
cap = cv2.VideoCapture(0)  # Kamera açılır. '0' varsayılan kamerayı temsil eder.

# MediaPipe Yüz Tespiti Modülü Başlatma
face = mp.solutions.face_detection  # MediaPipe yüz tespiti modülünü alır.
face_detection = face.FaceDetection(0.5)  # Yüz tespiti modelini başlatır, 0.5 güven eşiğiyle.

# Arka plan resmini yükle
bg_image = cv2.imread("images.jpeg")  # Belirtilen arka plan resmini okur.

# Sabit daire boyutu (kamera mesafesine bakmaksızın)
circle_radius = 100  # Sabit bir daire yarıçapı belirler. Kamera mesafesinden bağımsız.

while True:
    # Kameradan görüntü oku
    success, img = cap.read()  # Kameradan bir kare (frame) okur.
    if not success:
        break  # Eğer görüntü alınamazsa döngü sonlanır.

    # Görüntüyü RGB formatına çevir (MediaPipe için)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR'yi RGB'ye dönüştürür, çünkü MediaPipe RGB kullanır.
    results = face_detection.process(imgRGB)  # Yüz tespiti yapar.

    # Görüntü boyutları
    h, w, _ = img.shape  # Görüntünün yüksekliği (h), genişliği (w) ve kanal sayısını alır.

    # Eğer arka plan resmi yüklenmemişse, hata mesajı ver
    if bg_image is None:
        print("Hata: Arka plan resmi bulunamadı!")
        break  # Eğer arka plan resmi yoksa, hata verir ve döngüden çıkar.

    # Arka planı mevcut çözünürlüğün %150 genişliğinde ve yüksekliğinde ayarla
    scale_factor = 1.5  # Arka planın boyutunu %50 büyütmek için kullanılan faktör.
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)  # Yeni genişlik ve yükseklik hesaplanır.

    # Arka planın boyutunu yeniden boyutlandır
    bg_resized = cv2.resize(bg_image, (new_w, new_h))  # Arka planı yeniden boyutlandırır.

    # Yüz tespiti yapıldıysa, tespit edilen yüzleri işle
    if results.detections:  # Eğer yüzler tespit edildiyse, işlem yapar.
        for detection in results.detections:  # Tespit edilen her yüz için döngü başlatılır.
            cords = detection.location_data.relative_bounding_box  # Yüzün konum ve boyut bilgileri.

            # Yüzün konum ve boyutunu hesapla
            x, y, width, height = int(cords.xmin * w), int(cords.ymin * h), int(cords.width * w), int(cords.height * h)  # Yüzün tespit edilen koordinatları gerçek piksel boyutlarına dönüştürülür.

            # Yüzü yukarı kaydırmak için 'y' değerini azaltabiliriz
            #y -= 30  # Yüzü 30 piksel yukarı kaydır (Bu satır yorum halindedir, aktif edebilirsiniz).

            # Yüzü sağa kaydırmak için 'x' değerini artırabiliriz
            #x += 50  # Yüzü 50 piksel sağa kaydır (Bu satır yorum halindedir, aktif edebilirsiniz).

            # Yüzü genişletilmiş arka planın ortasına yerleştir
            center_x = int(new_w // 2.13)  # Arka planın X eksenindeki merkezini hesaplar. '2.13' ile biraz sağa kaydırır.
            center_y = int(new_h // 2)  # Arka planın Y eksenindeki merkezini hesaplar.

            # Yüzü yeniden merkezleyerek konumlandır
            new_x = center_x - width // 2  # Yüzün yeni X koordinatını hesaplar.
            new_y = center_y - height // 2  # Yüzün yeni Y koordinatını hesaplar.

            # Yüzün sınırlarını kontrol et, eğer taşarsa doğru şekilde sabitle
            new_x = max(0, min(new_x, new_w - width))  # Yüzün X koordinatını sınırlamak için kontrol eder.
            new_y = max(0, min(new_y, new_h - height))  # Yüzün Y koordinatını sınırlamak için kontrol eder.

            # Yüz görüntüsünü al ve yeniden boyutlandır
            face_resized = cv2.resize(img[y:y + height, x:x + width], (width, height))  # Yüzün görüntüsünü alır ve arka plan için yeniden boyutlandırır.

            # Sabit daire boyutunda maske oluştur (kamera mesafesinden bağımsız)
            mask = np.zeros((height, width), dtype=np.uint8)  # Yüzün üzerine uygulanacak maske oluşturur. Başlangıçta tamamen siyah (şeffaf).
            center = (width // 2, height // 2)  # Dairenin merkezi, yüzün merkezi olarak belirlenir.
            cv2.circle(mask, center, circle_radius, 255, -1)  # Maske üzerinde sabit boyutta bir daire çizer. Daire içi tamamen beyaz (255).

            # Maskeyi yüz üzerine uygula
            face_circular = cv2.bitwise_and(face_resized, face_resized, mask=mask)  # Yüzün üzerine maskeyi uygular.

            # Maskeyi 3 kanalına dönüştür, böylece 3 renk kanalı ile işlem yapılabilir
            mask_3channel = cv2.merge([mask, mask, mask])  # Maskeyi 3 kanallı bir hale getirir (RGB).

            # Yüzü eklemeden önce, arka planı maske alanında temizle (şeffaflaştır)
            bg_resized[new_y:new_y + height, new_x:new_x + width] = cv2.bitwise_and(
                bg_resized[new_y:new_y + height, new_x:new_x + width],  # Arka planın bu kısmını maske ile temizler.
                bg_resized[new_y:new_y + height, new_x:new_x + width],
                mask=cv2.bitwise_not(mask)  # Maskenin tersi ile işlem yapılır, yani maske dışındaki alanı temizler.
            )

            # Yuvarlak yüzü arka plan üzerine ekle
            bg_resized[new_y:new_y + height, new_x:new_x + width] += face_circular  # Yüzü dairesel şekilde arka plana ekler.

    # Yeni genişletilmiş arka planı ekrana yansıt
    cv2.imshow("Expanded Background with Circular Face", bg_resized)  # Arka plan ile birlikte yuvarlak yüzü ekrana gösterir.

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(20) & 0xFF == ord('q'):  # 'q' tuşuna basıldığında döngüyü sonlandırır.
        break

# Kaynağı serbest bırak ve tüm pencereleri kapat
cap.release()  # Kamerayı serbest bırakır.
cv2.destroyAllWindows()  # Tüm pencereleri kapatır.
