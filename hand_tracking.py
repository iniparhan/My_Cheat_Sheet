import cv2
import mediapipe as mp
import time     # Digunakan untuk menghitung frame per second (FPS).

cap = cv2.VideoCapture(0)   # Membuka kamera dengan ID 0 (kamera utama). Jika memiliki kamera eksternal, bisa ganti dengan 1 atau 2, misalnya cv2.VideoCapture(1).

mpHands = mp.solutions.hands    # Modul untuk mendeteksi tangan.
hands = mpHands.Hands()     # Membuat objek deteksi tangan.
mpDraw = mp.solutions.drawing_utils     # Digunakan untuk menggambar landmark tangan.

pTime = 0   # Waktu frame sebelumnya.
cTime = 0   # Waktu frame saat ini.

while True:
    success, img = cap.read()   # Membaca frame dari kamera.
    img = cv2.flip(img, 1)  # Membalik gambar (mirror mode) agar tampilan lebih natural.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Mengubah format warna dari BGR ke RGB, karena Mediapipe bekerja dengan RGB.
    results = hands.process(imgRGB)     # Mendeteksi tangan dan landmark-nya dalam gambar.

    if results.multi_hand_landmarks:    # Berisi data landmark tangan jika terdeteksi.
        for handLms in results.multi_hand_landmarks:    # Untuk setiap tangan yang terdeteksi.
            for id, lm in enumerate(handLms.landmark):  # Untuk setiap landmark (21 titik) di tangan.
                h, w, c = img.shape     # Gambar memiliki ukuran (height, width, channel).
                cx, cy = int(lm.x * w), int(lm.y * h)   # Koordinat landmark (lm.x, lm.y) diberikan dalam skala 0-1. Mengalikan lm.x * w dan lm.y * h untuk mendapatkan koordinat piksel sebenarnya (cx, cy).
                print(id, cx, cy)   # Menampilkan ID landmark dan posisi dalam piksel.
                # Kode ini menggambar lingkaran berwarna ungu (255, 0, 255) di titik-titik ini.
                if id == 4 : # 4 → Ujung ibu jari
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if id == 8 : # 8 → Ujung telunjuk
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if id == 12 : # 12 → Ujung jari tengah
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if id == 16 : # 16 → Ujung jari manis
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                if id == 20 : # 20 → Ujung kelingking
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)   # Menghubungkan titik-titik landmark agar membentuk struktur tangan.

    cTime = time.time()
    fps = 1 / (cTime - pTime)   # dihitung dengan rumus 1 / waktu antara dua frame.
    pTime = cTime

    cv2.putText(img, str(int(fps)), 
                (10, 70),   # Menampilkan teks FPS pada posisi (10,70).
                cv2.FONT_HERSHEY_PLAIN, 3,  # Ukuran font = 3, warna ungu (255, 0, 255)
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF     # Menunggu input tombol selama 1 ms.
    if key == ord('q'):
        break

cap.release()   # Menutup kamera.
cv2.destroyAllWindows()     # Menutup semua jendela OpenCV.
