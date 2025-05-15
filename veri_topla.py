import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
cap = cv2.VideoCapture(0)

harf = input("Kaydetmek istediğin harfi gir (örn: A): ").upper()

# Letters klasörü yoksa oluştur
if not os.path.exists("Letters"):
    os.makedirs("Letters")

csv_path = os.path.join("Letters", f"{harf}_data.csv")
csv_file = open(csv_path, "a", newline="")
csv_writer = csv.writer(csv_file)

print("Elini kameraya göster ve 's' tuşuna basınca veri kaydedilecek. Çıkmak için 'q'.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)
    cv2.imshow("Veri Toplama", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and results.multi_hand_landmarks:
        # 21 landmark x ve y değerlerini kaydet
        data = []
        for lm in results.multi_hand_landmarks[0].landmark:
            data.extend([lm.x, lm.y])
        csv_writer.writerow(data)
        print(f"{harf} için veri kaydedildi.")
    elif key == ord('q'):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()