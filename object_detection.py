# import cv2
# import mediapipe as mp
# import time
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# from ultralytics import YOLO
# import csv
# import glob
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# import joblib

# def put_turkish_text(img, text, org, font_size=32, color=(0, 255, 0)):
#     pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(pil_img)
    
#     try:
#         font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", font_size)
#     except:
#         font = ImageFont.load_default()
    
#     draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
#     return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# # Türkçe çeviri sözlüğü
# tr_labels = {
#     'person': 'İnsan',
#     'bicycle': 'Bisiklet',
#     'car': 'Araba',
#     'bottle': 'Şişe',
#     'wine glass': 'Çay Bardağı',
#     'cup': 'Bardak',
#     'keyboard': 'Klavye',
#     'cell phone': 'Telefon',
#     'book': 'Kitap',
#     'toothbrush': 'Kalem',
#     'pen': 'Kalem',
#     'pencil': 'Kalem',
# }

# def get_tr_label(en_label):
#     return tr_labels.get(en_label, en_label)

# def count_fingers(hand_landmarks):
#     # Parmak ucu ve orta eklem noktaları
#     finger_tips = [8, 12, 16, 20]  # İşaret, orta, yüzük ve serçe parmak uçları
#     finger_mids = [6, 10, 14, 18]  # Parmakların orta eklemleri
#     thumb_tip = 4
#     thumb_mid = 3
    
#     fingers = 0
    
#     # Başparmak kontrolü (x koordinatına göre)
#     if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_mid].x:
#         fingers += 1
    
#     # Diğer parmaklar için kontrol (y koordinatına göre)
#     for tip, mid in zip(finger_tips, finger_mids):
#         # Eğer parmak ucu, orta eklemden yukarıdaysa parmak açık demektir
#         if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mid].y:
#             fingers += 1
            
#     return fingers

# def detect_custom_gesture(hand_landmarks):
#     # Parmak uçları ve eklem noktaları
#     index_tip = hand_landmarks.landmark[8]  # İşaret parmağı ucu
#     middle_tip = hand_landmarks.landmark[12]  # Orta parmak ucu
#     ring_tip = hand_landmarks.landmark[16]  # Yüzük parmağı ucu
#     pinky_tip = hand_landmarks.landmark[20]  # Serçe parmağı ucu
#     thumb_tip = hand_landmarks.landmark[4]  # Başparmak ucu
    
#     # Parmak orta eklemleri
#     index_mid = hand_landmarks.landmark[6]
#     middle_mid = hand_landmarks.landmark[10]
#     ring_mid = hand_landmarks.landmark[14]
#     pinky_mid = hand_landmarks.landmark[18]
    
#     # Özel hareket: Başparmak işaret ve orta parmak arasından
#     ozel_hareket = (
#         # Tüm parmaklar kapalı
#         index_tip.y > index_mid.y and
#         middle_tip.y > middle_mid.y and
#         ring_tip.y > ring_mid.y and
#         pinky_tip.y > pinky_mid.y and
#         # Başparmak işaret ve orta parmak arasında
#         thumb_tip.x > index_tip.x and
#         thumb_tip.x < middle_tip.x and
#         thumb_tip.y < index_tip.y
#     )
    
#     # Hareketi kontrol et ve mesaj döndür
#     if ozel_hareket:
#         return "NAH!"
    
#     return None

# def detect_facial_expression(face_landmarks):
#     """Yüz ifadelerini tespit eder"""
    
#     # Ağız noktaları
#     upper_lip = face_landmarks.landmark[13]  # Üst dudak
#     lower_lip = face_landmarks.landmark[14]  # Alt dudak
    
#     # Kaş noktaları
#     left_eyebrow = face_landmarks.landmark[282]  # Sol kaş
#     right_eyebrow = face_landmarks.landmark[52]  # Sağ kaş
    
#     # Göz noktaları
#     left_eye_top = face_landmarks.landmark[386]  # Sol göz üstü
#     left_eye_bottom = face_landmarks.landmark[374]  # Sol göz altı
#     right_eye_top = face_landmarks.landmark[159]  # Sağ göz üstü
#     right_eye_bottom = face_landmarks.landmark[145]  # Sağ göz altı
    
#     # Dudaklar arası mesafe (gülümseme kontrolü)
#     mouth_distance = abs(upper_lip.y - lower_lip.y)
    
#     # Kaşlar arası mesafe (üzgün/kızgın kontrolü)
#     eyebrow_distance = abs(left_eyebrow.y - right_eyebrow.y)
    
#     # Göz açıklığı
#     left_eye_open = abs(left_eye_top.y - left_eye_bottom.y)
#     right_eye_open = abs(right_eye_top.y - right_eye_bottom.y)
    
#     # İfadeleri kontrol et
#     if mouth_distance > 0.05:  # Gülümseme
#         return "Gülümsüyor 😊"
#     elif eyebrow_distance > 0.02:  # Kaşlar çatık
#         return "Kızgın 😠"
#     elif left_eye_open < 0.02 and right_eye_open < 0.02:  # Gözler kapalı
#         return "Gözler Kapalı 😌"
#     else:
#         return "Normal 😐"

# def detect_sign_language_letters(hand_landmarks):
#     """
#     C: Tüm parmaklar açık (uçlar tabanlardan yukarıda), uçlar tabanlara orta mesafede, başparmak ve işaret arası mesafe büyük, işaret ve serçe arası mesafe büyük
#     A: Tüm parmaklar kapalı (yumruk), başparmak yana çıkık
#     B: Tüm parmaklar açık ve bitişik, başparmak avuç içine kapalı
#     D: Sadece işaret parmağı açık, diğerleri kapalı, başparmak işarete yakın
#     E: Tüm parmaklar kıvrılmış, uçlar avuç içine değiyor, başparmak parmakların önünde
#     """
#     tips = [8, 12, 16, 20]
#     mids = [6, 10, 14, 18]
#     base = [5, 9, 13, 17]
#     thumb_tip = hand_landmarks.landmark[4]
#     index_tip = hand_landmarks.landmark[8]
#     pinky_tip = hand_landmarks.landmark[20]
#     wrist = hand_landmarks.landmark[0]
#     debug_msgs = []
#     # --- C harfi ---
#     fingers_open = all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base[i]].y for i, tip in enumerate(tips))
#     mid_curled = all(0.08 < abs(hand_landmarks.landmark[tip].y - hand_landmarks.landmark[base[i]].y) < 0.18 for i, tip in enumerate(tips))
#     thumb_index_dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
#     thumb_index_far = thumb_index_dist > 0.13
#     index_pinky_dist = ((index_tip.x - pinky_tip.x) ** 2 + (index_tip.y - pinky_tip.y) ** 2) ** 0.5
#     index_pinky_far = index_pinky_dist > 0.18
#     if fingers_open: debug_msgs.append('C-fingers_open')
#     if mid_curled: debug_msgs.append('C-mid_curled')
#     if thumb_index_far: debug_msgs.append('C-thumb_index_far')
#     if index_pinky_far: debug_msgs.append('C-index_pinky_far')
#     if fingers_open and mid_curled and thumb_index_far and index_pinky_far:
#         return "C", debug_msgs
#     # --- A harfi ---
#     fingers_closed = all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mid].y for tip, mid in zip(tips, mids))
#     thumb_out = abs(thumb_tip.x - wrist.x) > 0.10
#     if fingers_closed: debug_msgs.append('A-fingers_closed')
#     if thumb_out: debug_msgs.append('A-thumb_out')
#     if fingers_closed and thumb_out:
#         return "A", debug_msgs
#     # --- B harfi ---
#     fingers_open_b = all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mid].y for tip, mid in zip(tips, mids))
#     close_fingers = (
#         abs(hand_landmarks.landmark[8].x - hand_landmarks.landmark[12].x) < 0.07 and
#         abs(hand_landmarks.landmark[12].x - hand_landmarks.landmark[16].x) < 0.07 and
#         abs(hand_landmarks.landmark[16].x - hand_landmarks.landmark[20].x) < 0.07
#     )
#     thumb_in = abs(thumb_tip.x - wrist.x) < 0.08
#     if fingers_open_b: debug_msgs.append('B-fingers_open')
#     if close_fingers: debug_msgs.append('B-close_fingers')
#     if thumb_in: debug_msgs.append('B-thumb_in')
#     if fingers_open_b and close_fingers and thumb_in:
#         return "B", debug_msgs
#     # --- D harfi ---
#     index_open = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
#     others_closed = (
#         hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and
#         hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
#         hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
#     )
#     thumb_near_index = abs(thumb_tip.x - index_tip.x) < 0.07 and abs(thumb_tip.y - index_tip.y) < 0.07
#     if index_open: debug_msgs.append('D-index_open')
#     if others_closed: debug_msgs.append('D-others_closed')
#     if thumb_near_index: debug_msgs.append('D-thumb_near_index')
#     if index_open and others_closed and thumb_near_index:
#         return "D", debug_msgs
#     # --- E harfi ---
#     fingers_curled = all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip-2].y for tip in tips)
#     thumb_in_front = thumb_tip.y > index_tip.y and thumb_tip.x < index_tip.x
#     if fingers_curled: debug_msgs.append('E-fingers_curled')
#     if thumb_in_front: debug_msgs.append('E-thumb_in_front')
#     if fingers_curled and thumb_in_front:
#         return "E", debug_msgs
#     return None, debug_msgs

# def main():
#     # YOLO modelini yükle
#     model = YOLO('yolov8n.pt')

#     # MediaPipe el ve yüz tespitini başlat
#     mp_hands = mp.solutions.hands
#     mp_face_mesh = mp.solutions.face_mesh
#     mp_drawing = mp.solutions.drawing_utils
    
#     hands = mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=2,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )
    
#     face_mesh = mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )

#     # Kamerayı başlat
#     cap = cv2.VideoCapture(0)
#     # Ekran boyutunu büyüt
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 1280x720 çözünürlük
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     # FPS ve mod değişkenleri
#     fps_time = time.time()
#     frame_count = 0
#     fps = 0
#     mode = "el"  # Başlangıç modu

#     clf = joblib.load("hand_sign_svm.pkl")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Kamera okunamadı!")
#             break

#         # Sadece yatay çevirme yap
#         frame = cv2.flip(frame, 1)  # 1: sadece yatay çevirme

#         # FPS hesapla
#         frame_count += 1
#         if time.time() - fps_time >= 1.0:
#             fps = frame_count
#             frame_count = 0
#             fps_time = time.time()

#         # Mod bilgisini göster (yazı boyutları küçültüldü)
#       #  frame = put_turkish_text(frame, f"Mod: {'El Tespiti' if mode == 'el' else 'Nesne Tespiti' if mode == 'nesne' else 'Mimik Tespiti'}", 
#       #                 (10, 25), font_size=20)
#        # frame = put_turkish_text(frame, "Mod değiştirmek için: E -> El, N -> Nesne, M -> Mimik", 
#         #                       (10, 70), font_size=20)
#         #frame = put_turkish_text(frame, f"FPS: {fps}", (10, 45), font_size=20)

#         if mode == "mimik":
#             # Mimik tespiti modu
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             face_results = face_mesh.process(rgb_frame)
            
#             if face_results.multi_face_landmarks:
#                 for face_landmarks in face_results.multi_face_landmarks:
#                     # Yüz mesh'ini çiz
#                     mp_drawing.draw_landmarks(
#                         frame,
#                         face_landmarks,
#                         mp_face_mesh.FACEMESH_CONTOURS,
#                         landmark_drawing_spec=None,
#                         connection_drawing_spec=mp_drawing.DrawingSpec(
#                             color=(0, 255, 0), thickness=1, circle_radius=1)
#                     )
                    
#                     # Mimik tespiti
#                     expression = detect_facial_expression(face_landmarks)
#                     frame = put_turkish_text(frame, f"İfade: {expression}", 
#                                           (10, 110), font_size=20)

#         elif mode == "el":
#             # El tespiti modu
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             hand_results = hands.process(rgb_frame)
            
#             if hand_results.multi_hand_landmarks:
#                 hand_landmarks_list = list(hand_results.multi_hand_landmarks)
#                 detected = None
#                 debug_msgs = []
#                 for hand_landmarks in hand_landmarks_list:
#                     detected, debug_msgs = detect_sign_language_letters(hand_landmarks)
#                     if detected:
#                         frame = put_turkish_text(frame, f"{detected} Harfi Tespit Edildi", 
#                             (frame.shape[1]//2 - 200, frame.shape[0]//2), 
#                             font_size=72, 
#                             color=(0, 255, 0))
#                     mp_drawing.draw_landmarks(
#                         frame,
#                         hand_landmarks,
#                         mp_hands.HAND_CONNECTIONS,
#                         mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
#                         mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
#                     )
#                     for idx, lm in enumerate(hand_landmarks.landmark):
#                         cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
#                         frame = put_turkish_text(frame, f"{idx}", (cx, cy), font_size=14, color=(255, 0, 0))
#                 # Debug mesajlarını ekrana yazdır
#                 if debug_msgs:
#                     frame = put_turkish_text(frame, ", ".join(debug_msgs), (10, frame.shape[0]-30), font_size=16, color=(0,255,255))

#         else:
#             # Nesne tespiti modu (insan hariç)
#             yolo_results = model(frame, 
#                                conf=0.3,
#                                classes=[i for i in range(80) if model.names[i] != 'person'])
            
#             for result in yolo_results[0].boxes.data:
#                 x1, y1, x2, y2, conf, class_id = result
#                 class_name = model.names[int(class_id)]
#                 tr_class_name = get_tr_label(class_name)
                
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                 frame = put_turkish_text(frame, f"{tr_class_name} {conf:.2f}", 
#                                        (int(x1), int(y1)-20), font_size=18)

#         # Görüntüyü göster
#         cv2.imshow('Nesne, El ve Mimik Tespiti', frame)
        
#         # Tuş kontrolü
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('e'):
#             mode = "el"
#         elif key == ord('n'):
#             mode = "nesne"
#         elif key == ord('m'):
#             mode = "mimik"

#     # Temizlik
#     cap.release()
#     cv2.destroyAllWindows()
#     hands.close()
#     face_mesh.close()

# if __name__ == "__main__":
#     main()

# X, y = [], []
# for file in glob.glob("*_data.csv"):
#     label = file[0].upper()  # Dosya adı başındaki harf
#     data = np.loadtxt(file, delimiter=",")
#     if data.ndim == 1:
#         data = data.reshape(1, -1)
#     X.append(data)
#     y += [label] * data.shape[0]
# X = np.vstack(X)
# y = np.array(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# clf = SVC(kernel='rbf', probability=True)
# clf.fit(X_train, y_train)
# print("Test accuracy:", clf.score(X_test, y_test))
# joblib.dump(clf, "hand_sign_svm.pkl") 