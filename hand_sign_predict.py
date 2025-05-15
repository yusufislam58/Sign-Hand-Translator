import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import os
import threading
import pygame

pygame.mixer.init()

COLORS = {
    'background': (18, 18, 18),      # Koyu arka plan
    'text': (255, 255, 255),         # Beyaz metin
    'accent': (0, 255, 170),         # Turkuaz vurgu
    'warning': (255, 82, 82),        # Kırmızı uyarı
    'success': (76, 175, 80),        # Yeşil başarı
    'overlay': (0, 0, 0, 180)        # Yarı saydam siyah
}

clf = joblib.load("hand_sign_svm.pkl")

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 1920x1080 çözünürlük
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

max_chars_per_line = 20  
max_lines = 3
lines = [""]
last_time = time.time()
last_pred = None
last_blink_time = time.time()
blink_cooldown = 0.5 
cursor_visible = True
cursor_time = time.time()

CONTROL_MODE_EYE = "eye"
CONTROL_MODE_HEAD = "head"
current_control_mode = CONTROL_MODE_EYE

def detect_eye_blinks(face_landmarks):
    """Sağ ve sol göz kırpma tespiti"""
    if face_landmarks is None:
        return False, False
    
    left_eye_top = face_landmarks.landmark[386]
    left_eye_bottom = face_landmarks.landmark[374]
    
    right_eye_top = face_landmarks.landmark[159]
    right_eye_bottom = face_landmarks.landmark[145]
    
    left_eye_open = abs(left_eye_top.y - left_eye_bottom.y)
    right_eye_open = abs(right_eye_top.y - right_eye_bottom.y)
    
    left_eye_closed = left_eye_open < 0.02
    right_eye_closed = right_eye_open < 0.02
    
    return left_eye_closed, right_eye_closed

def detect_head_position(face_landmarks):
    """Kafa pozisyonunu tespit et"""
    if face_landmarks is None:
        return None
    
    nose_tip = face_landmarks.landmark[1]
    return nose_tip.y

def create_rounded_rectangle(img, x1, y1, x2, y2, radius, color):
    """Yumuşak köşeli dikdörtgen çiz"""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    return img

def put_turkish_text(img, text, position, font_size=32, color=(255, 255, 255), background=None):
    """Türkçe karakterleri destekleyen metin yazma fonksiyonu"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text_bbox = draw.textbbox(position, text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    if background:
        padding = 10
        draw.rectangle(
            [(position[0] - padding, position[1] - padding),
             (position[0] + text_width + padding, position[1] + text_height + padding)],
            fill=background
        )
    
    draw.text(position, text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def speak_word(word):
    """Kelimeyi sesli okuma fonksiyonu"""
    try:
        tts = gTTS(text=word, lang='tr')
        tts.save("temp_word.mp3")
        pygame.mixer.music.load("temp_word.mp3")
        pygame.mixer.music.play()
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove("temp_word.mp3")
    except Exception as e:
        print(f"Ses okuma hatası: {e}")

def speak_text(text):
    """Metni sesli okuma fonksiyonu"""
    try:
        tts = gTTS(text=text, lang='tr')
        tts.save("temp_text.mp3")
        pygame.mixer.music.load("temp_text.mp3")
        pygame.mixer.music.play()
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove("temp_text.mp3")
    except Exception as e:
        print(f"Ses okuma hatası: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)
    left_eye_closed, right_eye_closed = detect_eye_blinks(face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None)
    head_position = detect_head_position(face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None)
    
    HEAD_UP_THRESHOLD = 0.5    # Yukarı bakma eşiği
    HEAD_DOWN_THRESHOLD = 0.7  # Aşağı bakma eşiği
    
    prediction = None
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y])
            pred = clf.predict([data])[0]
            prediction = pred

    now = time.time()
    if (now - last_blink_time > blink_cooldown):
        if current_control_mode == CONTROL_MODE_EYE:
            if right_eye_closed and not left_eye_closed:  # Sadece sağ göz kapalı - silme işlemi
                if lines and lines[-1]:  # Eğer son satırda karakter varsa
                    lines[-1] = lines[-1][:-1]  # Son karakteri sil
                elif len(lines) > 1:  # Eğer birden fazla satır varsa
                    lines.pop()  # Son satırı sil
                    if not lines:  # Eğer hiç satır kalmadıysa
                        lines = [""]
                last_blink_time = now
            elif left_eye_closed and not right_eye_closed:  # Sadece sol göz kapalı - yazma işlemi
                if prediction:  # Sadece tahmin varsa ekle
                    char_to_add = ' ' if prediction == 'BOSLUK' else prediction
                    if len(lines[-1]) < max_chars_per_line:
                        lines[-1] += char_to_add
                        # Eğer boşluk eklendiyse ve öncesinde kelime varsa, kelimeyi oku
                        if char_to_add == ' ' and lines[-1].strip():
                            last_word = lines[-1].strip().split()[-1]
                            # Sesli okuma işlemini ayrı bir thread'de başlat
                            threading.Thread(target=speak_word, args=(last_word,), daemon=True).start()
                    else:
                        if len(lines) < max_lines:
                            lines.append(char_to_add)
                        else:
                            lines.pop(0)
                            lines.append(char_to_add)
                    last_blink_time = now
        elif current_control_mode == CONTROL_MODE_HEAD and head_position is not None:
            if head_position > HEAD_DOWN_THRESHOLD:  # Kafa aşağıda - yazma
                if prediction and (now - last_blink_time > blink_cooldown):
                    char_to_add = ' ' if prediction == 'BOSLUK' else prediction
                    if len(lines[-1]) < max_chars_per_line:
                        lines[-1] += char_to_add
                    else:
                        if len(lines) < max_lines:
                            lines.append(char_to_add)
                        else:
                            lines.pop(0)
                            lines.append(char_to_add)
                    last_blink_time = now
            elif head_position < HEAD_UP_THRESHOLD:  # Kafa yukarıda - silme
                if (now - last_blink_time > blink_cooldown):
                    if lines and lines[-1]:  # Eğer son satırda karakter varsa
                        lines[-1] = lines[-1][:-1]  # Son karakteri sil
                    elif len(lines) > 1:  # Eğer birden fazla satır varsa
                        lines.pop()  # Son satırı sil
                        if not lines:  # Eğer hiç satır kalmadıysa
                            lines = [""]
                    last_blink_time = now

    text_area_height = max_lines * 70 + 50  
    frame = create_rounded_rectangle(frame, 20, 20, frame.shape[1]-20, text_area_height, 15, COLORS['background'])
    
    frame = put_turkish_text(frame, "Yazı Alanı", (40, 30), font_size=20, 
                           color=COLORS['accent'], background=COLORS['background'])
    
    for i, line in enumerate(lines):
        frame = put_turkish_text(frame, line, (40, 70 + i*70), font_size=36, 
                               color=COLORS['text'])
    
    if cursor_visible:
        cursor_x = 40 + len(lines[-1]) * 25
        cursor_y = 70 + (len(lines) - 1) * 70
        frame = put_turkish_text(frame, "|", (cursor_x, cursor_y), font_size=36, 
                               color=COLORS['accent'])
    
    if prediction:
        frame = put_turkish_text(frame, f"Tahmin: {prediction}", 
                               (40, text_area_height + 20), font_size=24, 
                               color=COLORS['success'], background=COLORS['background'])
    
    control_panel_y = frame.shape[0] - 160 
    frame = create_rounded_rectangle(frame, 20, control_panel_y, frame.shape[1]-20, frame.shape[0]-20, 15, COLORS['background'])
    
    mode_text = "Göz Kontrolü" if current_control_mode == CONTROL_MODE_EYE else "Kafa Kontrolü"
    frame = put_turkish_text(frame, f"Mod: {mode_text}", 
                           (40, control_panel_y + 25), font_size=24, 
                           color=COLORS['accent'])
    
    status_y = control_panel_y + 25
    if current_control_mode == CONTROL_MODE_EYE:
        frame = put_turkish_text(frame, f"Sol Göz: {'●' if right_eye_closed  else '○'}", 
                               (540, status_y), font_size=24, 
                               color=COLORS['text'])
        
        frame = put_turkish_text(frame, f"Sağ Göz: {'●' if left_eye_closed else '○'}", 
                               (300, status_y), font_size=24, 
                               color=COLORS['text'])
        
        frame = put_turkish_text(frame, "Sol Göz: Sil | Sağ Göz: Yaz", 
                               (40, control_panel_y + 60), font_size=20, 
                               color=COLORS['text'])
    else:
        if head_position is not None:
            head_status = "↑" if head_position < HEAD_UP_THRESHOLD else "↓" if head_position > HEAD_DOWN_THRESHOLD else "→"
            frame = put_turkish_text(frame, f"Kafa: {head_status}", 
                                   (540, status_y), font_size=24, 
                                   color=COLORS['accent'])
            
            frame = put_turkish_text(frame, "Yukarı: Sil | Aşağı: Yaz", 
                                   (40, control_panel_y + 60), font_size=20, 
                                   color=COLORS['text'])
    
    frame = put_turkish_text(frame, "G: Göz Modu | K: Kafa Modu | C: Temizle | E: Oku | Q: Çıkış", 
                           (40, control_panel_y + 100), font_size=20, 
                           color=COLORS['text'])
    
    cv2.namedWindow("El Harfi Tahmini", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("El Harfi Tahmini", 1280, 720)
    cv2.imshow("El Harfi Tahmini", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  
        lines = [""]
        last_pred = None
    elif key == ord('g'):  
        current_control_mode = CONTROL_MODE_EYE
    elif key == ord('k'):  
        current_control_mode = CONTROL_MODE_HEAD
    elif key == ord('e'): 
        full_text = ' '.join([line.strip() for line in lines if line.strip()])
        if full_text:  
            threading.Thread(target=speak_text, args=(full_text,), daemon=True).start()

cap.release()
cv2.destroyAllWindows() 