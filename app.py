from flask import Flask, render_template, Response, jsonify, send_file
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
from PIL import Image, ImageDraw, ImageFont
import time
import os
from io import BytesIO
import random

app = Flask(__name__)
socketio = SocketIO(app)

# MediaPipe ve model yükleme
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
model = joblib.load("hand_sign_svm.pkl")

# Renk tanımlamaları
COLORS = {
    'background': (18, 18, 18),
    'text': (255, 255, 255),
    'accent': (0, 255, 170),
    'warning': (255, 82, 82),
    'success': (76, 175, 80),
    'overlay': (0, 0, 0, 180)
}

# Kontrol modları
CONTROL_MODE_EYE = "eye"
CONTROL_MODE_HEAD = "head"
current_control_mode = CONTROL_MODE_EYE

# Metin alanı ayarları
max_chars_per_line = 20
max_lines = 3
lines = [""]
last_time = time.time()
last_pred = None
last_blink_time = time.time()
blink_cooldown = 0.5
cursor_visible = True
cursor_time = time.time()

# --- Sıralı Eğitim Modu State ---
SEQUENCE_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z']
sequence_state = {
    'current_index': 0,
    'score': 0,
    'success': False,
    'success_time': 0,
}

# --- Rastgele Pratik Modu State ---
PRACTICE_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z']
practice_state = {
    'target_letter': random.choice(PRACTICE_ALPHABET),
    'score': 0,
    'total': 0,
    'success': False,
    'success_time': 0,
    'waiting_until': 0,
}

# --- Kelime Pratik Modu State ---
WORD_LIST = [
    "ARABA", "MASA", "KALEM", "DEFTER",
    "OKUL", "PARK", "MARKET", "TELEFON",
    "BARDAK", "TABAK", "SANDALYE", "KAPI"
]
word_state = {
    'target_word': random.choice(WORD_LIST),
    'current_index': 0,
    'score': 0,
    'total': 0,
    'success': False,
    'success_time': 0,
}

# --- Hız Testi Modu State ---
speed_state = {
    'target_word': random.choice(WORD_LIST),
    'current_index': 0,
    'score': 0,
    'start_time': 0,
    'duration': 60,  # saniye
    'success': False,
    'success_time': 0,
    'game_over': False,
}

def detect_eye_blinks(face_landmarks):
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
    if face_landmarks is None:
        return None
    nose_tip = face_landmarks.landmark[1]
    return nose_tip.y

def create_rounded_rectangle(img, x1, y1, x2, y2, radius, color):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    return img

def put_turkish_text(img, text, position, font_size=32, color=(255, 255, 255), background=None):
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

def process_frame(frame_data):
    global lines, last_blink_time, current_control_mode
    try:
        if not frame_data or ',' not in frame_data:
            return None, None
        encoded_data = frame_data.split(',')[1]
        if not encoded_data:
            return None, None
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        if nparr.size == 0:
            return None, None
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return None, None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)
        face_results = face_mesh.process(rgb)
        left_eye_closed, right_eye_closed = detect_eye_blinks(
            face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        )
        head_position = detect_head_position(
            face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        )
        HEAD_UP_THRESHOLD = 0.5
        HEAD_DOWN_THRESHOLD = 0.7
        prediction = None
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])
                pred = model.predict([data])[0]
                prediction = pred
        # --- Göz kırpma ile yazı ekleme/silme ---
        now = time.time()
        blink_cooldown = 0.5
        if (now - last_blink_time > blink_cooldown):
            if current_control_mode == CONTROL_MODE_EYE:
                if right_eye_closed and not left_eye_closed:
                    if lines and lines[-1]:
                        lines[-1] = lines[-1][:-1]
                    elif len(lines) > 1:
                        lines.pop()
                        if not lines:
                            lines = [""]
                    last_blink_time = now
                elif left_eye_closed and not right_eye_closed:
                    if prediction:
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
            elif current_control_mode == CONTROL_MODE_HEAD and head_position is not None:
                if head_position > HEAD_DOWN_THRESHOLD:
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
                elif head_position < HEAD_UP_THRESHOLD:
                    if (now - last_blink_time > blink_cooldown):
                        if lines and lines[-1]:
                            lines[-1] = lines[-1][:-1]
                        elif len(lines) > 1:
                            lines.pop()
                            if not lines:
                                lines = [""]
                        last_blink_time = now
        # --- Sadeleştirilmiş görüntü ---
        # (İstenirse sadece tahmin edilen harfi köşeye ekleyebiliriz)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64, prediction
    except Exception as e:
        print(f"Görüntü işleme hatası: {str(e)}")
        return None, None

def load_reference_image_base64(letter):
    image_path = os.path.join('hand_sign_images', f'{letter}.png')
    if os.path.exists(image_path):
        with open(image_path, 'rb') as img_file:
            return 'data:image/png;base64,' + base64.b64encode(img_file.read()).decode('utf-8')
    return ''

def color_word_html(word, current_index):
    html = ''
    for i, letter in enumerate(word):
        if i < current_index:
            html += f'<span style="color:#4caf50;">{letter}</span>'
        elif i == current_index:
            html += f'<span style="color:#00ffaa; text-decoration:underline;">{letter}</span>'
        else:
            html += f'<span style="color:#fff;">{letter}</span>'
    return html

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(data):
    try:
        frame_base64, prediction = process_frame(data)
        if frame_base64 is not None:
            socketio.emit('result', {
                'frame': frame_base64,
                'prediction': prediction,
                'text_lines': lines
            })
    except Exception as e:
        print(f"Frame işleme hatası: {str(e)}")

@socketio.on('key_press')
def handle_key_press(data):
    global current_control_mode, lines
    
    key = data.get('key')
    if key == 'c':
        lines = [""]
    elif key == 'g':
        current_control_mode = CONTROL_MODE_EYE
    elif key == 'k':
        current_control_mode = CONTROL_MODE_HEAD

@socketio.on('start_sequence_training')
def start_sequence_training():
    sequence_state['current_index'] = 0
    sequence_state['score'] = 0
    sequence_state['success'] = False
    sequence_state['success_time'] = 0

@socketio.on('sequence_frame')
def handle_sequence_frame(data):
    try:
        # Frame decode
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)
        target_letter = SEQUENCE_ALPHABET[sequence_state['current_index']]
        prediction = None
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])
                pred = model.predict([data])[0]
                prediction = pred
        # Başarı kontrolü
        success = False
        current_time = time.time()
        
        if prediction == target_letter:
            if not sequence_state['success']:
                # İlk doğru harf yapıldığında
                sequence_state['success'] = True
                sequence_state['success_time'] = current_time
                success = True
            elif current_time - sequence_state['success_time'] > 1.0:
                # 1 saniye geçtiyse yeni harfe geç
                sequence_state['score'] += 1
                sequence_state['current_index'] = (sequence_state['current_index'] + 1) % len(SEQUENCE_ALPHABET)
                sequence_state['success'] = False
                sequence_state['success_time'] = 0
            else:
                # 1 saniye dolmadan önce success true kalsın
                success = True
        else:
            # Yanlış harf yapıldığında
            sequence_state['success'] = False
            sequence_state['success_time'] = 0
            success = False
            
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        # Progress
        progress = f"İlerleme: {sequence_state['current_index']+1}/{len(SEQUENCE_ALPHABET)}"
        # Skor
        score = f"Skor: {sequence_state['score']}"
        # Referans resim
        ref_image = load_reference_image_base64(target_letter)
        socketio.emit('sequence_result', {
            'frame': frame_base64,
            'target_letter': target_letter,
            'progress': progress,
            'ref_image': ref_image,
            'success': success,
            'score': score
        })
    except Exception as e:
        print(f"Sıralı eğitim frame hatası: {str(e)}")

@socketio.on('start_practice_training')
def start_practice_training():
    try:
        practice_state['target_letter'] = random.choice(PRACTICE_ALPHABET)
        practice_state['score'] = 0
        practice_state['total'] = 0
        practice_state['success'] = False
        practice_state['success_time'] = 0
        practice_state['waiting_until'] = 0
        print("Pratik modu başlatıldı")
    except Exception as e:
        print(f"Pratik modu başlatma hatası: {e}")

@socketio.on('practice_frame')
def handle_practice_frame(data):
    try:
        if not data or ',' not in data:
            return
            
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)
        
        current_time = time.time()
        prediction = None
        hand_detected = False
        verification_complete = False
        
        # Bekleme süresi kontrolü
        if practice_state.get('waiting_until', 0) > current_time:
            socketio.emit('practice_update', {
                'frame': base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8'),
                'target_letter': practice_state['target_letter'],
                'prediction': None,
                'score': f"Skor: {practice_state['score']}/{practice_state['total']}",
                'success': False,
                'verification_complete': False,
                'progress': 0,
                'hand_detected': False
            })
            return
        
        # El algılama ve tahmin
        if hand_results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in hand_results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])
                pred = model.predict([data])[0]
                prediction = pred
        
        # Doğrulama süreci
        if hand_detected:
            if not practice_state['success']:
                practice_state['success'] = True
                practice_state['success_time'] = current_time
            
            elapsed_time = current_time - practice_state['success_time']
            
            # 1 saniye el algılandıktan sonra doğrulama yap
            if elapsed_time >= 1.0:
                verification_complete = True
                if prediction == practice_state['target_letter']:
                    practice_state['score'] += 1
                    practice_state['total'] += 1
                    practice_state['success'] = False
                    practice_state['success_time'] = 0
                    practice_state['waiting_until'] = current_time + 2.0  # 2 saniye bekle
                else:
                    practice_state['total'] += 1
                    practice_state['success'] = False
                    practice_state['success_time'] = 0
                    practice_state['waiting_until'] = current_time + 2.0  # 2 saniye bekle
        else:
            practice_state['success'] = False
            practice_state['success_time'] = 0
        
        # Bekleme süresi bittiyse yeni harf seç
        if current_time >= practice_state.get('waiting_until', 0) and practice_state.get('waiting_until', 0) > 0:
            practice_state['target_letter'] = random.choice(PRACTICE_ALPHABET)
            practice_state['waiting_until'] = 0
        
        # Frame'i encode et
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Sonuçları gönder
        socketio.emit('practice_update', {
            'frame': frame_base64,
            'target_letter': practice_state['target_letter'],
            'prediction': prediction if hand_detected else None,
            'score': f"Skor: {practice_state['score']}/{practice_state['total']}",
            'success': verification_complete and prediction == practice_state['target_letter'],
            'verification_complete': verification_complete,
            'progress': min(100, int((current_time - practice_state['success_time']) * 100)) if practice_state['success'] else 0,
            'hand_detected': hand_detected and not verification_complete  # Doğrulama tamamlandıysa el algılama göstergesini kapat
        })
        
    except Exception as e:
        print(f"Pratik modu hatası: {e}")

@socketio.on('start_word_training')
def start_word_training():
    word_state['target_word'] = random.choice(WORD_LIST)
    word_state['current_index'] = 0
    word_state['score'] = 0
    word_state['total'] = 0
    word_state['success'] = False
    word_state['success_time'] = 0

@socketio.on('word_frame')
def handle_word_frame(data):
    try:
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)
        prediction = None
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])
                pred = model.predict([data])[0]
                prediction = pred
        # Başarı kontrolü
        success = False
        target_word = word_state['target_word']
        idx = word_state['current_index']
        if idx < len(target_word) and prediction == target_word[idx]:
            if not word_state['success']:
                word_state['success'] = True
                word_state['success_time'] = time.time()
            elif time.time() - word_state['success_time'] > 1.0:
                word_state['current_index'] += 1
                word_state['success'] = False
                word_state['success_time'] = 0
                success = True
                # Kelime tamamlandıysa
                if word_state['current_index'] >= len(target_word):
                    word_state['score'] += 1
                    word_state['total'] += 1
                    word_state['target_word'] = random.choice(WORD_LIST)
                    word_state['current_index'] = 0
        else:
            word_state['success'] = False
            word_state['success_time'] = 0
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        # Skor
        score = f"Tamamlanan Kelime: {word_state['score']}/{word_state['total']}"
        # Hedef kelimeyi renklendir
        target_word_colored = color_word_html(word_state['target_word'], word_state['current_index'])
        socketio.emit('word_result', {
            'frame': frame_base64,
            'target_word_colored': target_word_colored,
            'score': score,
            'success': success
        })
    except Exception as e:
        print(f"Kelime pratik frame hatası: {str(e)}")

@socketio.on('start_speed_test')
def start_speed_test():
    speed_state['target_word'] = random.choice(WORD_LIST)
    speed_state['current_index'] = 0
    speed_state['score'] = 0
    speed_state['start_time'] = time.time()
    speed_state['success'] = False
    speed_state['success_time'] = 0
    speed_state['game_over'] = False

@socketio.on('speed_frame')
def handle_speed_frame(data):
    try:
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)
        prediction = None
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])
                pred = model.predict([data])[0]
                prediction = pred
        # Süre kontrolü
        elapsed = time.time() - speed_state['start_time']
        time_left = max(0, int(speed_state['duration'] - elapsed))
        game_over = time_left == 0
        if game_over and not speed_state['game_over']:
            speed_state['game_over'] = True
        # Başarı kontrolü
        success = False
        if not speed_state['game_over']:
            target_word = speed_state['target_word']
            idx = speed_state['current_index']
            if idx < len(target_word) and prediction == target_word[idx]:
                if not speed_state['success']:
                    speed_state['success'] = True
                    speed_state['success_time'] = time.time()
                elif time.time() - speed_state['success_time'] > 1.0:
                    speed_state['current_index'] += 1
                    speed_state['success'] = False
                    speed_state['success_time'] = 0
                    success = True
                    # Kelime tamamlandıysa
                    if speed_state['current_index'] >= len(target_word):
                        speed_state['score'] += 1
                        speed_state['target_word'] = random.choice(WORD_LIST)
                        speed_state['current_index'] = 0
            else:
                speed_state['success'] = False
                speed_state['success_time'] = 0
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        # Skor
        score = f"Tamamlanan Kelime: {speed_state['score']}"
        # Hedef kelimeyi renklendir
        target_word_colored = color_word_html(speed_state['target_word'], speed_state['current_index'])
        # Kalan süre
        time_left_str = f"Kalan Süre: {time_left} sn"
        socketio.emit('speed_result', {
            'frame': frame_base64,
            'target_word_colored': target_word_colored,
            'score': score,
            'time_left': time_left_str,
            'success': success,
            'game_over': speed_state['game_over']
        })
    except Exception as e:
        print(f"Hız testi frame hatası: {str(e)}")

if __name__ == '__main__':
    socketio.run(app, debug=True) 