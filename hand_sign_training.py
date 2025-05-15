import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import pygame
import joblib
import os
import random
import subprocess

# Ses için pygame başlat
pygame.mixer.init()

# Başarı sesi yükle
try:
    success_sound = pygame.mixer.Sound("sounds/rightanswer.mp3")
    word_complete_sound = pygame.mixer.Sound("sounds/correct.mp3")
except:
    print("UYARI: Ses dosyaları bulunamadı! Sesler devre dışı.")
    success_sound = None
    word_complete_sound = None

# Eğitilmiş modeli yükle
try:
    clf = joblib.load("hand_sign_svm.pkl")
except:
    print("HATA: El işareti modeli (hand_sign_svm.pkl) bulunamadı!")
    exit(1)

# Türkçe alfabe (büyük harfler)
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
            'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z']

# Pratik için kelimeler
WORDS = [
    "ARABA", "MASA", "KALEM", "DEFTER",
    "OKUL", "PARK", "MARKET", "TELEFON",
    "BARDAK", "TABAK", "SANDALYE", "KAPI"
]

# Renk paleti
COLORS = {
    'background': (18, 18, 18),      # Koyu arka plan
    'text': (255, 255, 255),         # Beyaz metin
    'accent': (0, 255, 170),         # Turkuaz vurgu
    'warning': (255, 82, 82),        # Kırmızı uyarı
    'success': (76, 175, 80),        # Yeşil başarı
    'overlay': (0, 0, 0, 180)        # Yarı saydam siyah
}

# MediaPipe el tanıma ayarları
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

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

def calculate_hand_similarity(landmarks1, landmarks2, threshold=0.1):
    """İki el pozisyonu arasındaki benzerliği hesapla"""
    if landmarks1 is None or landmarks2 is None:
        return 0.0
    
    total_diff = 0
    num_landmarks = len(landmarks1.landmark)
    
    for i in range(num_landmarks):
        lm1 = landmarks1.landmark[i]
        lm2 = landmarks2.landmark[i]
        
        # X ve Y koordinatları arasındaki farkı hesapla
        diff = np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)
        total_diff += diff
    
    similarity = 1.0 - (total_diff / num_landmarks)
    return max(0.0, similarity)

def load_reference_image(letter):
    """Referans el işareti fotoğrafını yükle"""
    image_path = os.path.join('hand_sign_images', f'{letter}.png')
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            # Resmi yeniden boyutlandır (300x300 piksel)
            return cv2.resize(img, (150, 150))
    return None

def show_main_menu():
    """Ana menüyü göster ve mod seçimini al"""
    menu_window = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.namedWindow("İşaret Dili Asistanı", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("İşaret Dili Asistanı", 1280, 720)

    while True:
        menu = menu_window.copy()
        
        # Menü başlığı
        menu = put_turkish_text(menu, "İşaret Dili Asistanı", 
                              (480, 100), font_size=72, 
                              color=COLORS['accent'])
        
        # Ana mod seçenekleri
        menu = put_turkish_text(menu, "1: Canlı Çeviri Modu", 
                              (480, 300), font_size=48, 
                              color=COLORS['text'])
        menu = put_turkish_text(menu, "2: Eğitim Modu", 
                              (480, 400), font_size=48, 
                              color=COLORS['text'])
        menu = put_turkish_text(menu, "Q: Çıkış", 
                              (480, 500), font_size=36, 
                              color=COLORS['text'])

        cv2.imshow("İşaret Dili Asistanı", menu)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            cv2.destroyWindow("İşaret Dili Asistanı")
            return "translate"
        elif key == ord('2'):
            cv2.destroyWindow("İşaret Dili Asistanı")
            return "training"
        elif key == ord('q'):
            cv2.destroyWindow("İşaret Dili Asistanı")
            return "quit"

def show_training_menu():
    """Eğitim menüsünü göster ve mod seçimini al"""
    menu_window = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.namedWindow("El İşareti Eğitimi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("El İşareti Eğitimi", 1280, 720)

    while True:
        menu = menu_window.copy()
        
        # Menü başlığı
        menu = put_turkish_text(menu, "El İşareti Eğitimi", 
                              (480, 100), font_size=72, 
                              color=COLORS['accent'])
        
        # Mod seçenekleri
        menu = put_turkish_text(menu, "1: Sıralı Eğitim Modu", 
                              (480, 250), font_size=36, 
                              color=COLORS['text'])
        menu = put_turkish_text(menu, "2: Rastgele Pratik Modu", 
                              (480, 350), font_size=36, 
                              color=COLORS['text'])
        menu = put_turkish_text(menu, "3: Kelime Pratik Modu", 
                              (480, 450), font_size=36, 
                              color=COLORS['text'])
        menu = put_turkish_text(menu, "4: Hız Testi Modu", 
                              (480, 550), font_size=36, 
                              color=COLORS['text'])
        menu = put_turkish_text(menu, "Q: Ana Menüye Dön", 
                              (480, 650), font_size=36, 
                              color=COLORS['text'])

        cv2.imshow("El İşareti Eğitimi", menu)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            cv2.destroyWindow("El İşareti Eğitimi")
            return "sequence"
        elif key == ord('2'):
            cv2.destroyWindow("El İşareti Eğitimi")
            return "practice"
        elif key == ord('3'):
            cv2.destroyWindow("El İşareti Eğitimi")
            return "word"
        elif key == ord('4'):
            cv2.destroyWindow("El İşareti Eğitimi")
            return "speed"
        elif key == ord('q'):
            cv2.destroyWindow("El İşareti Eğitimi")
            return "back"

def practice_mode():
    """Rastgele harflerle pratik modu"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_letter = random.choice(ALPHABET)
    success_threshold = 0.85
    success_duration = 1.5  # Pratik modunda daha kısa süre
    success_start_time = None
    success_hold_time = 0.0
    last_prediction = None
    show_success_message = False
    success_message_time = None
    score = 0
    total_attempts = 0
    
    cv2.namedWindow("El İşareti Pratik", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("El İşareti Pratik", 1280, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # UI elementleri
        frame = create_rounded_rectangle(frame, 20, 20, frame.shape[1]-20, 150, 15, COLORS['background'])
        
        # Mevcut harf
        frame = put_turkish_text(frame, f"Göster: {current_letter}", 
                               (40, 40), font_size=48, 
                               color=COLORS['accent'])
        
        # Skor
        score_text = f"Skor: {score}/{total_attempts}" if total_attempts > 0 else "Skor: 0/0"
        frame = put_turkish_text(frame, score_text, 
                               (40, 100), font_size=24, 
                               color=COLORS['text'])
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y])
            
            prediction = clf.predict([data])[0]
            last_prediction = prediction
            
            frame = put_turkish_text(frame, f"Algılanan: {prediction}", 
                                   (frame.shape[1] - 300, 100), font_size=24, 
                                   color=COLORS['text'])
            
            if prediction == current_letter:
                if success_start_time is None:
                    success_start_time = time.time()
                
                success_hold_time = time.time() - success_start_time
                
                if success_hold_time >= success_duration:
                    show_success_message = True
                    success_message_time = time.time()
                    success_start_time = None
                    score += 1
                    total_attempts += 1
                    current_letter = random.choice(ALPHABET)  # Yeni rastgele harf
            else:
                if success_start_time is not None:
                    total_attempts += 1
                success_start_time = None
                success_hold_time = 0.0

        elif success_start_time is not None:
            success_start_time = None
            success_hold_time = 0.0
        
        # İlerleme çubuğu
        if success_start_time is not None:
            progress_width = int((success_hold_time / success_duration) * (frame.shape[1] - 40))
            cv2.rectangle(frame, (20, frame.shape[0] - 40), 
                         (20 + progress_width, frame.shape[0] - 20), 
                         COLORS['success'], -1)
        
        # Başarı mesajı
        if show_success_message:
            frame = put_turkish_text(frame, "Doğru!", 
                                  (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                                  font_size=72, color=COLORS['success'])
            
            if time.time() - success_message_time >= 0.5:  # Daha kısa mesaj süresi
                show_success_message = False
        
        # Yardım metni
        help_text = "Q: Menüye Dön | R: Sıfırla"
        frame = put_turkish_text(frame, help_text, 
                               (frame.shape[1] - 300, frame.shape[0] - 40), 
                               font_size=20, color=COLORS['text'])
        
        cv2.imshow("El İşareti Pratik", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            score = 0
            total_attempts = 0
            current_letter = random.choice(ALPHABET)
            success_start_time = None
            success_hold_time = 0.0
            show_success_message = False
    
    cap.release()
    cv2.destroyAllWindows()

def training_mode():
    """Sıralı eğitim modu"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_letter_index = 0
    success_threshold = 0.85
    success_duration = 2.0
    success_start_time = None
    success_hold_time = 0.0
    last_prediction = None
    show_success_message = False
    success_message_time = None
    show_new_letter = False
    new_letter_time = None
    pause_detection = False  # Algılamayı duraklatmak için yeni değişken
    
    cv2.namedWindow("El İşareti Eğitimi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("El İşareti Eğitimi", 1280, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # UI elementleri
        # Üst bilgi paneli
        frame = create_rounded_rectangle(frame, 20, 20, frame.shape[1]-20, 150, 15, COLORS['background'])
        
        # Mevcut harf
        current_letter = ALPHABET[current_letter_index]
        frame = put_turkish_text(frame, f"Harf: {current_letter}", 
                               (40, 40), font_size=48, 
                               color=COLORS['accent'])
        
        # İlerleme
        progress = f"İlerleme: {current_letter_index + 1}/{len(ALPHABET)}"
        frame = put_turkish_text(frame, progress, 
                               (40, 100), font_size=24, 
                               color=COLORS['text'])
        
        # Referans el işareti fotoğrafını göster
        ref_image = load_reference_image(current_letter)
        if ref_image is not None:
            # Sağ üst köşede yarı saydam arka plan oluştur
            x_offset = frame.shape[1] - 320
            y_offset = 170
            
            # Yarı saydam arka plan
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_offset-10, y_offset-10), 
                         (x_offset + ref_image.shape[1]+10, y_offset + ref_image.shape[0]+10), 
                         COLORS['background'], -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Referans görüntüyü yerleştir
            frame[y_offset:y_offset + ref_image.shape[0], 
                  x_offset:x_offset + ref_image.shape[1]] = ref_image
            
            # Başlık ekle
            frame = put_turkish_text(frame, "Referans El İşareti:", 
                                   (x_offset-10, y_offset-40), font_size=20, 
                                   color=COLORS['text'])
        else:
            frame = put_turkish_text(frame, f"'{current_letter}' harfi için referans görüntü yok", 
                                   (frame.shape[1] - 500, 200), font_size=20, 
                                   color=COLORS['warning'])
        
        if not pause_detection and results.multi_hand_landmarks:  # Algılama duraklatılmamışsa devam et
            hand_landmarks = results.multi_hand_landmarks[0]
            
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y])
            
            prediction = clf.predict([data])[0]
            last_prediction = prediction
            
            frame = put_turkish_text(frame, f"Algılanan: {prediction}", 
                                   (frame.shape[1] - 300, 100), font_size=24, 
                                   color=COLORS['text'])
            
            if prediction == current_letter and not show_new_letter:
                if success_start_time is None:
                    success_start_time = time.time()
                
                success_hold_time = time.time() - success_start_time
                
                if success_hold_time >= success_duration:
                    show_success_message = True
                    success_message_time = time.time()
                    success_start_time = None
                    # Başarı sesi çal
                    if success_sound:
                        success_sound.play()
                    # Algılamayı duraklat
                    pause_detection = True
            else:
                success_start_time = None
                success_hold_time = 0.0

        elif success_start_time is not None:
            success_start_time = None
            success_hold_time = 0.0
        
        # Başarı mesajını kontrol et ve yeni harfe geç
        if show_success_message:
            frame = put_turkish_text(frame, "Harika!", 
                                  (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                                  font_size=72, color=COLORS['success'])
            
            if time.time() - success_message_time >= 1.0:  # 1 saniye bekle
                current_letter_index = (current_letter_index + 1) % len(ALPHABET)
                show_success_message = False
                success_start_time = None
                success_hold_time = 0.0
                show_new_letter = True
                new_letter_time = time.time()
                pause_detection = False  # Algılamayı tekrar başlat

        # Yeni harf duyurusunu göster
        if show_new_letter:
            new_letter = ALPHABET[current_letter_index]
            frame = put_turkish_text(frame, f"YENİ HARF: {new_letter}", 
                                  (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                                  font_size=84, color=COLORS['accent'])
            
            if time.time() - new_letter_time >= 1.0:
                show_new_letter = False
        
        # İlerleme çubuğu
        if success_start_time is not None:
            progress_width = int((success_hold_time / success_duration) * (frame.shape[1] - 40))
            cv2.rectangle(frame, (20, frame.shape[0] - 40), 
                         (20 + progress_width, frame.shape[0] - 20), 
                         COLORS['success'], -1)
        
        # Yardım metni
        help_text = "Q: Menüye Dön | R: Yeniden Başla"
        frame = put_turkish_text(frame, help_text, 
                               (frame.shape[1] - 300, frame.shape[0] - 40), 
                               font_size=20, color=COLORS['text'])
        
        cv2.imshow("El İşareti Eğitimi", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            current_letter_index = 0
            success_start_time = None
            success_hold_time = 0.0
            show_success_message = False
            show_new_letter = False
            pause_detection = False
    
    cap.release()
    cv2.destroyAllWindows()

def word_practice_mode():
    """Kelime pratik modu"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_word = random.choice(WORDS)
    current_letter_index = 0
    completed_letters = []  # Tamamlanan harflerin indeksleri
    success_threshold = 0.85
    success_duration = 1.5
    success_start_time = None
    success_hold_time = 0.0
    last_prediction = None
    show_success_message = False
    success_message_time = None
    score = 0
    total_words = 0
    
    cv2.namedWindow("Kelime Pratik", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Kelime Pratik", 1280, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # UI elementleri
        frame = create_rounded_rectangle(frame, 20, 20, frame.shape[1]-20, 150, 15, COLORS['background'])
        
        # Kelimeyi göster (harfleri renklendir)
        word_x = 40
        word_y = 40
        letter_spacing = 60
        
        for i, letter in enumerate(current_word):
            if i < current_letter_index:  # Tamamlanan harfler
                color = COLORS['success']
            elif i == current_letter_index:  # Aktif harf
                color = COLORS['accent']
            else:  # Henüz gelmeyen harfler
                color = COLORS['text']
            
            frame = put_turkish_text(frame, letter, 
                                   (word_x + i * letter_spacing, word_y), 
                                   font_size=48, color=color)
        
        # Skor
        score_text = f"Tamamlanan Kelime: {score}/{total_words}" if total_words > 0 else "Tamamlanan Kelime: 0/0"
        frame = put_turkish_text(frame, score_text, 
                               (40, 100), font_size=24, 
                               color=COLORS['text'])
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y])
            
            prediction = clf.predict([data])[0]
            last_prediction = prediction
            
            frame = put_turkish_text(frame, f"Algılanan: {prediction}", 
                                   (frame.shape[1] - 300, 100), font_size=24, 
                                   color=COLORS['text'])
            
            if prediction == current_word[current_letter_index]:
                if success_start_time is None:
                    success_start_time = time.time()
                
                success_hold_time = time.time() - success_start_time
                
                if success_hold_time >= success_duration:
                    show_success_message = True
                    success_message_time = time.time()
                    success_start_time = None
                    current_letter_index += 1
                    
                    # Harf tamamlandığında ses çal
                    if success_sound:
                        success_sound.play()
                    
                    # Kelime tamamlandı mı kontrol et
                    if current_letter_index >= len(current_word):
                        score += 1
                        total_words += 1
                        # Kelime tamamlandığında farklı bir ses çal
                        if word_complete_sound:
                            word_complete_sound.play()
                        current_word = random.choice(WORDS)
                        current_letter_index = 0
                        show_success_message = True
                        success_message_time = time.time()
            else:
                success_start_time = None
                success_hold_time = 0.0

        elif success_start_time is not None:
            success_start_time = None
            success_hold_time = 0.0
        
        # İlerleme çubuğu
        if success_start_time is not None:
            progress_width = int((success_hold_time / success_duration) * (frame.shape[1] - 40))
            cv2.rectangle(frame, (20, frame.shape[0] - 40), 
                         (20 + progress_width, frame.shape[0] - 20), 
                         COLORS['success'], -1)
        
        # Başarı mesajı
        if show_success_message:
            if current_letter_index == 0:  # Yeni kelimeye geçildi
                frame = put_turkish_text(frame, "Kelime Tamamlandı!", 
                                      (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                                      font_size=72, color=COLORS['success'])
            else:  # Harf tamamlandı
                frame = put_turkish_text(frame, "Doğru!", 
                                      (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                                      font_size=72, color=COLORS['success'])
            
            if time.time() - success_message_time >= 0.5:
                show_success_message = False
        
        # Yardım metni
        help_text = "Q: Menüye Dön | R: Sıfırla"
        frame = put_turkish_text(frame, help_text, 
                               (frame.shape[1] - 300, frame.shape[0] - 40), 
                               font_size=20, color=COLORS['text'])
        
        cv2.imshow("Kelime Pratik", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            score = 0
            total_words = 0
            current_word = random.choice(WORDS)
            current_letter_index = 0
            success_start_time = None
            success_hold_time = 0.0
            show_success_message = False
    
    cap.release()
    cv2.destroyAllWindows()

def speed_test_mode():
    """1 dakikalık hız testi modu"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_word = random.choice(WORDS)
    current_letter_index = 0
    success_threshold = 0.85
    success_duration = 1.0  # Hız testi için daha kısa süre
    success_start_time = None
    success_hold_time = 0.0
    last_prediction = None
    show_success_message = False
    success_message_time = None
    completed_words = 0
    test_start_time = time.time()
    test_duration = 60  # 60 saniye
    game_over = False
    
    cv2.namedWindow("Hız Testi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hız Testi", 1280, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # UI elementleri
        frame = create_rounded_rectangle(frame, 20, 20, frame.shape[1]-20, 150, 15, COLORS['background'])
        
        # Kalan süreyi hesapla
        elapsed_time = time.time() - test_start_time
        remaining_time = max(0, test_duration - elapsed_time)
        
        if remaining_time == 0 and not game_over:
            game_over = True
            if word_complete_sound:
                word_complete_sound.play()
        
        # Kelimeyi göster (harfleri renklendir)
        if not game_over:
            word_x = 40
            word_y = 40
            letter_spacing = 60
            
            for i, letter in enumerate(current_word):
                if i < current_letter_index:
                    color = COLORS['success']
                elif i == current_letter_index:
                    color = COLORS['accent']
                else:
                    color = COLORS['text']
                
                frame = put_turkish_text(frame, letter, 
                                       (word_x + i * letter_spacing, word_y), 
                                       font_size=48, color=color)
            
            # Süre ve skor
            time_text = f"Kalan Süre: {int(remaining_time)} sn"
            frame = put_turkish_text(frame, time_text, 
                                   (frame.shape[1] - 300, 40), font_size=36, 
                                   color=COLORS['accent'])
            
            score_text = f"Tamamlanan Kelime: {completed_words}"
            frame = put_turkish_text(frame, score_text, 
                                   (40, 100), font_size=24, 
                                   color=COLORS['text'])
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y])
                
                prediction = clf.predict([data])[0]
                last_prediction = prediction
                
                frame = put_turkish_text(frame, f"Algılanan: {prediction}", 
                                       (frame.shape[1] - 300, 100), font_size=24, 
                                       color=COLORS['text'])
                
                if prediction == current_word[current_letter_index]:
                    if success_start_time is None:
                        success_start_time = time.time()
                    
                    success_hold_time = time.time() - success_start_time
                    
                    if success_hold_time >= success_duration:
                        show_success_message = True
                        success_message_time = time.time()
                        success_start_time = None
                        current_letter_index += 1
                        
                        # Harf tamamlandığında ses çal
                        if success_sound:
                            success_sound.play()
                        
                        # Kelime tamamlandı mı kontrol et
                        if current_letter_index >= len(current_word):
                            completed_words += 1
                            current_word = random.choice(WORDS)
                            current_letter_index = 0
                            show_success_message = True
                            success_message_time = time.time()
                else:
                    success_start_time = None
                    success_hold_time = 0.0

            elif success_start_time is not None:
                success_start_time = None
                success_hold_time = 0.0
            
            # İlerleme çubuğu
            if success_start_time is not None:
                progress_width = int((success_hold_time / success_duration) * (frame.shape[1] - 40))
                cv2.rectangle(frame, (20, frame.shape[0] - 40), 
                             (20 + progress_width, frame.shape[0] - 20), 
                             COLORS['success'], -1)
        else:
            # Oyun sonu ekranı
            frame = put_turkish_text(frame, "SÜRE DOLDU!", 
                                   (frame.shape[1]//2 - 150, frame.shape[0]//2 - 100), 
                                   font_size=72, color=COLORS['warning'])
            
            result_text = f"Toplam Tamamlanan Kelime: {completed_words}"
            frame = put_turkish_text(frame, result_text, 
                                   (frame.shape[1]//2 - 250, frame.shape[0]//2), 
                                   font_size=48, color=COLORS['success'])
            
            wpm_text = f"Dakikada Kelime Sayısı: {completed_words}"
            frame = put_turkish_text(frame, wpm_text, 
                                   (frame.shape[1]//2 - 250, frame.shape[0]//2 + 80), 
                                   font_size=48, color=COLORS['accent'])
        
        # Yardım metni
        if game_over:
            help_text = "Q: Menüye Dön | R: Tekrar Dene"
        else:
            help_text = "Q: Menüye Dön"
        frame = put_turkish_text(frame, help_text, 
                               (frame.shape[1] - 300, frame.shape[0] - 40), 
                               font_size=20, color=COLORS['text'])
        
        cv2.imshow("Hız Testi", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and game_over:
            # Testi yeniden başlat
            current_word = random.choice(WORDS)
            current_letter_index = 0
            completed_words = 0
            test_start_time = time.time()
            game_over = False
            success_start_time = None
            success_hold_time = 0.0
            show_success_message = False
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        # Ana menüyü göster
        main_choice = show_main_menu()
        
        if main_choice == "quit":
            break
        elif main_choice == "translate":
            # Canlı çeviri modunu başlat
            try:
                subprocess.run(["python3", "hand_sign_predict.py"])
            except Exception as e:
                print(f"Hata: Canlı çeviri modu başlatılamadı - {e}")
        elif main_choice == "training":
            # Eğitim menüsünü göster
            while True:
                training_choice = show_training_menu()
                
                if training_choice == "back":
                    break
                elif training_choice == "sequence":
                    training_mode()
                elif training_choice == "practice":
                    practice_mode()
                elif training_choice == "word":
                    word_practice_mode()
                elif training_choice == "speed":
                    speed_test_mode()

if __name__ == "__main__":
    main() 