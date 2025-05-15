import cv2
import numpy as np
import subprocess
import sys
import os

class MenuButton:
    def __init__(self, text, x, y, width, height, action):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.action = action
        self.is_hovered = False

    def draw(self, frame):
        bg_color = (76, 175, 80) if self.is_hovered else (52, 152, 219)  # Yeşil/Mavi
        
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height),
                     bg_color, -1)
        
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height),
                     (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        text_size = cv2.getTextSize(self.text, font, font_scale, 2)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        
        cv2.putText(frame, self.text, (text_x, text_y), font, font_scale, 
                    (255, 255, 255), 2)

    def check_hover(self, mouse_pos):
        x, y = mouse_pos
        self.is_hovered = (self.x <= x <= self.x + self.width and 
                          self.y <= y <= self.y + self.height)
        return self.is_hovered

def create_rounded_rectangle(img, x1, y1, x2, y2, radius, color):
    """Yumuşak köşeli dikdörtgen çiz"""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    return img

def main():
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    WINDOW_NAME = "İşaret Dili Çeviri Menüsü"
    
    mouse_x, mouse_y = 0, 0
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x, mouse_y = x, y
        elif event == cv2.EVENT_LBUTTONDOWN:
            for button in buttons:
                if button.check_hover((x, y)):
                    button.action()
    
    def start_translation():
        subprocess.Popen([sys.executable, "hand_sign_predict.py"])
    
    def show_help():
        help_text = """
        Kullanım Kılavuzu:
        
        1. Çeviri Başlat:
           - Program başladığında kamera açılacak
           - El işaretleriniz otomatik algılanacak
           
        2. Kontrol Modları:
           - Göz Kontrolü (G tuşu)
           - Kafa Kontrolü (K tuşu)
           
        3. Temel Komutlar:
           - Temizle: C tuşu
           - Çıkış: Q tuşu
           - Sesli Okuma: E tuşu
           
        4. İşaretlerin Algılanması:
           - Elinizi kameraya gösterin
           - Sabit tutun
           - Sonucu ekranda görün
        """
        print(help_text)
    
    def exit_program():
        cv2.destroyAllWindows()
        sys.exit()
    
    button_width = 300
    button_height = 60
    button_x = (WINDOW_WIDTH - button_width) // 2
    
    buttons = [
        MenuButton("Çeviri Başlat", button_x, 250, button_width, button_height, start_translation),
        MenuButton("Yardım", button_x, 350, button_width, button_height, show_help),
        MenuButton("Çıkış", button_x, 450, button_width, button_height, exit_program)
    ]
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    frame[:] = (18, 18, 18)  
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1) 
    
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    
    while True:
        frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        frame[:] = (18, 18, 18) 
        
        title_height = 120
        create_rounded_rectangle(frame, 0, 0, WINDOW_WIDTH, title_height, 15, (52, 152, 219))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "İşaret Dili Çeviri Sistemi"
        title_size = cv2.getTextSize(title, font, 1.5, 2)[0]
        title_x = (WINDOW_WIDTH - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 70), font, 1.5, (255, 255, 255), 2)
        
        subtitle = "Engelsiz İletişim İçin"
        subtitle_size = cv2.getTextSize(subtitle, font, 0.8, 2)[0]
        subtitle_x = (WINDOW_WIDTH - subtitle_size[0]) // 2
        cv2.putText(frame, subtitle, (subtitle_x, title_height - 20), font, 0.8, (255, 255, 255), 2)
        
        for button in buttons:
            button.check_hover((mouse_x, mouse_y))
            button.draw(frame)
        
        # Alt bilgi
        footer_text = "© 2024 İşaret Dili Çeviri Sistemi"
        footer_size = cv2.getTextSize(footer_text, font, 0.6, 2)[0]
        footer_x = (WINDOW_WIDTH - footer_size[0]) // 2
        cv2.putText(frame, footer_text, (footer_x, WINDOW_HEIGHT - 30), font, 0.6, (128, 128, 128), 2)
        
        cv2.imshow(WINDOW_NAME, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 