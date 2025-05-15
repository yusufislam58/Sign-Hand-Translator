# Bilet Kontrol Programı

Bu program, belirtilen web sitesinde bilet durumunu sürekli olarak kontrol eden bir Python scriptidir.

## Özellikler

- Belirli aralıklarla otomatik kontrol
- Masaüstü bildirimleri
- Hata yönetimi
- Kolay özelleştirilebilir yapı

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. `ticket_checker.py` dosyasını açın ve aşağıdaki değişkenleri düzenleyin:
   - `url`: Kontrol edilecek web sitesinin adresi
   - `check_interval`: Kontrol aralığı (saniye cinsinden)
   - `check_tickets` fonksiyonunda web sitesine özel selector'ları ayarlayın

## Kullanım

Programı başlatmak için:
```bash
python ticket_checker.py
```

## Özelleştirme

Web sitesine özel kontroller için `check_tickets` fonksiyonunu düzenlemeniz gerekmektedir. Mevcut kod örnek olarak hazırlanmıştır ve hedef web sitesine göre özelleştirilmelidir.

## Notlar

- Program çalışırken terminal açık kalmalıdır
- Çıkmak için Ctrl+C tuşlarını kullanın
- Web sitesinin robot kontrollerine takılmamak için uygun aralıklarla kontrol yapılmalıdır 