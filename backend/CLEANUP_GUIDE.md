# OHLCV Data Cleanup Guide

Bu rehber, anomali içeren veya hatalı OHLCV verilerini veritabanından silmek için kullanılır.

## Kullanım Senaryoları

### Senaryo 1: Piyasa Crash Sonrası Temizlik
Dün gece beklenmedik bir haber veya olay nedeniyle piyasada ani bir düşüş oldu ve bu anormal veriyi AI modelinden uzak tutmak istiyorsun.

### Senaryo 2: Veri Toplama Hatası
Veri toplama sırasında bir hata oluştu ve hatalı veriler DB'ye kaydedildi.

### Senaryo 3: Test Verilerini Silme
Test amaçlı toplanan verileri temizlemek istiyorsun.

---

## Komutlar

### 1️⃣ Önce Önizle (Dry-Run) - HER ZAMAN BUNUNLA BAŞLA

```bash
# Son 4 saatlik veriyi önizle (SİLMEZ, sadece gösterir)
python backend/cleanup_recent_data.py --hours 4
```

**Örnek Çıktı:**
```
============================================================
🔍 PREVIEW: Records to be deleted (Last 4 hours)
============================================================
  📊 BTCUSDT      1h   :    4 records
  📊 ETHUSDT      1h   :    4 records
  📊 BNBUSDT      1h   :    4 records
------------------------------------------------------------
  📈 TOTAL: 12 records will be deleted
============================================================

⚠️  DRY RUN MODE - No data will be deleted
   Add --execute flag to actually delete the data
```

---

### 2️⃣ Gerçekten Sil (Execute Mode)

**UYARI:** Bu işlem GERİ ALINMAZ! Önce mutlaka dry-run ile kontrol et.

```bash
# Son 4 saatlik veriyi SİL
python backend/cleanup_recent_data.py --hours 4 --execute
```

**Örnek İnteraktif Onay:**
```
⚠️  WARNING: This operation is IRREVERSIBLE!
   All data from the specified time range will be permanently deleted.

❓ Do you want to continue? (yes/no): yes

============================================================
✅ DELETION COMPLETE
============================================================
  🗑️  BTCUSDT      1h   :    4 records deleted
  🗑️  ETHUSDT      1h   :    4 records deleted
  🗑️  BNBUSDT      1h   :    4 records deleted
------------------------------------------------------------
  📈 TOTAL: 12 records deleted
============================================================
```

---

## Kullanım Örnekleri

### 📅 Örnek 1: Son 3 Saat (En Yaygın)

```bash
# 1. Önce önizle
python backend/cleanup_recent_data.py --hours 3

# 2. Emin olduysan sil
python backend/cleanup_recent_data.py --hours 3 --execute
```

---

### 📅 Örnek 2: Belirli Bir Zamandan Sonraki Tüm Veriler

Diyelim ki **11 Ekim 2025, 15:00**'den sonra crash başladı ve o andan sonraki tüm veriyi silmek istiyorsun:

```bash
# 1. Önce önizle
python backend/cleanup_recent_data.py --after "2025-10-11 15:00:00"

# 2. Emin olduysan sil
python backend/cleanup_recent_data.py --after "2025-10-11 15:00:00" --execute
```

---

### 📅 Örnek 3: Belirli Bir Zaman Aralığı

**14:00 - 18:00** arasındaki verileri silmek istiyorsun:

```bash
# 1. Önce önizle
python backend/cleanup_recent_data.py --start "2025-10-11 14:00:00" --end "2025-10-11 18:00:00"

# 2. Emin olduysan sil
python backend/cleanup_recent_data.py --start "2025-10-11 14:00:00" --end "2025-10-11 18:00:00" --execute
```

---

### 📅 Örnek 4: Sadece Son 1 Saat (Hafif Düzeltme)

```bash
# Son 1 saatlik veriyi sil
python backend/cleanup_recent_data.py --hours 1 --execute
```

---

## Güvenlik Özellikleri

### 1. Dry-Run Mode (Default)
`--execute` bayrağı olmadan çalıştırırsanız, **hiçbir şey silinmez**, sadece ne silineceği gösterilir.

```bash
# Bu komut hiçbir şey SİLMEZ, sadece GÖSTERÜR
python backend/cleanup_recent_data.py --hours 4
```

### 2. İnteraktif Onay
`--execute` ile bile çalıştırsanız, script önce onay sorar:

```
❓ Do you want to continue? (yes/no):
```

Sadece `yes` veya `y` yazdığınızda silme işlemi başlar.

### 3. Detaylı Önizleme
Silmeden önce her symbol/interval için kaç kayıt silineceği gösterilir.

---

## Zaman Formatı

Tüm tarih/saat girdileri şu formatta olmalı:

```
YYYY-MM-DD HH:MM:SS
```

**Örnekler:**
- `2025-10-11 15:30:00` ✅
- `2025-10-11 15:30` ❌ (saniye eksik)
- `11-10-2025 15:30:00` ❌ (yanlış format)

---

## Sık Sorulan Sorular (FAQ)

### ❓ Silinen verileri geri getirebilir miyim?
**Hayır.** Bu işlem kalıcıdır ve geri alınamaz. Bu yüzden önce **mutlaka dry-run ile önizleme** yapın.

### ❓ Feature'ları da silmeli miyim?
**Hayır.** Bu script sadece `ohlcv` tablosunu temizler. Feature'lar henüz hesaplanmadıysa (senin durumunda öyle), herhangi bir şey yapmana gerek yok.

### ❓ DB'de ne kadar veri var kontrol edebilir miyim?
Evet:

```bash
# PostgreSQL'e bağlan
psql -U postgres -d crypto_db

# Toplam kayıt sayısını gör
SELECT COUNT(*) FROM ohlcv;

# Symbol'e göre kayıt sayısı
SELECT symbol, COUNT(*) FROM ohlcv GROUP BY symbol;

# Son 4 saatlik kayıt sayısı
SELECT COUNT(*) FROM ohlcv WHERE open_time >= NOW() - INTERVAL '4 hours';
```

### ❓ Script çalışırken hata verirse ne olur?
Script transaction kullanıyor. Hata durumunda hiçbir şey silinmez veya kısmi silme olmaz.

### ❓ Main.py çalışırken cleanup yapabilir miyim?
**Evet**, ama önerilmez. Önce `main.py`'yi durdur, sonra cleanup yap, sonra yeniden başlat.

```bash
# 1. main.py'yi durdur (Ctrl+C)
# 2. Cleanup yap
python backend/cleanup_recent_data.py --hours 4 --execute
# 3. main.py'yi yeniden başlat
python backend/main.py
```

---

## Komut Satırı Yardımı

Tüm seçenekleri görmek için:

```bash
python backend/cleanup_recent_data.py --help
```

**Çıktı:**
```
usage: cleanup_recent_data.py [-h] (--hours HOURS | --after AFTER)
                               [--start START] [--end END] [--execute]

Cleanup recent OHLCV data from database

optional arguments:
  -h, --help       show this help message and exit
  --hours HOURS    Delete data from last N hours
  --after AFTER    Delete data after timestamp (format: "YYYY-MM-DD HH:MM:SS")
  --start START    Start of time range (use with --end)
  --end END        End of time range (use with --start)
  --execute        Actually delete data (without this flag, only preview)

Examples:
  # Preview deletion of last 4 hours (dry-run)
  python cleanup_recent_data.py --hours 4

  # Actually delete last 4 hours
  python cleanup_recent_data.py --hours 4 --execute

  # Delete data after specific timestamp
  python cleanup_recent_data.py --after "2025-10-11 15:00:00" --execute

  # Delete data in a time range
  python cleanup_recent_data.py --start "2025-10-11 14:00:00"
                                --end "2025-10-11 18:00:00" --execute
```

---

## İleriye Dönük: Otomatik Anomaly Detection

Gelecekte, anomali tespiti için otomatik bir sistem eklenebilir:

```python
# Planlanan feature (henüz yok)
python backend/detect_anomalies.py --hours 24
# Bu komut son 24 saatteki anomalileri tespit edip işaretler
# Manuel olarak onayladıktan sonra otomatik silebilir
```

---

## Yedekleme (Opsiyonel)

Silmeden önce yedek almak istersen:

```bash
# PostgreSQL dump (sadece ohlcv tablosu)
pg_dump -U postgres -d crypto_db -t ohlcv > backup_ohlcv_$(date +%Y%m%d_%H%M%S).sql

# Geri yükleme (gerekirse)
psql -U postgres -d crypto_db < backup_ohlcv_20251011_150000.sql
```

---

## Özet: Hızlı Başlangıç

Piyasa crash'i sonrası son 4 saatlik veriyi silmek için:

```bash
# 1. Neyin silineceğine bak (risk yok)
python backend/cleanup_recent_data.py --hours 4

# 2. Her şey tamam görünüyorsa, gerçekten sil
python backend/cleanup_recent_data.py --hours 4 --execute

# 3. "yes" yazıp onayla
```

**Bu kadar!** 🎉

---

## İletişim

Sorularınız için:
- GitHub Issues: `crypto-ai-agent` repo
- Dokümantasyon: `/docs`

**Önemli:** Her zaman önce **dry-run** yap! 🔒
