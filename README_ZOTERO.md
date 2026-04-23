
## Zotero Entegrasyonu (Herkes İçin)

ECE v3 artık doğrudan her kullanıcının kendi Zotero kütüphanesine bağlanabiliyor. Hiçbir API anahtarı repoda saklanmıyor.

### 1. Kurulum
```bash
pip install -r requirements.txt
cp .env.example .env
```

### 2. Zotero anahtarını ekle
En kolay yol:
```bash
python setup_zotero.py
```
Sizden şunları isteyecek:
- ZOTERO_LIBRARY_ID: zotero.org/settings/keys sayfasında "Your userID"
- ZOTERO_API_KEY: aynı sayfada oluşturacağınız personal key

Alternatif (manuel):
.env dosyanıza:
```
ZOTERO_LIBRARY_ID=1234567
ZOTERO_API_KEY=abc123...
ZOTERO_LIBRARY_TYPE=user
```

### 3. Test et
```bash
python zotero_connector.py list-collections
```

### 4. Zotero'dan içeri al
```bash
# Önizleme
python zotero_connector.py preview --tag felsefe --limit 5

# Neo4j'ye aktar
python zotero_connector.py import --topic "Heidegger" --tag felsefe --default-class critical --epoch-start 1920 --epoch-end 1976
```

Sonra normal ECE akışını çalıştır:
```bash
python ECE_v3_ALL_IN_ONE.py start --topic "Heidegger" --year 1927 --thread-id heidegger-1927
```

**Gizlilik notu:** `zotero_connector.py` kimlik bilgilerini sadece ortam değişkenlerinden okur, hiçbir zaman diske veya loga yazmaz. `ZOTERO_LOCAL=true` yaparsanız API anahtarı bile gerekmez - çalışan Zotero 7+ uygulamanızdan okur.
