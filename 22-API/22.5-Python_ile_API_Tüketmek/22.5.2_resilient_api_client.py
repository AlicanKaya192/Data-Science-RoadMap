################################################
# DAYANIKLI (RESILIENT) BİR API İSTEMCİSİ İNŞA ETMEK
################################################

# İş Problemi: 22.5.1-Python_ile_API_Tüketmek.pdf dosyasında anlattığımız, production kalitesindeki
# bir API istemcisinin sahip olması gereken 5 özelliği (session kullanımı, akıllı tekrar deneme,
# hata ayrımı, önbellekleme, otomatik sayfalama) uçtan uca, çalışan bir Python sınıfına dökmek.
#
# Bu script'i TAMAMEN OFFLINE ve HERKESTE AYNI ŞEKİLDE ÇALIŞACAK hale getirmek için, dış bir genel
# API'ye bağlanmak yerine, kendi basit "Kitap Kataloğu" API'mizi yerel bir portta (127.0.0.1) arka
# plan iş parçacığında (thread) ayağa kaldırıyoruz ve istemcimizi buna karşı test ediyoruz. Böylece
# internet bağlantısı olmayan ya da üçüncü taraf bir servisin o an kesintili olduğu bir ortamda bile
# script sorunsuz ve tekrar üretilebilir (reproducible) şekilde çalışır.

# Gerekli kütüphaneyi kurmak için: pip install requests

import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

##################################################
# 1. SAHTE (MOCK) BİR "KİTAP KATALOĞU" API'Sİ KURMAK
##################################################

# 47 kitaplık sahte bir katalog - gerçekçi bir sayfalama senaryosu için.
TUM_KITAPLAR = [
    {"id": i, "baslik": f"Kitap #{i}", "yazar": f"Yazar {chr(65 + i % 26)}"}
    for i in range(1, 48)
]
SAYFA_BOYUTU = 10

# İstemcimizin "hata toleransını" test etmek için: bu endpoint ilk 2 çağrıda bilerek 503 döner,
# 3. çağrıda başarılı olur (geçici bir sunucu arızasını simüle eder).
kararsiz_endpoint_sayaci = {"deneme": 0}

# Bu endpoint her 3 istekten birinde 429 (Too Many Requests) döner - hız sınırlama simülasyonu.
hiz_siniri_sayaci = {"istek": 0}


class KitapKatalogHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Konsolu kirletmemek için sunucunun kendi loglarını susturuyoruz.

    def _json_yanit(self, status_code, payload, extra_headers=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/kitaplar":
            sayfa = int(qs.get("sayfa", ["1"])[0])
            baslangic = (sayfa - 1) * SAYFA_BOYUTU
            bitis = baslangic + SAYFA_BOYUTU
            sayfa_verisi = TUM_KITAPLAR[baslangic:bitis]
            self._json_yanit(200, {
                "kitaplar": sayfa_verisi,
                "sayfa": sayfa,
                "sonraki_sayfa_var_mi": bitis < len(TUM_KITAPLAR),
            })

        elif parsed.path.startswith("/kitaplar/"):
            kitap_id = int(parsed.path.split("/")[-1])
            eslesen = next((k for k in TUM_KITAPLAR if k["id"] == kitap_id), None)
            if eslesen:
                self._json_yanit(200, eslesen)
            else:
                self._json_yanit(404, {"hata": "Kitap bulunamadı", "kitap_id": kitap_id})

        elif parsed.path == "/kararsiz":
            kararsiz_endpoint_sayaci["deneme"] += 1
            if kararsiz_endpoint_sayaci["deneme"] < 3:
                self._json_yanit(503, {"hata": "Sunucu geçici olarak meşgul, tekrar deneyin"})
            else:
                self._json_yanit(200, {"mesaj": "Başarılı!", "kacinci_denemede": kararsiz_endpoint_sayaci["deneme"]})

        elif parsed.path == "/hiz-siniri":
            hiz_siniri_sayaci["istek"] += 1
            if hiz_siniri_sayaci["istek"] % 3 == 0:
                self._json_yanit(429, {"hata": "Çok fazla istek"}, extra_headers={"Retry-After": "1"})
            else:
                self._json_yanit(200, {"mesaj": "OK", "istek_no": hiz_siniri_sayaci["istek"]})

        else:
            self._json_yanit(404, {"hata": "Endpoint bulunamadı"})


def sunucuyu_baslat():
    server = ThreadingHTTPServer(("127.0.0.1", 0), KitapKatalogHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


sunucu, PORT = sunucuyu_baslat()
BASE_URL = f"http://127.0.0.1:{PORT}"
print(f"Sahte Kitap Kataloğu API'si başlatıldı: {BASE_URL}")

##################################################
# 2. DAYANIKLI (RESILIENT) API İSTEMCİSİ SINIFI
##################################################


class ResilientAPIClient:
    """22.5.1-Python_ile_API_Tüketmek.pdf, Bölüm 3'te anlatılan 5 prensibi uygulayan API istemcisi:
    (1) Session ile bağlantı yeniden kullanımı, (2) otomatik + akıllı tekrar deneme
    (sadece 429/5xx'te, exponential backoff ile), (3) 4xx hatalarında tekrar denememe,
    (4) basit TTL önbellek, (5) otomatik sayfalama (generator)."""

    def __init__(self, base_url, max_deneme=4, cache_ttl_saniye=30, zaman_asimi=5):
        self.base_url = base_url.rstrip("/")
        self.zaman_asimi = zaman_asimi
        self.cache_ttl = cache_ttl_saniye
        self._cache = {}  # {url: (deger, kaydedilme_zamani)}

        self.session = requests.Session()

        # urllib3'ün Retry mekanizması: SADECE 429 ve 5xx durum kodlarında, üstel geri çekilmeyle
        # (backoff_factor * (2 ** (deneme_no - 1))) otomatik tekrar dener. 4xx (400, 401, 404 gibi)
        # kalıcı hatalarda HİÇ tekrar denemez (bkz. 22.5.1, Bölüm 3.3).
        retry_stratejisi = Retry(
            total=max_deneme,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_stratejisi)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _cache_oku(self, url):
        if url in self._cache:
            deger, kayit_zamani = self._cache[url]
            if time.time() - kayit_zamani < self.cache_ttl:
                return deger
            del self._cache[url]  # süresi dolmuş kaydı temizle
        return None

    def get(self, path, params=None, cache_kullan=True):
        """Tek bir GET isteği atar. HTTPAdapter + Retry sayesinde 429/5xx hatalarında OTOMATİK
        olarak tekrar dener; 4xx hatalarında (örn. 404) HİÇ tekrar denemeden hemen sonucu döner."""
        url = f"{self.base_url}{path}"
        cache_anahtari = f"{url}?{params}"

        if cache_kullan:
            onbellek_sonucu = self._cache_oku(cache_anahtari)
            if onbellek_sonucu is not None:
                return onbellek_sonucu

        yanit = self.session.get(url, params=params, timeout=self.zaman_asimi)

        if cache_kullan and yanit.status_code == 200:
            self._cache[cache_anahtari] = (yanit, time.time())

        return yanit

    def tum_kitaplari_getir(self):
        """Sayfalama detaylarını (bkz. 22.3.1, Bölüm 4) çağıran koddan tamamen gizleyen bir
        generator. Kullanan kod sadece 'for kitap in client.tum_kitaplari_getir():' yazar."""
        sayfa = 1
        while True:
            yanit = self.get("/kitaplar", params={"sayfa": sayfa}, cache_kullan=False)
            yanit.raise_for_status()
            veri = yanit.json()
            for kitap in veri["kitaplar"]:
                yield kitap
            if not veri["sonraki_sayfa_var_mi"]:
                break
            sayfa += 1


##################################################
# 3. MANUEL RETRY DÖNGÜSÜ (Mekanizmayı Şeffaf Göstermek İçin)
##################################################

# HTTPAdapter + Retry "sihirli" görünebilir; ne yaptığını somut görmek için AYNI mantığı elle de
# yazıyoruz. Gerçek projelerde genellikle yukarıdaki otomatik yaklaşım tercih edilir.

def manuel_retry_ile_istek(url, max_deneme=4, taban_bekleme=0.3):
    """Üstel geri çekilmeyi (exponential backoff) elle uygulayan bir GET isteği (bkz. 22.5.1,
    Bölüm 3.2). Sadece 429/5xx'te tekrar dener; 4xx'te hemen döner."""
    for deneme in range(1, max_deneme + 1):
        yanit = requests.get(url, timeout=5)

        if yanit.status_code < 400:
            return yanit

        if yanit.status_code < 500 and yanit.status_code != 429:
            # Kalıcı istemci hatası (örn. 404) - tekrar denemenin anlamı yok.
            print(f"  [Deneme {deneme}] Kalıcı hata ({yanit.status_code}) - tekrar denenmiyor.")
            return yanit

        bekleme_suresi = taban_bekleme * (2 ** (deneme - 1))
        # Sunucu Retry-After başlığı verdiyse, ona öncelik ver.
        if "Retry-After" in yanit.headers:
            bekleme_suresi = float(yanit.headers["Retry-After"])
        print(f"  [Deneme {deneme}] Geçici hata ({yanit.status_code}) - {bekleme_suresi:.1f}sn bekleyip tekrar denenecek.")
        time.sleep(bekleme_suresi)

    return yanit  # Tüm denemeler tükendi, son yanıtı döndür.


##################################################
# 4. İSTEMCİYİ TEST ETMEK
##################################################

print("\n=== TEST 1: Basit GET isteği ===")
client = ResilientAPIClient(BASE_URL)
yanit = client.get("/kitaplar/5")
assert yanit.status_code == 200, "Beklenmeyen durum kodu!"
kitap = yanit.json()
print(f"5 numaralı kitap: {kitap}")
assert kitap["id"] == 5

print("\n=== TEST 2: 404 - kalıcı hata, tekrar deneme YAPILMAMALI ===")
baslangic = time.time()
yanit = client.get("/kitaplar/9999")
sure = time.time() - baslangic
assert yanit.status_code == 404
print(f"404 yanıtı {sure:.3f} saniyede geldi (tekrar deneme YOK, çok hızlı olmalı) -> {yanit.json()}")
assert sure < 1.0, "404'te tekrar deneme yapılmamalı, cok uzun surdu!"

print("\n=== TEST 3: Kararsız endpoint - HTTPAdapter otomatik tekrar deniyor ===")
kararsiz_endpoint_sayaci["deneme"] = 0  # sayaci sifirla
baslangic = time.time()
yanit = client.get("/kararsiz", cache_kullan=False)
sure = time.time() - baslangic
assert yanit.status_code == 200, "Otomatik retry basarisiz oldu!"
print(f"Kararsız endpoint {sure:.2f} saniye içinde, otomatik tekrar denemelerle başarılı oldu: {yanit.json()}")

print("\n=== TEST 4: Manuel retry döngüsü ile aynı senaryo (mekanizmayı gözlemlemek için) ===")
kararsiz_endpoint_sayaci["deneme"] = 0
yanit = manuel_retry_ile_istek(f"{BASE_URL}/kararsiz")
assert yanit.status_code == 200
print(f"Manuel retry sonucu: {yanit.json()}")

print("\n=== TEST 5: Önbellekleme (Cache) - ikinci çağrı ağa gitmemeli ===")
client.get("/kitaplar/1")  # ilk çağrı - önbelleğe yazılır
baslangic = time.time()
yanit_cache = client.get("/kitaplar/1")  # ikinci çağrı - önbellekten dönmeli
sure_cache = time.time() - baslangic
print(f"Önbellekten okuma süresi: {sure_cache*1000:.2f} ms (gerçek bir ağ isteğinden çok daha hızlı olmalı)")
assert yanit_cache.json()["id"] == 1

print("\n=== TEST 6: Otomatik sayfalama (generator) - TÜM 47 kitabı tek döngüyle çekmek ===")
tum_kitaplar_listesi = list(client.tum_kitaplari_getir())
print(f"Toplam çekilen kitap sayısı: {len(tum_kitaplar_listesi)}")
assert len(tum_kitaplar_listesi) == 47, "Tum kitaplar cekilemedi!"
print(f"İlk kitap: {tum_kitaplar_listesi[0]['baslik']}, Son kitap: {tum_kitaplar_listesi[-1]['baslik']}")

print("\n=== TEST 7: Hız sınırı (429) - Retry-After'a saygı gösterilerek otomatik bekleniyor ===")
hiz_siniri_sayaci["istek"] = 0
basarili_istek_sayisi = 0
baslangic = time.time()
for _ in range(5):
    yanit = client.get("/hiz-siniri", cache_kullan=False)
    if yanit.status_code == 200:
        basarili_istek_sayisi += 1
sure = time.time() - baslangic
print(f"5 isteğin {basarili_istek_sayisi} tanesi başarılı oldu (429'lar otomatik retry ile aşıldı), toplam süre: {sure:.2f}sn")
assert basarili_istek_sayisi == 5, "Hiz siniri asilamadi!"

print("\n" + "=" * 60)
print("TÜM TESTLER BAŞARIYLA TAMAMLANDI.")
print("=" * 60)

sunucu.shutdown()
