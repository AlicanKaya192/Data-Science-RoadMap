################################################
# FASTAPI İLE KÜTÜPHANE YÖNETİM API'Sİ (PRODUCTION-GRADE ÖRNEK)
################################################

# İş Problemi: 22.6.1-Python_ile_API_Geliştirmek_FastAPI.pdf dosyasında anlattığımız katmanlı
# mimariyi (veri, şema, kimlik doğrulama, iş mantığı, sunum) uçtan uca, gerçek bir veritabanına
# (SQLite + SQLAlchemy) yazan, JWT ile korunan, sayfalanan ve filtrelenen, tam CRUD destekli bir
# REST API'ye dönüştürmek.
#
# Bu script'i çalıştırmak API'yi bir ağ portunda AYAĞA KALDIRMAZ; bunun yerine FastAPI'nin resmi
# test aracı olan TestClient kullanılır (bkz. 22.6.1, Bölüm 3.1) - bu sayede script tamamen
# offline, hızlı ve tekrar üretilebilir (reproducible) şekilde çalışır. Dosyanın en altında,
# gerçek bir sunucu olarak nasıl çalıştırılacağı da (uvicorn ile) ayrıca gösterilmiştir.

# Gerekli kütüphaneleri kurmak için:
# pip install fastapi uvicorn sqlalchemy pyjwt python-multipart pytest

import hashlib
import hmac
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import jwt
from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

##################################################
# 1. VERİ KATMANI (SQLAlchemy ORM Modelleri)
##################################################
# 22.6.1, Bölüm 3'te anlatıldığı gibi: kitaplar ve kullanıcılar gerçek bir SQLite veritabanı
# tablosunda saklanır (bkz. 15-SQL modülü, ilişkisel veritabanı kavramları).

Base = declarative_base()


class KullaniciDB(Base):
    __tablename__ = "kullanicilar"
    id = Column(Integer, primary_key=True, index=True)
    kullanici_adi = Column(String, unique=True, index=True, nullable=False)
    sifre_hash = Column(String, nullable=False)
    kitaplar = relationship("KitapDB", back_populates="sahip")


class KitapDB(Base):
    __tablename__ = "kitaplar"
    id = Column(Integer, primary_key=True, index=True)
    baslik = Column(String, index=True, nullable=False)
    yazar = Column(String, index=True, nullable=False)
    yil = Column(Integer, nullable=True)
    musait_mi = Column(Boolean, default=True)
    sahip_id = Column(Integer, ForeignKey("kullanicilar.id"))
    sahip = relationship("KullaniciDB", back_populates="kitaplar")


##################################################
# 2. ŞEMA KATMANI (Pydantic Modelleri)
##################################################
# 22.6.1, Bölüm 3'te vurgulandığı gibi: veritabanı modelleri (yukarıda) ile API'nin dışarıya
# gösterdiği şemalar (aşağıda) BİLİNÇLİ olarak AYRILIR - örn. sifre_hash ASLA dışarı sızmaz.

class KullaniciOlustur(BaseModel):
    kullanici_adi: str = Field(..., min_length=3, max_length=50, examples=["ayse_yilmaz"])
    sifre: str = Field(..., min_length=6, examples=["guclu-sifre-123"])


class KullaniciYanit(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    kullanici_adi: str


class KitapOlustur(BaseModel):
    baslik: str = Field(..., min_length=1, max_length=200, examples=["Suç ve Ceza"])
    yazar: str = Field(..., min_length=1, max_length=100, examples=["Dostoyevski"])
    yil: Optional[int] = Field(None, ge=0, le=2100, examples=[1866])


class KitapGuncelle(BaseModel):
    """PATCH icin - tum alanlar opsiyonel, sadece gonderilen alan degisir (bkz. 22.2.1, Bolum 3)."""
    baslik: Optional[str] = None
    yazar: Optional[str] = None
    yil: Optional[int] = None
    musait_mi: Optional[bool] = None


class KitapYanit(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    baslik: str
    yazar: str
    yil: Optional[int]
    musait_mi: bool
    sahip_id: int


class KitapSayfaYaniti(BaseModel):
    """22.3.1, Bolum 4'te anlatilan offset tabanli sayfalama yanit zarfi (envelope)."""
    toplam: int
    sayfa: int
    limit: int
    kitaplar: List[KitapYanit]


class TokenYaniti(BaseModel):
    access_token: str
    token_type: str = "bearer"


##################################################
# 3. GÜVENLİK YARDIMCILARI (Şifre Hash'leme + JWT)
##################################################
# Egitim ortaminda kurulum kolayligi icin (bcrypt bazi ortamlarda derleme sorunu cikarabiliyor),
# Python'un yerlesik hashlib.pbkdf2_hmac fonksiyonuyla, tuzlanmis (salted) sifre hash'leme
# uyguluyoruz. Production'da passlib[bcrypt]/argon2 gibi ozel tasarlanmis kutuphaneler tercih
# edilmelidir; temel prensip (asla duz metin sifre saklamamak) aynidir.

GIZLI_ANAHTAR = os.environ.get("KUTUPHANE_API_GIZLI_ANAHTAR", "egitim-amacli-demo-anahtari-asla-production-da-kullanma")
JWT_ALGORITMA = "HS256"
TOKEN_GECERLILIK_DAKIKA = 30


def sifreyi_hashle(sifre: str) -> str:
    tuz = os.urandom(16)
    hash_deger = hashlib.pbkdf2_hmac("sha256", sifre.encode(), tuz, 100_000)
    return tuz.hex() + ":" + hash_deger.hex()


def sifre_dogru_mu(sifre: str, kayitli_hash: str) -> bool:
    tuz_hex, hash_hex = kayitli_hash.split(":")
    tuz = bytes.fromhex(tuz_hex)
    beklenen = hashlib.pbkdf2_hmac("sha256", sifre.encode(), tuz, 100_000)
    return hmac.compare_digest(beklenen.hex(), hash_hex)


def jwt_token_uret(kullanici_adi: str) -> str:
    """bkz. 22.4.1, Bolum 2.3 - JWT'nin 3 parcasi (header, payload, signature)."""
    son_kullanma = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_GECERLILIK_DAKIKA)
    payload = {"sub": kullanici_adi, "exp": son_kullanma}
    return jwt.encode(payload, GIZLI_ANAHTAR, algorithm=JWT_ALGORITMA)


##################################################
# 4. VERİTABANI BAĞLANTISI VE BAĞIMLILIK (DEPENDENCY) FONKSİYONLARI
##################################################

# Bellek-ici (in-memory) SQLite + StaticPool: TUM baglantilar AYNI tek veritabani baglantisini
# paylasir, boylece veri kaybolmaz (normal :memory: her yeni baglantida BOS bir DB verir) ama
# hicbir DOSYA da diske yazilmaz. Bu, script her calistirildiginda (ya da pytest her import
# ettiginde) SIFIRDAN, tamamen izole bir veritabaniyla baslamasini garanti eder - onceki
# calismalardan kalan veri "sizintisi" (test pollution) riskini TAMAMEN ortadan kaldirir.
# Production'da bunun yerine 'sqlite:///./kutuphane.db' (kalici dosya) ya da PostgreSQL/MySQL
# baglanti dizesi kullanilir.
from sqlalchemy.pool import StaticPool

DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_db():
    """22.6.1, Bolum 2.3'te anlatilan Dependency Injection ornegi: her istek icin AYRI bir
    veritabani oturumu acilir ve istek bitince otomatik kapatilir."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def mevcut_kullaniciyi_getir(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> KullaniciDB:
    """Bu da bir Dependency: Authorization: Bearer <token> basligini cozup, gecerli kullaniciyi
    dondurur. Kimlik dogrulama gerektiren HER endpoint sadece bu fonksiyonu Depends() ile cagirir."""
    kimlik_hatasi = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Kimlik doğrulanamadı",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, GIZLI_ANAHTAR, algorithms=[JWT_ALGORITMA])
        kullanici_adi = payload.get("sub")
        if kullanici_adi is None:
            raise kimlik_hatasi
    except jwt.PyJWTError:
        raise kimlik_hatasi

    kullanici = db.query(KullaniciDB).filter(KullaniciDB.kullanici_adi == kullanici_adi).first()
    if kullanici is None:
        raise kimlik_hatasi
    return kullanici


##################################################
# 5. FASTAPI UYGULAMASI VE ORTAK AYARLAR
##################################################

app = FastAPI(
    title="Kütüphane Yönetim API'si",
    description="22-API modülü, 22.6 bölümü için örnek üretim-kalitesinde REST API.",
    version="1.0.0",
)

# bkz. 22.4.1, Bolum 4.2 ve 22.6.1, Bolum 4 - production'da ASLA allow_origins=["*"] kullanilmaz.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

##################################################
# 6. UÇ NOKTALAR (PATH OPERATIONS)
##################################################


@app.post("/register", response_model=KullaniciYanit, status_code=status.HTTP_201_CREATED, tags=["Kimlik Doğrulama"])
def kayit_ol(veri: KullaniciOlustur, db: Session = Depends(get_db)):
    """Yeni kullanıcı oluşturur. Kullanıcı adı benzersiz olmalıdır."""
    mevcut = db.query(KullaniciDB).filter(KullaniciDB.kullanici_adi == veri.kullanici_adi).first()
    if mevcut:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bu kullanıcı adı zaten kayıtlı")
    yeni_kullanici = KullaniciDB(kullanici_adi=veri.kullanici_adi, sifre_hash=sifreyi_hashle(veri.sifre))
    db.add(yeni_kullanici)
    db.commit()
    db.refresh(yeni_kullanici)
    return yeni_kullanici


@app.post("/token", response_model=TokenYaniti, tags=["Kimlik Doğrulama"])
def giris_yap(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """bkz. 22.4.1, Bolum 2.3 - kullanici adi/sifre dogrulanir, karsiliginda kisa omurlu bir
    JWT access token doner. OAuth2PasswordRequestForm, form-data (x-www-form-urlencoded)
    bekler - bu yuzden python-multipart bagimliligi gereklidir."""
    kullanici = db.query(KullaniciDB).filter(KullaniciDB.kullanici_adi == form_data.username).first()
    if not kullanici or not sifre_dogru_mu(form_data.password, kullanici.sifre_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı adı veya şifre hatalı",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": jwt_token_uret(kullanici.kullanici_adi), "token_type": "bearer"}


@app.get("/me", response_model=KullaniciYanit, tags=["Kimlik Doğrulama"])
def profilim(mevcut_kullanici: KullaniciDB = Depends(mevcut_kullaniciyi_getir)):
    """Giris yapmis kullanicinin kendi bilgisini dondurur - Depends() zincirinin en basit ornegi."""
    return mevcut_kullanici


@app.post("/books", response_model=KitapYanit, status_code=status.HTTP_201_CREATED, tags=["Kitaplar"])
def kitap_ekle(
    veri: KitapOlustur,
    mevcut_kullanici: KullaniciDB = Depends(mevcut_kullaniciyi_getir),
    db: Session = Depends(get_db),
):
    """Yeni kitap ekler. Kimlik dogrulama ZORUNLUDUR (bkz. 22.4.1)."""
    yeni_kitap = KitapDB(**veri.model_dump(), sahip_id=mevcut_kullanici.id)
    db.add(yeni_kitap)
    db.commit()
    db.refresh(yeni_kitap)
    return yeni_kitap


@app.get("/books", response_model=KitapSayfaYaniti, tags=["Kitaplar"])
def kitaplari_listele(
    sayfa: int = Query(1, ge=1, description="Sayfa numarası (1'den başlar)"),
    limit: int = Query(10, ge=1, le=100, description="Sayfa başına kayıt sayısı"),
    yazar: Optional[str] = Query(None, description="Yazara göre filtrele (kısmi eşleşme)"),
    sadece_musait: bool = Query(False, description="Sadece ödünç alınabilir kitapları göster"),
    db: Session = Depends(get_db),
):
    """Sayfalanmis, filtrelenebilir kitap listesi. KIMLIK DOGRULAMA GEREKTIRMEZ - GET /books
    genel (public) bir okuma islemidir (bkz. 22.4.1, Bolum 1 - AuthN/AuthZ ayrimi: okuma herkese
    acik olabilir, yazma islemleri kisitlanir)."""
    sorgu = db.query(KitapDB)
    if yazar:
        sorgu = sorgu.filter(KitapDB.yazar.ilike(f"%{yazar}%"))
    if sadece_musait:
        sorgu = sorgu.filter(KitapDB.musait_mi == True)  # noqa: E712

    toplam = sorgu.count()
    kitaplar = sorgu.offset((sayfa - 1) * limit).limit(limit).all()
    return {"toplam": toplam, "sayfa": sayfa, "limit": limit, "kitaplar": kitaplar}


@app.get("/books/{kitap_id}", response_model=KitapYanit, tags=["Kitaplar"])
def kitap_getir(kitap_id: int, db: Session = Depends(get_db)):
    kitap = db.query(KitapDB).filter(KitapDB.id == kitap_id).first()
    if not kitap:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{kitap_id} numaralı kitap bulunamadı")
    return kitap


def _kitabi_ve_sahiplik_kontrolunu_yap(kitap_id: int, mevcut_kullanici: KullaniciDB, db: Session) -> KitapDB:
    """22.4.1, Bolum 4'teki BOLA (Broken Object Level Authorization) riskine karsi ornek onlem:
    sadece token'in gecerli olmasi yetmez, kaynagin GERCEKTEN bu kullaniciya ait olup olmadigi da
    KONTROL EDILIR."""
    kitap = db.query(KitapDB).filter(KitapDB.id == kitap_id).first()
    if not kitap:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{kitap_id} numaralı kitap bulunamadı")
    if kitap.sahip_id != mevcut_kullanici.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Bu kitabı değiştirme yetkiniz yok")
    return kitap


@app.patch("/books/{kitap_id}", response_model=KitapYanit, tags=["Kitaplar"])
def kitap_kismi_guncelle(
    kitap_id: int,
    veri: KitapGuncelle,
    mevcut_kullanici: KullaniciDB = Depends(mevcut_kullaniciyi_getir),
    db: Session = Depends(get_db),
):
    """bkz. 22.2.1, Bolum 3 - PATCH sadece GONDERILEN alanlari degistirir."""
    kitap = _kitabi_ve_sahiplik_kontrolunu_yap(kitap_id, mevcut_kullanici, db)
    guncellenecek_alanlar = veri.model_dump(exclude_unset=True)
    for alan, deger in guncellenecek_alanlar.items():
        setattr(kitap, alan, deger)
    db.commit()
    db.refresh(kitap)
    return kitap


@app.delete("/books/{kitap_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Kitaplar"])
def kitap_sil(
    kitap_id: int,
    mevcut_kullanici: KullaniciDB = Depends(mevcut_kullaniciyi_getir),
    db: Session = Depends(get_db),
):
    kitap = _kitabi_ve_sahiplik_kontrolunu_yap(kitap_id, mevcut_kullanici, db)
    db.delete(kitap)
    db.commit()
    return None


##################################################
# 7. TESTLER (pytest ile de calistirilabilir - dosya adi rakamla basladigi icin:
#    `pytest --import-mode=importlib 22.6.2_kutuphane_api.py`)
##################################################
# TestClient, GERCEK bir ag baglantisi kurmadan, FastAPI uygulamasinin TUM middleware/routing/
# validation katmanlarindan gecen sureç-ici (in-process) istekler yapar (bkz. 22.6.1, Bolum 3.1).

client = TestClient(app)


def _benzersiz_kullanici_adi():
    return f"test_kullanici_{int(time.time() * 1_000_000) % 10_000_000}"


def _yeni_kullanici_ve_token(sifre: str = "guclu-sifre-123") -> str:
    """Yardimci fonksiyon (pytest tarafindan test olarak TOPLANMAZ - ismi 'test_' ile baslamiyor):
    yeni bir kullanici kaydedip token'ini dondurur, diger testlerde tekrar tekrar kullanilir."""
    kullanici_adi = _benzersiz_kullanici_adi()
    client.post("/register", json={"kullanici_adi": kullanici_adi, "sifre": sifre})
    giris_yaniti = client.post("/token", data={"username": kullanici_adi, "password": sifre})
    return giris_yaniti.json()["access_token"]


def test_kayit_ve_giris_akisi():
    kullanici_adi = _benzersiz_kullanici_adi()
    kayit_yaniti = client.post("/register", json={"kullanici_adi": kullanici_adi, "sifre": "guclu-sifre-123"})
    assert kayit_yaniti.status_code == 201, kayit_yaniti.text
    assert kayit_yaniti.json()["kullanici_adi"] == kullanici_adi

    # Ayni kullanici adiyla ikinci kayit -> 400 beklenir.
    tekrar_yaniti = client.post("/register", json={"kullanici_adi": kullanici_adi, "sifre": "baska-sifre"})
    assert tekrar_yaniti.status_code == 400

    giris_yaniti = client.post("/token", data={"username": kullanici_adi, "password": "guclu-sifre-123"})
    assert giris_yaniti.status_code == 200
    assert "access_token" in giris_yaniti.json()


def test_yanlis_sifre_401_doner():
    kullanici_adi = _benzersiz_kullanici_adi()
    client.post("/register", json={"kullanici_adi": kullanici_adi, "sifre": "dogru-sifre"})
    yanit = client.post("/token", data={"username": kullanici_adi, "password": "yanlis-sifre"})
    assert yanit.status_code == 401


def test_token_olmadan_kitap_eklenemez():
    yanit = client.post("/books", json={"baslik": "Test Kitap", "yazar": "Test Yazar"})
    assert yanit.status_code == 401


def test_kitap_crud_akisi():
    token = _yeni_kullanici_ve_token()
    basliklar = {"Authorization": f"Bearer {token}"}

    olustur_yaniti = client.post("/books", json={"baslik": "Suç ve Ceza", "yazar": "Dostoyevski", "yil": 1866}, headers=basliklar)
    assert olustur_yaniti.status_code == 201
    kitap = olustur_yaniti.json()
    assert kitap["musait_mi"] is True
    kitap_id = kitap["id"]

    getir_yaniti = client.get(f"/books/{kitap_id}")
    assert getir_yaniti.status_code == 200
    assert getir_yaniti.json()["baslik"] == "Suç ve Ceza"

    patch_yaniti = client.patch(f"/books/{kitap_id}", json={"musait_mi": False}, headers=basliklar)
    assert patch_yaniti.status_code == 200
    assert patch_yaniti.json()["musait_mi"] is False
    assert patch_yaniti.json()["yazar"] == "Dostoyevski", "PATCH sadece gonderilen alani degistirmeli!"

    sil_yaniti = client.delete(f"/books/{kitap_id}", headers=basliklar)
    assert sil_yaniti.status_code == 204

    tekrar_getir_yaniti = client.get(f"/books/{kitap_id}")
    assert tekrar_getir_yaniti.status_code == 404


def test_baskasinin_kitabini_silemezsin_403():
    token_a = _yeni_kullanici_ve_token()
    token_b = _yeni_kullanici_ve_token()

    olustur_yaniti = client.post(
        "/books", json={"baslik": "A'nin Kitabi", "yazar": "Biri"}, headers={"Authorization": f"Bearer {token_a}"}
    )
    kitap_id = olustur_yaniti.json()["id"]

    sil_denemesi = client.delete(f"/books/{kitap_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert sil_denemesi.status_code == 403, "BOLA korumasi calismiyor - baskasinin kitabi silinebildi!"


def test_sayfalama_ve_filtreleme():
    token = _yeni_kullanici_ve_token()
    basliklar = {"Authorization": f"Bearer {token}"}
    for i in range(15):
        client.post("/books", json={"baslik": f"Sayfalama Kitabi {i}", "yazar": "OrtakYazarX"}, headers=basliklar)

    sayfa1 = client.get("/books", params={"sayfa": 1, "limit": 10, "yazar": "OrtakYazarX"})
    assert sayfa1.status_code == 200
    veri = sayfa1.json()
    assert veri["toplam"] == 15
    assert len(veri["kitaplar"]) == 10

    sayfa2 = client.get("/books", params={"sayfa": 2, "limit": 10, "yazar": "OrtakYazarX"})
    assert len(sayfa2.json()["kitaplar"]) == 5


def test_gecersiz_veri_422_doner():
    """bkz. 22.6.1, Bolum 2.2 - Pydantic dogrulamasi basarisiz olursa FastAPI otomatik 422 doner."""
    token = _yeni_kullanici_ve_token()
    basliklar = {"Authorization": f"Bearer {token}"}
    # 'baslik' alani ZORUNLU ama gonderilmedi.
    yanit = client.post("/books", json={"yazar": "Biri"}, headers=basliklar)
    assert yanit.status_code == 422


if __name__ == "__main__":
    testler = [
        test_kayit_ve_giris_akisi,
        test_yanlis_sifre_401_doner,
        test_token_olmadan_kitap_eklenemez,
        test_kitap_crud_akisi,
        test_baskasinin_kitabini_silemezsin_403,
        test_sayfalama_ve_filtreleme,
        test_gecersiz_veri_422_doner,
    ]
    print(f"Toplam {len(testler)} test çalıştırılıyor...\n")
    for test_fn in testler:
        test_fn()
        print(f"  [OK] {test_fn.__name__}")
    print("\n" + "=" * 60)
    print("TÜM TESTLER BAŞARIYLA TAMAMLANDI.")
    print("=" * 60)

    print("\nOluşan OpenAPI şemasındaki uç nokta sayısı:", len(app.openapi()["paths"]))

    print("\nNot: Bu API'yi gercek bir sunucu olarak calistirmak icin (bu script'in disinda):")
    print("  uvicorn 22.6.2_kutuphane_api:app --reload")
    print("Sonra tarayicidan http://127.0.0.1:8000/docs adresini ziyaret edin (Swagger UI, bkz. 22.7).")
