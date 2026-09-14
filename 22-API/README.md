# API (Application Programming Interface) Geliştirme ve Kullanımı

API'nin ne olduğundan başlayıp; HTTP protokolüne, REST tasarım prensiplerine, kimlik doğrulama ve
güvenliğe, Python ile hem var olan API'leri dayanıklı şekilde tüketmeye hem de FastAPI ile kendi
production-grade REST API'nizi sıfırdan inşa etmeye, dokümantasyon/test pratiklerine, alternatif
mimarilere (GraphQL, gRPC, Webhook) ve son olarak production'da API yönetimi/ölçeklendirmeye kadar
uçtan uca, son derece kapsamlı bir modül.

- **22.1 - API Temelleri:**
    - **22.1.1 - API_Nedir_Temel_Kavramlar.pdf:** API'nin tanımı, istemci-sunucu modeli, gerçek
      hayat analojisi (garson), API türleri (Web/Kütüphane/OS/Donanım), erişim modelleri
      (Public/Private/Partner), API ekonomisi (Stripe, Twilio, Google Maps, OpenAI/Anthropic örnekleri).
    - **22.1.2 - Tekrar_İçin_Sorular.pdf:** 22.1 konusuna özel 12 soruluk çoktan seçmeli test ve cevap anahtarı.
- **22.2 - HTTP Protokolü:**
    - **22.2.1 - HTTP_Protokolü_ve_Anatomisi.pdf:** HTTP'nin durumsuzluğu, istek/yanıt anatomisi,
      HTTP metodları ve semantikleri (GET/POST/PUT/PATCH/DELETE), güvenlik (safe) ve idempotentlik
      kavramları, durum kodu aileleri (1xx-5xx), önemli HTTP başlıkları.
    - **22.2.2 - Tekrar_İçin_Sorular.pdf:** 22.2 konusuna özel 12 soruluk çoktan seçmeli test ve cevap anahtarı.
- **22.3 - REST Mimarisi ve Tasarımı:**
    - **22.3.1 - REST_Mimarisi_ve_API_Tasarımı.pdf:** Roy Fielding'in 6 kısıtlaması, Richardson
      Olgunluk Modeli, kaynak modelleme ve URI tasarım kuralları, sayfalama stratejileri
      (offset vs cursor), API sürümleme stratejileri.
    - **22.3.2 - Tekrar_İçin_Sorular.pdf:** 22.3 konusuna özel 12 soruluk çoktan seçmeli test ve cevap anahtarı.
- **22.4 - Kimlik Doğrulama ve Güvenlik:**
    - **22.4.1 - Kimlik_Doğrulama_ve_API_Güvenliği.pdf:** Authentication vs Authorization, API Key/Basic
      Auth/JWT/OAuth 2.0, hız sınırlama algoritmaları (Token Bucket, Leaky Bucket, Sabit/Kayan Pencere),
      OWASP API Güvenliği Top 10 özeti (BOLA dahil), HTTPS/TLS ve CORS.
    - **22.4.2 - Tekrar_İçin_Sorular.pdf:** 22.4 konusuna özel 12 soruluk çoktan seçmeli test ve cevap anahtarı.
- **22.5 - Python ile API Tüketmek:**
    - **22.5.1 - Python_ile_API_Tüketmek.pdf:** requests kütüphanesi temelleri, production kalitesinde
      bir API istemcisinin 5 prensibi (session, akıllı retry, hata ayrımı, önbellekleme, otomatik sayfalama).
    - **22.5.2_resilient_api_client.py / .ipynb:** Tamamen offline, kendi yerel sahte "Kitap Kataloğu"
      API'sine (arka planda thread'de çalışan) karşı test edilen, dayanıklı bir API istemci sınıfı:
      `requests.Session` + `HTTPAdapter`/`Retry` ile otomatik üstel geri çekilme, manuel retry
      döngüsü karşılaştırması, TTL önbellek, otomatik sayfalama generator'ı, hız sınırı (429) ve
      geçici (503) hata senaryoları — 7 ayrı testle doğrulanmış.
    - **22.5.3 - Tekrar_İçin_Sorular.pdf:** 22.5 konusuna özel 12 soruluk çoktan seçmeli test ve cevap anahtarı.
- **22.6 - Python ile API Geliştirmek:**
    - **22.6.1 - Python_ile_API_Geliştirmek_FastAPI.pdf:** FastAPI'nin temel bileşenleri (path
      operations, Pydantic doğrulama, dependency injection, hata yönetimi), katmanlı API mimarisi,
      tutarlı hata formatı ve CORS ayarları.
    - **22.6.2_kutuphane_api.py / .ipynb:** Production-grade bir "Kütüphane Yönetim API'si": SQLAlchemy
      ORM + bellek-içi SQLite, PBKDF2 ile şifre hash'leme, JWT tabanlı kimlik doğrulama, tam CRUD
      (create/read/patch/delete), sayfalama + filtreleme, BOLA korumalı sahiplik kontrolü, FastAPI
      `TestClient` ile 7 ayrı senaryoda test edilmiş (pytest uyumlu).
    - **22.6.3 - Tekrar_İçin_Sorular.pdf:** 22.6 konusuna özel 12 soruluk çoktan seçmeli test ve cevap anahtarı.
- **22.7 - Dokümantasyon ve Test:**
    - **22.7.1 - API_Dokümantasyonu_ve_Test_Etme.pdf:** OpenAPI/Swagger spesifikasyonu, FastAPI'nin
      otomatik `/docs` ve `/redoc` dokümantasyonu, Postman/Insomnia, birim/entegrasyon/sözleşme
      testleri, mock tabanlı test yaklaşımları.
    - **22.7.2 - Tekrar_İçin_Sorular.pdf:** 22.7 konusuna özel 12 soruluk çoktan seçmeli test ve cevap anahtarı.
- **22.8 - GraphQL, gRPC ve Webhooklar:**
    - **22.8.1 - GraphQL_gRPC_ve_Webhooklar.pdf:** REST'in over/under-fetching sınırları, GraphQL,
      gRPC ve Protocol Buffers, webhook'lar (imza doğrulama, idempotentlik), WebSocket'ler.
    - **22.8.2 - Tekrar_İçin_Sorular.pdf:** 22.8 konusuna özel 12 soruluk çoktan seçmeli test ve cevap anahtarı.
- **22.9 - API Yönetimi ve Ölçekleme:**
    - **22.9.1 - API_Yönetimi_Ölçekleme_ve_Gerçek_Dünya_Örnekleri.pdf:** Monolitikten mikroservislere,
      API Gateway deseni, izlenebilirlik (log/metrik/iz), SLA/SLO, Stripe/Twitter-X/GitHub vaka çalışmaları.
    - **22.9.2 - Tekrar_İçin_Sorular.pdf:** 22.9 konusuna özel 12 soruluk çoktan seçmeli test ve cevap anahtarı.
- **22.10 - Genel Tekrar Soruları:**
    - **22.10.1 - Genel_Tekrar_İçin_Sorular.pdf:** Modülün tamamını (22.1 - 22.9) kapsayan 30 soruluk
      çoktan seçmeli test ve cevap anahtarı.

> **🔗 Ek Kaynaklar:**
>
> *   [FastAPI Resmi Dokümantasyonu](https://fastapi.tiangolo.com/) — 22.6 bölümünde kullanılan çerçevenin resmi kılavuzu.
> *   [MDN HTTP Dokümantasyonu](https://developer.mozilla.org/tr/docs/Web/HTTP) — HTTP protokolü için kapsamlı referans.
> *   [OpenAPI Specification](https://www.openapis.org/) — 22.7 bölümünde incelenen API tanımlama standardı.
> *   [OWASP API Security Top 10](https://owasp.org/www-project-api-security/) — 22.4 bölümünde özetlenen güvenlik risklerinin resmi kaynağı.
> *   [Roy Fielding'in REST Tezi](https://ics.uci.edu/~fielding/pubs/dissertation/top.htm) — REST mimarisinin orijinal, akademik kaynağı.
