# ADR-002: Vektör Veritabanı Seçimi — FAISS vs Alternatifleri

## Bağlam

RAG mimarisinde kullanıcı sorusu ile ilgili döküman parçalarını (chunk) bulmak için bir vektör veritabanına ihtiyaç var. Metin parçaları sayısal vektörlere dönüştürülüp bu veritabanında saklanıyor, sorgu geldiğinde anlam benzerliğine göre en yakın parçalar getiriliyor.

## Değerlendirilen Seçenekler

### Seçenek A — Cloud Tabanlı (Pinecone, Weaviate, Qdrant Cloud)
- Vektörler dışarıdaki sunucularda saklanır
- Aylık abonelik ücreti var
- İnternet bağlantısı zorunlu
- Döküman içeriği dışarıya çıkar

### Seçenek B — Chroma (Local veya Cloud)
- Local modda çalışabilir
- Kurulumu FAISS'e göre daha karmaşık
- Cloud versiyonu da var, ikisi arasında geçiş riski var

### Seçenek C — FAISS (Meta AI Research)
- Tamamen local, sadece bir kütüphane
- Kurulum yok, servis yok, ücret yok
- Tüm vektörler `index.faiss` ve `index.pkl` dosyalarında, yerel makinede
- İnternet bağlantısı sıfır

## Karar: FAISS

İki sebepten dolayı:

**1. Local ve Ücretsiz:** ADR-001'deki LLM kararıyla aynı mantık. Veri dışarı çıkmaz, maliyet sıfır, bağımlılık yok.

**2. Ölçek Uyumu:** FAISS milyonlarca vektörde yavaşlar. Ancak bu projede kaynak bir SSS dökümanı — birkaç yüz chunk. Bu ölçek için FAISS idealdir. Büyük ölçekli production sistemi olsaydı cloud alternatifler değerlendirilebilirdi.

## Sonuçlar

**Artılar:**
- Sıfır maliyet
- Veri tamamen yerel
- Basit kullanım, ekstra servis gerektirmez

**Eksiler:**
- Çok büyük veri setlerinde performans düşer
- Dağıtık (distributed) sistem kurulamaz
- Gerçek zamanlı güncelleme için ek iş gerekir
