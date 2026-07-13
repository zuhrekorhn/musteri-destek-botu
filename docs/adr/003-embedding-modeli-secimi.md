# ADR-003: Embedding Modeli Seçimi — LaBSE vs Alternatifleri

## Bağlam

Vektör veritabanına metin parçalarını eklemek ve kullanıcı sorgusunu aramak için bir embedding modeli gerekiyor. Bu model metni sayısal vektöre çeviriyor. Sistemin Türkçe sorulara doğru cevap verebilmesi için modelin Türkçeyi anlam düzeyinde kavraması şart.

## Değerlendirilen Seçenekler

### Seçenek A — text-embedding-ada-002 (OpenAI)
- Yüksek kaliteli embedding
- API key gerektirir, token başına ücretli
- Veri dışarıya gider
- Türkçe desteği sınırlı

### Seçenek B — all-MiniLM-L6-v2 (HuggingFace)
- Local çalışır, ücretsiz
- İngilizce için iyi ama Türkçe semantik benzerlikte zayıf

### Seçenek C — LaBSE (Language-agnostic BERT Sentence Embeddings)
- Google tarafından geliştirildi
- 109 dili destekler, Türkçe dahil
- Local çalışır, ücretsiz
- Türkçe semantik benzerlikte yüksek doğruluk

## Karar: LaBSE

İki sebepten dolayı:

**1. Türkçe Semantik Anlama:** Sistem Türkçe SSS dökümanını işliyor ve Türkçe sorulara cevap veriyor. Türkçe bilmeyen bir model "iade" ile "geri vermek" arasındaki anlam bağını kuramaz. Bu durumda kullanıcı doğru soruyu sorsa bile sistem yanlış chunk getirir, yanlış veya boş cevap üretir. Buna semantic mismatch denir.

**2. Local ve Ücretsiz:** ADR-001 ve ADR-002 ile aynı mimari karar çizgisi. API bağımlılığı yok, veri dışarı çıkmıyor, maliyet sıfır.

## Sonuçlar

**Artılar:**
- Türkçe dahil 109 dil desteği
- Tamamen local, sıfır maliyet
- Yüksek semantik doğruluk

**Eksiler:**
- all-MiniLM gibi hafif modellere kıyasla daha fazla RAM kullanır
- İlk yüklemede model dosyaları indirileceğinden internet bağlantısı bir kez gerekir
