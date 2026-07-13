# ADR-001: LLM Seçimi — Cloud vs Local

## Bağlam

Müşteri destek botu, kullanıcıların sorularını doğal dille yanıtlıyor. Bu yanıtları üretecek bir dil modeline ihtiyacımız var. İki ana kategori değerlendirildi.

## Değerlendirilen Seçenekler

### Seçenek A — Cloud LLM (OpenAI GPT, Google Gemini, Anthropic Claude...)
- Her sorgu dışarıdaki bir sunucuya gider
- Token başına ücret ödenir
- İnternet bağlantısı zorunlu
- Müşteri verisi şirket dışına çıkar

### Seçenek B — Local LLM (Ollama + Gemma3:4b)
- Her şey şirketin kendi makinesinde çalışır
- Kurulum maliyeti dışında ek ücret yok
- İnternet bağlantısı gerekmez
- Veri dışarı çıkmaz

## Karar: Local LLM (Ollama + Gemma3:4b)

Üç sebepten dolayı:

**1. Maliyet:** Cloud LLM'lerde her sorgu ücretlidir. Günde binlerce müşteri sorusu olan bir e-ticaret sisteminde bu maliyet aylık yüzlerce dolara ulaşabilir. Local modelde bu maliyet sıfırdır.

**2. Veri Gizliliği ve Yasal Uyumluluk:** Müşteri sipariş bilgileri, adres ve kişisel veriler içerebilir. KVKK (Türkiye) ve GDPR (Avrupa) bu tür verilerin yurt dışı sunuculara gönderilmesini kısıtlar. Local çalışmak bu yasal riski tamamen ortadan kaldırır.

**3. Bağımsızlık:** Dış servise bağımlı olmamak, internet kesintisi veya API değişikliklerinden etkilenmemek anlamına gelir.

## Sonuçlar

**Artılar:**
- Sıfır işletme maliyeti
- Tam veri gizliliği
- İnternet bağımsız çalışma

**Eksiler:**
- Güçlü bir local LLM çalıştırmak için yeterli donanım (RAM/GPU) gerekir
- Cloud modellere kıyasla yanıt kalitesi daha düşük olabilir
- Model güncellemeleri manuel yapılır
