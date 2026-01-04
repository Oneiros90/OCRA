# OCR + Traduzione in Italiano

Questo progetto fornisce uno script Python che estrae il testo da un'immagine tramite OCR e lo traduce automaticamente in italiano usando un modello AI multilingua.

## Requisiti
- macOS o Windows con Python 3.10+
- Accesso a internet la prima volta (per scaricare i pesi dei modelli EasyOCR e Hugging Face)

## Setup ambiente virtuale
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Utilizzo
```bash
python ocr_translate.py path/alla/immagine.jpg \
  --ocr-langs ru,en,it,fr,es,de,pt \
  --model-choice marian_ru_it \
  --max-chars 600 \
  --source-lang ru \
  --save output.txt
# aggiungi --allow-gpu per usare una GPU disponibile
```

Parametri principali:
- `image` (posizionale): percorso dell'immagine da processare.
- `--ocr-langs`: elenco (separato da virgole) dei codici lingua supportati da EasyOCR da caricare (default include `ru` per il cirillico oltre a en/it/fr/es/de/pt).
- `--model-choice`: preset di modello Hugging Face (m2m100, marian_ru_it, mbart50) con default `m2m100`.
- `--model`: ID personalizzato su Hugging Face che sovrascrive `--model-choice`.
- `--max-chars`: lunghezza massima dei chunk inviati al modello di traduzione.
- `--source-lang`: codice lingua (es. `ru`, `en`, `fr`) del testo OCR riconosciuto; viene passato direttamente al modello di traduzione.
- `--force-cpu`: forza l'uso della CPU; ora è attivo di default.
- `--allow-gpu`: disattiva `--force-cpu` e consente l'uso della GPU se presente.
- `--save`: se indicato, salva il testo tradotto nel percorso scelto.

## Note tecniche
- EasyOCR supporta oltre 80 lingue: consulta la documentazione ufficiale per l'elenco completo dei codici da usare in `--ocr-langs`.
- Il modello `facebook/m2m100_418M` gestisce centinaia di lingue e produce output in italiano forzando il token `it`. Puoi sostituirlo con modelli MarianMT o MBART se preferisci.
- Alla prima esecuzione verranno scaricati i pesi; successivamente verranno riutilizzati dalla cache locale.
- Se desideri velocizzare l'esecuzione e disponi di una GPU compatibile, installa la variante di Torch ottimizzata per il tuo sistema e non usare `--force-cpu`.
- Alcune lingue (es. cirillico: `ru`, `uk`, `bg`, ecc.) richiedono un modello EasyOCR separato compatibile solo con l'inglese: lo script crea automaticamente più reader e aggrega i risultati in un'unica uscita. Ricordati di passare il relativo codice anche a `--source-lang`.
