# OCR AI Translator

End-to-end toolchain for extracting text from screenshots, documents, and UI mockups, then translating the content with an AI assistant. The project ships with both a command-line workflow and a modern Qt desktop application featuring live overlays, color customization, and LLM-backed translations.

## Requirements
- macOS or Windows with Python 3.10+
- Internet access the first time you run OCR (EasyOCR will download its model bundles) and whenever you request an OpenAI translation

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## CLI Usage
The legacy CLI still ships for batch scenarios and uses EasyOCR + Hugging Face models.

```bash
python ocr_translate.py path/to/image.jpg \
  --ocr-langs ru,en,it,fr,es,de,pt \
  --model-choice marian_ru_it \
  --max-chars 600 \
  --source-lang ru \
  --save output.txt

# add --allow-gpu to leverage a compatible GPU
```

Key options:
- `image`: path to the image you want to process.
- `--ocr-langs`: comma-separated EasyOCR language codes to preload (pick the smallest list that covers your document).
- `--model-choice`: Hugging Face preset (`m2m100`, `marian_ru_it`, `mbart50`).
- `--model`: explicit Hugging Face model ID that overrides `--model-choice`.
- `--max-chars`: upper bound for each translation chunk.
- `--source-lang`: language hint for the extracted text.
- `--force-cpu` / `--allow-gpu`: toggle hardware acceleration.
- `--save`: write the translated text to disk.

## Desktop Application
Launch the graphical interface with:

```bash
python ocr_gui.py
```

Highlights:
- **Image workflow** – load high-resolution assets, reset zoom-to-fit, and inspect overlays rendered directly on the canvas.
- **OCR controls** – pick input languages, hardware mode, and paragraph grouping before starting recognition.
- **Translation controls** – provide your OpenAI API key, desired target language, and optional “attach image” context to boost accuracy.
- **LLM overlays** – switch between original OCR text and translated text on each bounding box.
- **Color + export** – dedicated pickers for text/fill colors (alpha supported) and a one-click export of the annotated image.
- **Localization** – the UI auto-detects the system locale; English is the default and Italian is available when macOS/Windows is set to Italian.

## Localization
All user-facing strings live under `ocr_ai/gui/i18n/translations`. Extend the app to new languages by dropping additional JSON files into that directory. Each key is shared by the CLI helpers, worker threads, and Qt widgets, so adding a locale instantly updates the entire experience.

## Building Standalone Binaries
Create redistributable packages with PyInstaller.

### macOS (.app)
```bash
pyinstaller ocr_gui.py \
  --name OCRA \
  --noconsole \
  --add-data "ocr_ai:ocr_ai"

open dist/OCRA.app
```

### Windows (.exe)
```powershell
pyinstaller ocr_gui.py \
  --name OCRA \
  --noconsole \
  --add-data "ocr_ai;ocr_ai"

start dist/OCRA/OCRA.exe
```

Tips:
- Sign / notarize the macOS bundle before distribution.
- Freeze dependencies inside the virtual environment you used for packaging.
- Provide a custom icon via `--icon path/to/icon.ico` for a branded look.

## Troubleshooting
- The first OCR run downloads EasyOCR weights; allow a few minutes and keep the app open.
- Translation requires a valid OpenAI API key with access to `gpt-4o-mini` or a compatible model.
- GPU acceleration depends on your local PyTorch build; reinstall torch/torchvision with CUDA support if detection fails.
