---
title: AutoPrice Pro
emoji: "🚗"
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# AutoPrice Pro — KI-Preisschätzer für Gebrauchtwagen

**Semesterprojekt KI-Anwendungen (ZHAW, FS 2026).** Eine integrierte App, die
**ML Numeric Data**, **NLP** und **Computer Vision** in einer einzigen Pipeline
kombiniert.

## Was die App tut

1. Der Nutzer beschreibt einen Gebrauchtwagen in Freitext (Deutsch oder Englisch)
   und kann optional ein Foto hochladen.
2. **NLP**-Block: Ein LLM extrahiert strukturierte Felder (Marke, Baujahr,
   Kilometerstand, Kraftstoff, …) aus dem Text.
3. **Computer-Vision**-Block: GPT-4o-mini Vision klassifiziert den Karosserietyp
   aus dem Bild und gibt optional eine Markenvermutung ab.
4. **ML**-Block: Eine trainierte scikit-learn-Pipeline (GradientBoosting auf
   `log(price)`) kombiniert beides zu einer Preisvorhersage in EUR.
5. Das LLM erstellt eine kurze deutsche Erklärung der Vorhersage inkl.
   Unsicherheits-Hinweis.

## Schneller Einstieg

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. OpenAI-Key setzen (PowerShell):
$env:OPENAI_API_KEY = "sk-..."

# 3. Modell trainieren (erzeugt artifacts/final_model.joblib u.a.)
python src/train.py

# 3b. (optional) EDA-Plots & Key Findings regenerieren
python src/eda.py

# 4. App starten
python app.py
```

## Projektstruktur

```
KI_Anwendungen_Projekt/
├── app.py                          # Gradio-App (integriert alle drei Blöcke)
├── requirements.txt
├── README.md                       # diese Datei
├── documentation.md                # Pflicht-Doku nach Template
├── src/
│   ├── train.py                    # ML-Trainingspipeline (3 Iterationen)
│   ├── eda.py                      # EDA: Verteilungen, Korrelationen, Plots
│   ├── data_processing.py          # geteilte Feature-Engineering-Funktionen
│   ├── llm_client.py               # OpenAI-Hilfsfunktionen (Text + Vision)
│   ├── nlp_block.py                # Extraktion + Erklärung
│   └── vision_block.py             # Karosserie-Klassifikation
├── artifacts/
│   ├── final_model.joblib          # gespeichertes Modell
│   ├── metadata.json
│   ├── brand_defaults.csv          # marken-spezifische Defaults
│   ├── body_type_defaults.csv
│   ├── model_iterations.md         # automatisch generierter Trainings-Report
│   └── eda/                        # EDA-Output (auto-generiert von eda.py)
│       ├── eda_summary.md
│       └── *.png                   # 7 Plots (Verteilungen, Korrelationen, ...)
├── data/
│   └── autoscout24_germany.csv     # Haupt-Datensatz (46 405 Zeilen, EUR)
├── docs/
│   └── documentation_template_reference.md   # Original-Template als Referenz
├── examples/                       # (optional) Beispielbilder für die App
└── screenshots/                    # Screenshots für die Doku
```

## Datenquellen (neu, nicht aus dem Semester)

| Quelle | Typ | Grösse | Rolle |
|---|---|---|---|
| AutoScout24 Germany Dataset ([Mirror](https://raw.githubusercontent.com/leander-ms/autoscout_Analysis/main/autoscout24-germany-dataset.csv)) | strukturiert (CSV) | 46 405 Zeilen | ML-Trainingsdaten |
| OpenAI GPT-4o-mini (Text) | Sprachmodell | – | NLP-Extraktion + Erklärung |
| OpenAI GPT-4o-mini (Vision) | multimodales Modell | – | CV-Klassifikation |
| Benutzer-Foto | Bilddaten (Upload) | 1 pro Anfrage | CV-Input |

Die in der Aufgabenstellung explizit ausgeschlossenen Datenquellen
(Mietwohnungen Kanton Zürich, Hunderassen-Bilder) werden **nicht** verwendet.
Der Datensatz stammt aus dem deutschen Markt und ist damit für Schweizer
Käuferinnen und Käufer praktisch direkt anwendbar (EUR ≈ CHF, Stand 2026).

## Notwendige Umgebungsvariablen

| Variable | Pflicht | Default | Zweck |
|---|---|---|---|
| `OPENAI_API_KEY` | ja | – | NLP- und CV-Aufrufe |
| `OPENAI_MODEL` | nein | `gpt-4o-mini` | Text-LLM |
| `OPENAI_VISION_MODEL` | nein | `gpt-4o-mini` | Vision-LLM |

## Deployment

Die App wird als **Hugging Face Space** deployed (Gradio-SDK).
Beim Deployment muss `OPENAI_API_KEY` als _Secret_ gesetzt werden.

Falls die Artefakte (`artifacts/final_model.joblib` etc.) nicht im Repo
liegen, **trainiert die App sich beim ersten Start selbst** (auto-download
der CSV von GitHub, ~30-90 Sekunden Erstboot).

## Quick-Check: läuft alles?

```bash
python src/train.py        # erzeugt Artefakte und meldet CV-Metriken
python app.py              # startet Gradio auf http://127.0.0.1:7860
```

## Lizenz / Verantwortung

Die Preisvorhersage ist eine Schätzung auf Basis öffentlicher AutoScout24-Daten
aus Deutschland (Preise in EUR). Sie ersetzt keine professionelle Bewertung.
Details siehe `documentation.md`.
