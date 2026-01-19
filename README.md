# NLP Preprocessing Web App

A Flask web application for **NLP preprocessing**, **sentiment analysis**, and **spam detection** on Indonesian social media comments.

## Features

### Text Preprocessing

- **Cleaning**: Remove URLs, emojis, mentions (`@`), hashtags (`#`), punctuation, and digits
- **Tokenization**: Using NLTK word tokenizer
- **Stopword Removal**: Indonesian stopwords (NLTK + custom fallback list)

### Classification Systems

#### Naive Bayes Classifier

- Uses `CountVectorizer` + `MultinomialNB`
- Hybrid approach combining ML prediction with lexicon lookup for low-confidence cases
- Indonesian sentiment lexicons:
  - **Positive words**: "bagus", "keren", "mantap", etc.
  - **Negative words**: "jelek", "buruk", "kecewa", etc.
  - **Spam indicators**: "klik", "promo", "slot gacor", etc.

#### SVM Classifier

- Uses `TfidfVectorizer` + `LinearSVC`
- Wrapped with `CalibratedClassifierCV` for probability estimates

### Web Interface

| Route         | Description                            |
| ------------- | -------------------------------------- |
| `/`           | Home page - CSV preview                |
| `/show_all`   | Display entire dataset                 |
| `/preprocess` | Run preprocessing + classification     |
| `/comments`   | Per-comment Naive Bayes classification |
| `/svm`        | Per-comment SVM classification         |

## Data Flow

```
CSV File (dataNew.csv)
        │
        ▼
┌───────────────────┐
│ Read comments col │  (comments separated by "|")
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   clean_text()    │  Remove noise
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ classify_sentiment│  → positive/negative/neutral
│ classify_spam     │  → spam/not_spam
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Paginated HTML    │  Bootstrap-styled tables
└───────────────────┘
```

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Usage

```bash
python main.py
# Open http://127.0.0.1:5500 in your browser
```

## Notes

- On first run, NLTK will automatically download `punkt` and `stopwords` datasets
- Default data source is `dataNew.csv` in the same directory
- Comments separated by `|` are exploded into individual rows for per-comment analysis
- Classifiers are trained on synthetic data at startup
