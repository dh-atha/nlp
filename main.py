from flask import Flask, render_template_string, request, redirect, url_for
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure necessary NLTK data is available
try:
		nltk.data.find('tokenizers/punkt')
except LookupError:
		nltk.download('punkt', quiet=True)

try:
		nltk.data.find('corpora/stopwords')
except LookupError:
		nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Fallback Indonesian stopwords if not available via NLTK
DEFAULT_IND_STOPWORDS = {
		'yang','dan','di','ke','dari','ini','itu','untuk','pada','dengan','atau','sebagai',
		'adalah','saya','kamu','kita','dia','ya','nya','dengan','oleh','akan','tidak','tida',
		'ada','ini','itu','yg','nih','ga','gak','yg','supaya','biar','masa'
}

def get_stopwords(lang='indonesian'):
		try:
				sw = set(stopwords.words(lang))
				if len(sw) > 10:
						return sw
		except Exception:
				pass
		return DEFAULT_IND_STOPWORDS

STOPWORDS = get_stopwords('indonesian')

EMOJI_PATTERN = re.compile("["
													 "\U0001F600-\U0001F64F"  # emoticons
													 "\U0001F300-\U0001F5FF"  # symbols & pictographs
													 "\U0001F680-\U0001F6FF"  # transport & map symbols
													 "\U0001F1E0-\U0001F1FF"  # flags (iOS)
													 "\u2600-\u26FF\u2700-\u27BF]+", flags=re.UNICODE)

def clean_text(text: str) -> str:
		if pd.isna(text):
				return ""
		# convert to str
		text = str(text)
		# remove urls
		text = re.sub(r'http\S+|www\.\S+', ' ', text)
		# remove emojis
		text = EMOJI_PATTERN.sub(' ', text)
		# remove mentions and hashtags
		text = re.sub(r'[@#]\w+', ' ', text)
		# remove non-letter characters (keep unicode letters)
		text = re.sub(r'[^\w\s]', ' ', text)
		# remove digits
		text = re.sub(r'\d+', ' ', text)
		# collapse whitespace
		text = re.sub(r'\s+', ' ', text).strip()
		return text.lower()

def tokenize_and_filter(text: str):
		if not text:
				return []
		try:
				tokens = word_tokenize(text)
		except Exception:
				# fallback simple split
				tokens = text.split()
		# normalize tokens
		tokens = [t.lower() for t in tokens if len(t) > 1]
		# remove stopwords
		tokens = [t for t in tokens if t not in STOPWORDS]
		return tokens

INDEX_HTML = '''
<!doctype html>
<html lang="en">
	<head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<title>NLP Preprocessing</title>
		<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
	</head>
	<body class="bg-light">
		<div class="container py-4">
			<h1 class="mb-3">NLP Preprocessing - Simple Web UI</h1>
			<p class="text-muted">Pilih kolom teks dari dataset `dataNew.csv` untuk memproses (comments, caption, atau kolom lain).</p>

			{% if error %}
				<div class="alert alert-danger">{{ error }}</div>
			{% endif %}

			<form method="post" action="/preprocess" class="row g-3 mb-4">
				<div class="col-md-4">
					<label class="form-label">CSV Path</label>
					<input name="csv_path" class="form-control" value="dataNew.csv">
				</div>
				<div class="col-md-4">
					<label class="form-label">Text Column</label>
					<input name="text_col" class="form-control" value="comments">
				</div>
				<div class="col-md-2">
					<label class="form-label">Rows to show</label>
					<input name="n_rows" type="number" min="1" max="1000" class="form-control" value="20">
				</div>
				<div class="col-md-2 align-self-end">
					<button class="btn btn-primary w-100">Preprocess</button>
				</div>
			</form>

			{% if table_html %}
				<h5>Data Preview (original)</h5>
				<div class="table-responsive">{{ table_html | safe }}</div>
			{% endif %}

			<hr>
			<footer class="text-muted small">If NLTK data is missing, server will attempt to download it once.</footer>
		</div>
	</body>
</html>
'''

RESULT_HTML = '''
<!doctype html>
<html lang="en">
	<head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<title>Preprocessing Result</title>
		<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
	</head>
	<body class="bg-light">
		<div class="container py-4">
			<h1 class="mb-3">Preprocessing Result</h1>
			<a href="/" class="btn btn-secondary mb-3">&larr; Back</a>

			<h5>Processed (first {{ n_rows }} rows)</h5>
			<div class="table-responsive">{{ result_table | safe }}</div>

			<hr>
			<h6>Notes</h6>
			<ul>
				<li>Cleaning steps: remove URLs, emojis, mentions, punctuation, digits.</li>
				<li>Tokenization: NLTK word_tokenize (fallback to split on failure).</li>
				<li>Stopwords: NLTK Indonesian stopwords if available, otherwise fallback list.</li>
			</ul>
		</div>
	</body>
</html>
'''


@app.route('/', methods=['GET'])
def index():
		csv_path = os.path.join(os.getcwd(), 'dataNew.csv')
		table_html = None
		error = None
		if os.path.exists(csv_path):
				try:
						df = pd.read_csv(csv_path, nrows=20)
						table_html = df.head(20).to_html(classes='table table-sm table-striped', index=False, escape=True)
				except Exception as e:
						error = f"Error reading CSV: {e}"
		else:
				error = f"File not found: {csv_path}"

		return render_template_string(INDEX_HTML, table_html=table_html, error=error)


@app.route('/preprocess', methods=['POST'])
def preprocess():
		csv_path = request.form.get('csv_path', 'dataNew.csv')
		text_col = request.form.get('text_col', 'comments')
		try:
				n_rows = int(request.form.get('n_rows', 20))
		except Exception:
				n_rows = 20

		if not os.path.exists(csv_path):
				return render_template_string(INDEX_HTML, table_html=None, error=f"CSV not found: {csv_path}")

		try:
				df = pd.read_csv(csv_path)
		except Exception as e:
				return render_template_string(INDEX_HTML, table_html=None, error=f"Error reading CSV: {e}")

		if text_col not in df.columns:
				return render_template_string(INDEX_HTML, table_html=None, error=f"Column '{text_col}' not found in CSV. Available columns: {', '.join(df.columns)}")

		# show original preview
		preview_html = df.head(n_rows).to_html(classes='table table-sm table-striped', index=False, escape=True)

		# Apply preprocessing
		texts = df[text_col].astype(str).fillna('')

		def process_segments(text):
			# split on pipe | to get segments, strip spaces, ignore empty
			segments = [s.strip() for s in str(text).split('|')]
			segments = [s for s in segments if s]
			if not segments:
				return ('', '')
			cleaned_segs = [clean_text(s) for s in segments]
			token_segs = [tokenize_and_filter(s) for s in cleaned_segs]
			# join cleaned segments with ' | ' and tokens per segment joined by spaces
			cleaned_join = ' | '.join(cleaned_segs)
			tokens_join = ' | '.join([' '.join(t) for t in token_segs])
			return (cleaned_join, tokens_join)

		processed = texts.map(process_segments)
		cleaned = processed.map(lambda x: x[0])
		tokens = processed.map(lambda x: x[1])

		result_df = pd.DataFrame({
			'original': texts.head(n_rows),
			'cleaned_segments': cleaned.head(n_rows),
			'tokens_per_segment': tokens.head(n_rows)
		})

		result_table = result_df.to_html(classes='table table-sm table-striped', index=False, escape=True)

		return render_template_string(RESULT_HTML, result_table=result_table, n_rows=n_rows)


if __name__ == '__main__':
		# Use port 5000 by default. Run with `python main.py`.
		app.run(host='0.0.0.0', port=5500, debug=True)

