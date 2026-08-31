# FAQ Chatbot

A retrieval-based FAQ chatbot in Python. It matches a user's question against
a stored set of FAQs using NLP preprocessing (NLTK) and TF-IDF + cosine
similarity, then returns the best matching answer. Includes both a
command-line interface and a simple Flask web chat UI.

## Features

- 📚 FAQ dataset stored as JSON (`data/faqs.json`) — easy to edit/extend
- 🧹 Text preprocessing with NLTK: lowercasing, punctuation removal,
  tokenization, stopword removal, lemmatization
- 🔍 Matching via `TfidfVectorizer` + `cosine_similarity` (scikit-learn)
- ⚠️ Confidence threshold with a graceful fallback response for unmatched questions
- 💻 Command-line chat (`cli.py`)
- 🌐 Web chat UI (`app.py` + Flask + HTML/CSS/JS)

## Project Structure

```
faq-chatbot/
├── app.py                 # Flask web app
├── cli.py                 # Command-line chatbot
├── chatbot.py             # Core NLP + matching engine
├── requirements.txt
├── data/
│   └── faqs.json          # FAQ question/answer pairs
├── templates/
│   └── index.html         # Chat UI page
└── static/
    └── style.css           # Chat UI styling
```

## Setup

1. Clone the repo and enter the folder:
   ```bash
   git clone <your-repo-url>
   cd faq-chatbot
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Required NLTK data (`punkt`, `stopwords`, `wordnet`, `omw-1.4`) is
   downloaded automatically the first time you run the app.

## Usage

### Command-line chatbot

```bash
python cli.py
```

### Web chat UI

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Customizing the FAQs

Edit `data/faqs.json` and add entries in this format:

```json
{
  "question": "How do I reset my password?",
  "answer": "Go to the login page, click 'Forgot Password', and follow the instructions."
}
```

The vectorizer is refit automatically whenever `FAQChatbot()` is instantiated,
so no extra training step is required.

## How Matching Works

1. **Preprocessing** — user input and all stored FAQ questions are lowercased,
   stripped of punctuation, tokenized, filtered of stopwords, and lemmatized.
2. **Vectorization** — all FAQ questions are converted into TF-IDF vectors.
3. **Similarity** — the user's (also vectorized) question is compared against
   every FAQ vector using cosine similarity.
4. **Selection** — the FAQ with the highest similarity score is returned, as
   long as it clears a minimum confidence threshold (default `0.25`). Below
   that, the bot returns a fallback "I'm not sure" response instead of
   guessing.

## Extending This Project

- Swap TF-IDF for sentence embeddings (e.g. `sentence-transformers`) for
  semantic (not just keyword-based) matching.
- Add intent classification with a small ML classifier for multi-turn dialogs.
- Persist FAQs in a database instead of a JSON file.
- Add conversation logging/analytics to see which questions go unanswered.

## License

MIT — feel free to use and modify.
