"""
FAQ Chatbot core engine.

- Loads FAQ data (question/answer pairs) from a JSON file.
- Preprocesses text using NLTK (tokenization, stopword removal, lemmatization).
- Matches a user's question against the FAQ set using TF-IDF + cosine similarity.
- Returns the best matching answer (with a confidence score and a fallback
  response when nothing matches well enough).
"""

import json
import string
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_NLTK_PACKAGES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


def ensure_nltk_data():
    """Download required NLTK corpora/models if they aren't already present."""
    for path, package in REQUIRED_NLTK_PACKAGES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))
_punct_table = str.maketrans("", "", string.punctuation)


def preprocess(text: str) -> str:
    """
    Clean and normalize text:
    1. Lowercase
    2. Remove punctuation
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    Returns a single cleaned string ready for vectorization.
    """
    text = text.lower().translate(_punct_table)
    tokens = word_tokenize(text)
    cleaned = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in _stop_words
    ]
    return " ".join(cleaned)


class FAQChatbot:
    """A simple retrieval-based FAQ chatbot using TF-IDF + cosine similarity."""

    def __init__(self, faq_path: str = None, similarity_threshold: float = 0.25):
        if faq_path is None:
            faq_path = os.path.join(os.path.dirname(__file__), "data", "faqs.json")

        self.similarity_threshold = similarity_threshold
        self.faqs = self._load_faqs(faq_path)

        self.questions = [item["question"] for item in self.faqs]
        self.answers = [item["answer"] for item in self.faqs]

        # Preprocess all FAQ questions once, up front.
        self.processed_questions = [preprocess(q) for q in self.questions]

        # Fit TF-IDF vectorizer on the FAQ question corpus.
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)

    @staticmethod
    def _load_faqs(path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError("FAQ file must contain a non-empty list of Q/A objects.")
        return data

    def get_response(self, user_query: str):
        """
        Given a raw user query, return the best-matching FAQ answer.

        Returns a dict:
            {
                "answer": str,
                "matched_question": str or None,
                "score": float,
            }
        """
        if not user_query or not user_query.strip():
            return {
                "answer": "Please type a question and I'll do my best to help!",
                "matched_question": None,
                "score": 0.0,
            }

        cleaned_query = preprocess(user_query)

        if not cleaned_query:
            return {
                "answer": "I couldn't quite understand that. Could you rephrase your question?",
                "matched_question": None,
                "score": 0.0,
            }

        query_vec = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])

        if best_score < self.similarity_threshold:
            return {
                "answer": (
                    "I'm not sure I have an answer for that. "
                    "Could you try rephrasing, or contact support@example.com for help?"
                ),
                "matched_question": None,
                "score": best_score,
            }

        return {
            "answer": self.answers[best_idx],
            "matched_question": self.questions[best_idx],
            "score": best_score,
        }

    def top_matches(self, user_query: str, k: int = 3):
        """Return the top-k matching FAQs for debugging/inspection purposes."""
        cleaned_query = preprocess(user_query)
        query_vec = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked = sorted(
            zip(self.questions, self.answers, similarities),
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:k]


if __name__ == "__main__":
    bot = FAQChatbot()
    print("FAQChatbot loaded with", len(bot.questions), "FAQs. Try bot.get_response('...')")
