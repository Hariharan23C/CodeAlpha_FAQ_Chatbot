"""
Flask web app for the FAQ chatbot.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify
from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")

    result = bot.get_response(user_message)

    return jsonify({
        "answer": result["answer"],
        "matched_question": result["matched_question"],
        "score": round(result["score"], 3),
    })


if __name__ == "__main__":
    app.run(debug=True)
