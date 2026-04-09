import csv
import io
import json
import logging
import os
import pickle
import re
import string
from datetime import datetime
from pathlib import Path

import nltk
from flask import Flask, jsonify, render_template, request, send_file
from nltk.corpus import stopwords

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HISTORY_PATH = DATA_DIR / "history.json"
MODEL_PATH = BASE_DIR / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

DATA_DIR.mkdir(parents=True, exist_ok=True)
if not HISTORY_PATH.exists():
    HISTORY_PATH.write_text("[]", encoding="utf-8")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)
with VECTORIZER_PATH.open("rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def load_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        logger.warning("NLTK stopwords corpus not found. Using built-in fallback list.")
        return {
            "a", "an", "the", "is", "are", "am", "and", "or", "to", "of", "in",
            "for", "with", "on", "at", "by", "this", "that", "it", "as", "be",
            "from", "was", "were", "will", "can", "you", "your", "i", "we", "they",
        }


STOP_WORDS = load_stopwords()


def load_history():
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to load history: %s", exc)
        return []


def save_history(history):
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def preprocess(text):
    text = (text or "").lower()
    text = "".join(char for char in text if char not in string.punctuation)
    words = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(words)


def model_probability(features):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        return float(probs[1])
    # Fallback for classifiers without predict_proba
    pred = int(model.predict(features)[0])
    return 0.85 if pred == 1 else 0.15


def detect_suspicious_terms(processed_text, vector_features):
    if not processed_text.strip():
        return []

    tokens = processed_text.split()
    unique_tokens = sorted(set(tokens))

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        token_scores = {}
        for token in unique_tokens:
            if token in vectorizer.vocabulary_:
                idx = vectorizer.vocabulary_[token]
                tfidf_val = vector_features[0, idx]
                token_scores[token] = float(tfidf_val) * float(importances[idx])
        ranked = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
        return [tok for tok, _ in ranked[:6]]

    # Fallback heuristic by keyword pattern
    spam_keywords = {
        "free",
        "win",
        "winner",
        "prize",
        "urgent",
        "click",
        "offer",
        "limited",
        "cash",
        "claim",
        "bonus",
        "lottery",
    }
    return [tok for tok in unique_tokens if tok in spam_keywords][:6]


def safe_tone_suggestion(text, is_spam):
    if is_spam:
        return (
            "Consider removing urgency words, suspicious links, and exaggerated promises. "
            "Use a clear, factual tone and include transparent sender context."
        )
    return (
        "This message looks safe. You can improve clarity by adding a concise subject, "
        "keeping sentences short, and ending with a polite call-to-action."
    )


def try_gemini_prompt(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(prompt)
        return (response.text or "").strip()
    except Exception as exc:
        logger.warning("Gemini request failed: %s", exc)
        return None


def ai_rewrite_message(text, is_spam):
    prompt = (
        "Rewrite this text in a safer and trustworthy way. "
        if is_spam
        else "Improve this text to sound more polished and professional. "
    )
    prompt += f'\n\nText:\n"""\n{text}\n"""'
    output = try_gemini_prompt(prompt)
    return output if output else safe_tone_suggestion(text, is_spam)


def ai_explain_message(text, label, confidence, suspicious_terms):
    prompt = (
        "Explain in simple words why a spam classifier predicted this message.\n"
        f"Label: {label}\n"
        f"Confidence: {confidence:.4f}\n"
        f"Suspicious terms: {', '.join(suspicious_terms) if suspicious_terms else 'None'}\n"
        f'Message: """{text}"""\n'
        "Keep it short and helpful for a non-technical user."
    )
    output = try_gemini_prompt(prompt)
    if output:
        return output
    if label == "spam":
        return (
            "It was flagged as spam because it contains language patterns often used in "
            "promotional or deceptive messages, such as urgency and persuasive keywords."
        )
    return "It was marked as non-spam because the text appears conversational and low-risk."


def highlight_text(original_text, suspicious_terms):
    if not suspicious_terms:
        return original_text

    pattern = r"\b(" + "|".join(re.escape(term) for term in suspicious_terms) + r")\b"
    return re.sub(
        pattern,
        lambda match: f"<mark class='suspicious'>{match.group(0)}</mark>",
        original_text,
        flags=re.IGNORECASE,
    )


def run_prediction(message):
    processed = preprocess(message)
    features = vectorizer.transform([processed])
    spam_probability = model_probability(features)
    label = "spam" if spam_probability >= 0.5 else "ham"
    confidence = spam_probability if label == "spam" else 1 - spam_probability
    suspicious_terms = detect_suspicious_terms(processed, features) if label == "spam" else []
    suggestion = ai_rewrite_message(message, label == "spam")
    explanation = ai_explain_message(message, label, confidence, suspicious_terms)
    highlighted = highlight_text(message, suspicious_terms)

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "spam_probability": round(float(spam_probability), 4),
        "suspicious_terms": suspicious_terms,
        "highlighted_text": highlighted,
        "suggestion": suggestion,
        "explanation": explanation,
        "processed_text": processed,
    }


def append_history(message, prediction):
    history = load_history()
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "message": message,
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "spam_probability": prediction["spam_probability"],
    }
    history.append(record)
    # Keep last 1000 records for lightweight persistence.
    history = history[-1000:]
    save_history(history)
    return record


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "spam-detection-api"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message is required."}), 400

    prediction = run_prediction(message)
    history_record = append_history(message, prediction)
    return jsonify({"prediction": prediction, "history_record": history_record})


@app.route("/analyze", methods=["GET"])
def analyze():
    history = load_history()
    total = len(history)
    spam_count = sum(1 for item in history if item.get("label") == "spam")
    ham_count = total - spam_count
    spam_ratio = round((spam_count / total) * 100, 2) if total else 0.0
    ham_ratio = round((ham_count / total) * 100, 2) if total else 0.0
    avg_confidence = round(
        sum(float(item.get("confidence", 0)) for item in history) / total, 4
    ) if total else 0.0

    model_accuracy = float(os.getenv("MODEL_ACCURACY", "0.96"))
    return jsonify(
        {
            "total_messages_checked": total,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "spam_ratio": spam_ratio,
            "ham_ratio": ham_ratio,
            "average_confidence": avg_confidence,
            "model_accuracy": model_accuracy,
            "history_preview": history[-10:],
        }
    )


@app.route("/history", methods=["GET"])
def history():
    return jsonify({"history": load_history()[-100:]})


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    context = payload.get("context") or {}
    if not question:
        return jsonify({"error": "Question is required."}), 400

    prompt = (
        "You are an assistant for a spam detection app. "
        "Answer clearly and simply.\n"
        f"Question: {question}\n"
        f"Context JSON: {json.dumps(context)}"
    )
    answer = try_gemini_prompt(prompt)
    if not answer:
        answer = (
            "Spam messages usually use urgency, pressure, suspicious links, or unrealistic offers. "
            "If you share the message text, I can explain the signal in more detail."
        )
    return jsonify({"answer": answer})


@app.route("/bulk_predict", methods=["POST"])
def bulk_predict():
    if "file" not in request.files:
        return jsonify({"error": "CSV file is required."}), 400
    csv_file = request.files["file"]
    if not csv_file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a valid CSV file."}), 400

    content = csv_file.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        return jsonify({"error": "CSV appears empty."}), 400

    message_col = None
    for candidate in ("message", "text", "content"):
        if candidate in reader.fieldnames:
            message_col = candidate
            break
    if not message_col:
        return jsonify(
            {"error": "CSV must include one of these columns: message, text, content."}
        ), 400

    rows = []
    output_buffer = io.StringIO()
    writer = csv.DictWriter(
        output_buffer,
        fieldnames=["message", "label", "confidence", "spam_probability", "suggestion"],
    )
    writer.writeheader()

    for record in reader:
        message = (record.get(message_col) or "").strip()
        if not message:
            continue
        prediction = run_prediction(message)
        append_history(message, prediction)
        row = {
            "message": message,
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "spam_probability": prediction["spam_probability"],
            "suggestion": prediction["suggestion"],
        }
        rows.append(row)
        writer.writerow(row)

    if not rows:
        return jsonify({"error": "No valid message rows found in CSV."}), 400

    csv_result = output_buffer.getvalue().encode("utf-8")
    return send_file(
        io.BytesIO(csv_result),
        mimetype="text/csv",
        as_attachment=True,
        download_name="bulk_predictions.csv",
    )


@app.errorhandler(404)
def not_found(_err):
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def internal_error(err):
    logger.exception("Unhandled server error: %s", err)
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)