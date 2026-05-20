from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from app.rag_system import ask_question, rebuild_vector_store

DATA_FOLDER = Path(__file__).resolve().parent / "data"
ALLOWED_EXTENSIONS = {".txt", ".md"}

app = Flask(__name__, template_folder="app/templates",
            static_folder="app/static")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    history = data.get("history", [])
    print("ASK ROUTE WAS CALLED")
    print("QUESTION:", question)

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        result = ask_question(question, history=history)
        print("RESULT:", result)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Something went wrong: {exc}"}), 500


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only .txt and .md files are supported."}), 400

    filename = secure_filename(file.filename)
    DATA_FOLDER.mkdir(exist_ok=True)
    file.save(DATA_FOLDER / filename)

    try:
        rebuild_vector_store()
        return jsonify({"message": f"'{filename}' uploaded and knowledge base rebuilt."})
    except Exception as exc:
        return jsonify({"error": f"File saved but rebuild failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
