from flask import Flask, render_template, request, jsonify
from app.rag_system import ask_question

app = Flask(__name__, template_folder="app/templates",
            static_folder="app/static")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    print("ASK ROUTE WAS CALLED")
    print("QUESTION:", question)

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        result = ask_question(question)
        print("RESULT:", result)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Something went wrong: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
