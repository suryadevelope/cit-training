from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question")

    # Simple logic
    if "ai" in question.lower():
        answer = "AI is Artificial Intelligence"
    else:
        answer = "Unknown question"

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
