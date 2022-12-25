from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import interact

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    botname = request.get_json().get("bot")
    response = interact(text,botname)
    message = {"answer" : response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
