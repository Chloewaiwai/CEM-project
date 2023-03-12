from flask import Flask,request, jsonify
from flask_cors import CORS
from interact import get_response
import random

app = Flask(__name__)
CORS(app)

@app.route("/chatroom/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    print("TEDDY:", response[0])
    random_item = random.choice(response[1])
    print("emotions",response[1],random_item[0])
    message = {"teddyResponse" : response[0], "receivedEmotion" : "\U0001F600"}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
