from flask import Flask,request, jsonify
from flask_cors import CORS
from interact import get_response,get_emoji


app = Flask(__name__)
CORS(app)

@app.route("/chatroom/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    emoji,emotion=get_emoji(response[1])
    response=response[0][0]+emoji
    message = {"teddyResponse" : response, "receivedEmotion" :emotion}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
