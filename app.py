from flask import Flask,request, jsonify
from flask_cors import CORS
from interact import get_response,get_emoji,check_positive
from blenderbot import Blenderbot
from src.utils.constants import Question_word as question_word
from src.utils.constants import Greetings as greeting_word


app = Flask(__name__)
CORS(app)

@app.route("/chatroom/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    words = text.lower().split()
    is_greeting = False
    for greeting in greeting_word:
        print(greeting,text)
        if greeting in text:
            is_greeting = True
            break

    if any(word in question_word for word in words) or "?" in text or is_greeting:
        blenderbot = Blenderbot("facebook/blenderbot_small-90M")
        response = blenderbot.generate(text)
        emotion = "blender"
        print("blender")
    else:
        response = get_response(text)
        emoji,emotion=get_emoji(response[1])
        response=response[0][0]+emoji
        print("cem")
    
    message = {"teddyResponse" : response, "receivedEmotion" :emotion}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
