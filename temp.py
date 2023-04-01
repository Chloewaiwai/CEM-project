from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.utils import config

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot_small-90M")



class Blenderbot:
    def __init__(self,model_path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.batch_size = 1

    def generate(self, input):
            inputs = tokenizer([input], return_tensors="pt")
        
            reply_ids = model.generate(**inputs)

            reply = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            return reply
