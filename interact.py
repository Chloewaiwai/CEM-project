from src.models.CEM.model import CEM 
from src.utils import config
from src.utils.data.loader import prepare_data_seq
from src.utils.data.loader import Dataset
from src.utils.data.loader import encode_ctx
from src.utils.comet import Comet
import torch.utils.data as data
import torch
import torch
import logging
from src.utils import config
import torch.utils.data as data
from src.utils.constants import EMO_MAP as emo_map
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.utils.constants import EMOJI_MAP as emoji_map
import random




class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = emo_map
        self.analyzer = SentimentIntensityAnalyzer()

    def __len__(self):
        return len(self.data["context"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"]
        item["emotion_context"] = self.data["emotion_context"]
        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][0])
        )
        
        item["context"], item["context_mask"] = self.preprocess(item["context_text"])
        item["emotion_context"],item["emotion_context_mask"] = self.preprocess(item["emotion_context"])

        item["cs_text"] = self.data["utt_cs"]
        item["x_intent_txt"] = item["cs_text"][0]
        item["x_need_txt"] = item["cs_text"][1]
        item["x_want_txt"] = item["cs_text"][2]
        item["x_effect_txt"] = item["cs_text"][3]
        item["x_react_txt"] = item["cs_text"][4]

        item["x_intent"] = self.preprocess(item["x_intent_txt"], cs=True)
        item["x_need"] = self.preprocess(item["x_need_txt"], cs=True)
        item["x_want"] = self.preprocess(item["x_want_txt"], cs=True)
        item["x_effect"] = self.preprocess(item["x_effect_txt"], cs=True)
        item["x_react"] = self.preprocess(item["x_react_txt"], cs="react")
        return item

    def preprocess(self, arr, anw=False, cs=None, emo=False):
        
        """Converts words to ids."""
        if anw:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in arr
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)
        elif cs:
            sequence = [config.CLS_idx] if cs != "react" else []
            for sent in arr:
                sequence += [
                    self.vocab.word2index[word]
                    for word in sent
                    if word in self.vocab.word2index and word not in ["to", "none"]
                ]

            return torch.LongTensor(sequence)
        elif emo:
            x_emo = [config.CLS_idx]
            x_emo_mask = [config.CLS_idx]
            for i, ew in enumerate(arr):
                x_emo += [
                    self.vocab.word2index[ew]
                    if ew in self.vocab.word2index
                    else config.UNK_idx
                ]
                x_emo_mask += [self.vocab.word2index["CLS"]]

            assert len(x_emo) == len(x_emo_mask)
            return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)

        else:
            x_dial = [config.CLS_idx]
            x_mask = [config.CLS_idx]
            for i, sentence in enumerate(arr):
                x_dial += [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]
                spk = (
                    self.vocab.word2index["USR"]
                    if i % 2 == 0
                    else self.vocab.word2index["SYS"]
                )
                x_mask += [spk for _ in range(len(sentence))]
            assert len(x_dial) == len(x_mask)
            return torch.LongTensor(x_dial), torch.LongTensor(x_mask)

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    ## input
    input_batch, input_lengths = merge(item_info["context"])
    mask_input, mask_input_lengths = merge(item_info["context_mask"])
    emotion_batch, emotion_lengths = merge(item_info["emotion_context"])

    input_batch = input_batch.to(config.device)
    mask_input = mask_input.to(config.device)

    d = {}
    d["input_batch"] = input_batch
    d["input_lengths"] = torch.LongTensor(input_lengths)
    d["mask_input"] = mask_input
    d["emotion_context_batch"] = emotion_batch.to(config.device)

    ##text
    d["input_txt"] = item_info["context_text"]

    relations = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
    for r in relations:
        pad_batch, _ = merge(item_info[r])
        pad_batch = pad_batch.to(config.device)
        d[r] = pad_batch
        d[f"{r}_txt"] = item_info[f"{r}_txt"]
    return d

train_set, dev_set, test_set, vocab, dec_num = prepare_data_seq(
    batch_size=config.batch_size
)
model = CEM(
            vocab,
            decoder_number=32,
            is_eval=True,
            model_file_path='save\CEM_19999_41.8034',
        )
model.to(config.device)
model.eval()
model.is_eval = True
relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
comet = Comet("data\comet-atomic_2020_BART", config.device)
logging.info("Model is built.")

def get_emoji(emotions):
    #emoji= emoji_map.get(emotion_word)
    print(emotions)
    random.shuffle(emotions)
    print(emotions)
    for emotion in emotions:
        if emotion[0] in emoji_map:
            emoji= emoji_map.get(emotion[0])
            return emoji,emotion[0]
        

def get_response(msg):
   
    #sentence="I am going to meet my boyfriend!"
    data_dict = {
            "context": [],
            "emotion_context": [],
            "utt_cs": [],
        }

    encode_ctx(msg,data_dict,comet)
    dataset_input = Dataset(data_dict, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_input, batch_size=1, shuffle=False, collate_fn=collate_fn
    )
    return CEM.chatmodel(model,data_loader_tst,max_dec_step=50)

if __name__ == "__main__":
    print("==================================")    
    print("TEDDY: Hey there! How's you going?")
    while True:
        sentence=input("You: ")
        if sentence == "bye" or sentence == "":
            print("Bot: Bye!")
            print("==================================")
            break

        reply=get_response(sentence)
        
        print("emotion", reply[0])
        emoji=get_emoji(reply[1])
        print("TEDDY:", reply[0][0],emoji)