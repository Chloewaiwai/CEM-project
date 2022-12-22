from src.models.CEM.model import CEM 
from src.utils import config
from src.utils.data.loader import prepare_data_seq
from src.utils.data.loader import collate_fn
from src.utils.data.loader import Dataset
import torch
import nltk

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


sentence=tokenize("This weekend is fun")
print(sentence)

train_set, dev_set, test_set, vocab, dec_num = prepare_data_seq(
        batch_size=config.batch_size
    )

def prepro(arr):
    x_dial = [config.CLS_idx]
    x_mask = [config.CLS_idx]
    for i, sentence in enumerate(arr):
        x_dial += [
            vocab.word2index[word]
            if word in vocab.word2index
            else config.UNK_idx
            for word in sentence
        ]
        print(x_dial)
        spk = (
            vocab.word2index["USR"]
            if i % 2 == 0
            else vocab.word2index["SYS"]
        )
        x_mask += [spk for _ in range(len(sentence))]
    assert len(x_dial) == len(x_mask)

    return torch.LongTensor(x_dial), torch.LongTensor(x_mask)
context, context_mask = prepro([sentence])
print(sentence)
print(context)

input_batch, input_lengths = collate_fn.merge(item_info["context"])


model = CEM(
            vocab,
            decoder_number=dec_num,
            is_eval=True,
            model_file_path='save\CEM_19999_41.8034',
        )
model.eval()
sys_input={"input_batch": "Hello, nice to meet you!", "input_lengths": len(sentence),"input_txt":sentence}
reply=CEM.chatmodel(model, sys_input,max_dec_step=50)
print(reply)