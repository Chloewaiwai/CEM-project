from src.models.CEM.model import CEM 
from src.utils import config
from src.utils.data.loader import prepare_data_seq
from src.utils.data.loader import Dataset
from src.utils.data.loader import Lang
from src.utils.comet import Comet
import torch
import nltk

'''read_files(vocab=Lang(
                {
                    config.UNK_idx: "UNK",
                    config.PAD_idx: "PAD",
                    config.EOS_idx: "EOS",
                    config.SOS_idx: "SOS",
                    config.USR_idx: "USR",
                    config.SYS_idx: "SYS",
                    config.CLS_idx: "CLS",
                }
            )'''


def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


sentence=[tokenize("This weekend is fun")]
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
        spk = (
            vocab.word2index["USR"]
            if i % 2 == 0
            else vocab.word2index["SYS"]
        )
        x_mask += [spk for _ in range(len(sentence))]
    assert len(x_dial) == len(x_mask)

    return [torch.LongTensor(x_dial)], [torch.LongTensor(x_mask)]

def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, torch.LongTensor(lengths)

context, context_mask = prepro(sentence)
input_batch, input_lengths = merge(context)
mask_input, mask_input_lengths = merge(context_mask)
print(context_mask,mask_input)

input_batch = input_batch.to(config.device)
mask_input = input_batch.to(config.device)

batch={'input_batch':input_batch,'input_lengths':input_lengths,'input_txt':sentence,'mask_input':mask_input}
print(batch)


model = CEM(
            vocab,
            decoder_number=dec_num,
            is_eval=True,
            model_file_path='save\CEM_19999_41.8034',
        )
model.eval()
reply=CEM.chatmodel(model, batch,max_dec_step=50)
print(reply)