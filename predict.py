import configparser
from utils import *
from train import sort_by_seq_len
import utils
import torch
from model.model import lstm_crf
from mydataasets import collate_fn
import numpy as np

model_path = '/tmp/pycharm_project_229/sequence_labeling/result/Imi3gb/save/model_19.pt'
dict_path = Constants.fusion_train
cfg_path = './config/fusion_word2vec.cfg'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 将文本转换为index，找不到则为unk
def convert_instance_to_idx_seq(stcs, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in stcs]


def load_model():
    print(f'[INFO] loading model from {model_path}')
    print(f'[INFO] dict path : {dict_path}')
    dictdata = torch.load(dict_path)
    word_to_idx = dictdata['word2idx']
    tag_to_idx = dictdata['tag2idx']
    idx_to_tag = [tag for tag, _ in sorted(tag_to_idx.items(), key=lambda x: x[1])]
    conf = configparser.ConfigParser()
    conf.read(cfg_path)
    torch.manual_seed(conf.getint('model', 'MANUAL_SEED'))
    model = lstm_crf(len(word_to_idx), len(tag_to_idx), conf).to(device)
    print(model)
    model.eval()
    load_checkpoint(model_path, model)
    model.eval()
    print(f'[INFO] loading model finished{model_path}')
    return model, word_to_idx, tag_to_idx, idx_to_tag


def clean_pad(data):
    '''
    清除tensor中的pad并变为list[list]
    :param data:
    :return:
    '''
    seq_len = utils.get_seq_len_batch(data)
    data = data.tolist()
    clean_data = []
    for idx, sl in enumerate(seq_len):
        clean_data.append(data[idx][1:sl])
    return clean_data


def predict(model, word_to_idx, text):
    seq_text_id = convert_instance_to_idx_seq(text, word_to_idx)
    seq_text_id = collate_fn(seq_text_id).to(device)
    sorted_indices, desorted_indices = sort_by_seq_len(seq_text_id)
    seq_text_id = seq_text_id[sorted_indices]
    seq_tag_id = model.decode(seq_text_id)

    seq_tag_id = collate_fn(seq_tag_id)
    seq_tag_id = seq_tag_id[desorted_indices]
    seq_tag_id = clean_pad(seq_tag_id)

    return seq_tag_id


def convert_idx_to_word(seq_idx, vocab):
    return ''.join([vocab[idx] for idx in seq_idx])


def visual_chunck_list(chunck_list, sentence, vocab):
    visual_chuncks = []
    for chunck in chunck_list:
        chunck_name = convert_idx_to_word(sentence[chunck.b_idx:chunck.b_idx + chunck.length], vocab)
        visual_chuncks.append(chunck_name)
    return visual_chuncks


def get_test_text():
    text = []
    with open('test_samples.txt', 'r') as f:
        for line in f:
            text.append(line.strip())
    return text


if __name__ == "__main__":
    text_list = get_test_text()

    model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    idx2tag = {v: k for k, v in tag_to_idx.items()}
    with torch.no_grad():
        guess_tag_list = predict(model, word_to_idx, text_list)
    # for tag in guess_tag_list:
    #     print(len(tag))
    for text, tag in zip(text_list, guess_tag_list):
        print(text)
        # tag = [idx_to_tag[idx] for idx in tag]
        # print(tag)
        # print(idx_to_tag)
        chunck_list = utils.bio_to_chuncks(tag, idx_to_tag)
        for chunck in chunck_list:
            print(text[chunck.b_idx + 1:chunck.b_idx + chunck.length + 1], chunck.type)
        print('-' * 10)
