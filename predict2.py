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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    with open('../fusion_web/truesample.txt', 'r') as f:
        for line in f:
            text.append(line.strip())
    random.shuffle(text)

    return text


if __name__ == "__main__":
    with open('truesample.txt', 'w') as w:
        text_all_list = get_test_text()
        BATCH_SZ = 32
        batch = int(len(text_all_list) / 32) - 1
        print(len(text_all_list))
        print(batch)
        model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
        idx2tag = {v: k for k, v in tag_to_idx.items()}
        with torch.no_grad():
            for i in range(batch):
                text_list = text_all_list[i * 32:(i + 1) * 32]
                guess_tag_list = predict(model, word_to_idx, text_list)
                for text, tag in zip(text_list, guess_tag_list):
                    chunck_list = utils.bio_to_chuncks(tag, idx_to_tag)
                    fusion_set = set()
                    part_set = set()
                    for chunck in chunck_list:
                        if chunck.type == 'part':
                            part_set.add(text[chunck.b_idx + 1:chunck.b_idx + chunck.length + 1])
                        if chunck.type == 'fusion':
                            fusion_set.add(text[chunck.b_idx + 1:chunck.b_idx + chunck.length + 1])
                    if len(fusion_set) == 1 and len(part_set) == 2:
                        fusion_token = fusion_set.pop()
                        p1_token = part_set.pop()
                        p2_token = part_set.pop()

                        if len(fusion_token) == 2 and len(p1_token) == 2 and len(p2_token) == 2:

                            if (p1_token[0] not in fusion_token and p1_token[1] not in fusion_token) or (
                                    p2_token[0] not in fusion_token and p2_token[1] not in fusion_token):
                                continue
                            else:
                                w.write(text + '\n')
