import xlrd
import torch
import numpy as np
import Constants
import stringUtil
import gensim
import utils

out_pkl = Constants.triger_fusion_test
out_sample_pkl = Constants.sample_data_path

tag_2_id = Constants.tag_2_id


def make_tag(t1, text, tag, b_tag=1):
    begin = 0
    while text.find(t1, begin) >= 0:
        tix = text.find(t1, begin)
        tag[tix] = b_tag
        for i in range(len(t1) - 1):
            tag[tix + 1 + i] = b_tag + 1
        begin = text.find(t1, begin) + len(t1)

    return tag


def make_text_tag(f1, t2, t3, text):
    tag = [Constants.tag_2_id['O']] * len(text)
    tag = make_tag(f1, text, tag, b_tag=4)
    # tag = make_tag(t2, text, tag, b_tag=6)
    # tag = make_tag(t3, text, tag, b_tag=6)
    text = list(text)

    assert len(text) == len(tag)
    return text, tag


# 将文本转换为index，找不到则为unk
def convert_instance_to_idx_seq(stcs, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in stcs]


if __name__ == '__main__':
    train_ratio = 1
    min_word = 1
    emb_dim = 300

    word2idx = Constants.word_2_id

    stcs = []
    tags = []
    with open('./data/fusion_data/fusion_test_cleaned.txt', 'r') as f:
        # with open('/home/spman_x/disk/liu_codes/sogo_seed_result_cleaned.txt', 'r') as f:
        for line in f:
            i = line.strip().split('\t')
            text, tag = make_text_tag(i[2], i[0], i[1], i[3])
            stcs.append(text)
            tags.append(tag)

    data = torch.load(Constants.triger_fusion_train)
    stcs = convert_instance_to_idx_seq(stcs, data['word2idx'])

    test_data = {
        'word2idx': data['word2idx'],
        'tag2idx': data['tag2idx'],
        'vocab': data['vocab'],
        'embedding_weight': data['embedding_weight'],
        'test': {
            'text': stcs,
            'tag': tags
        }
    }

    torch.save(test_data, Constants.fusion_test)
    #
    # with open('./data/clean_data.txt', 'w') as f:
    #     for i in range(5):
    #         for j in range(len(stcs[i])):
    #             print(stcs[i][j], tags[i][j])
    #         print()
    print('[Info] Finish.')
    for i in range(5):
        pstc = [data['vocab'][idx] for idx in stcs[i]]
        ptag = tags[i]
        for j in range(len(pstc)):
            print(pstc[j], ptag[j])
        print()
