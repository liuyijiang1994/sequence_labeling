import torch
import numpy as np
import Constants
import gensim
import random

out_pkl = Constants.triger_fusion_train
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
    train_ratio = 0.7
    min_word = 1
    emb_dim = 300

    word2idx = Constants.word_2_id

    stcs = []
    tags = []

    with open('./data/fusion_data/sogo_seed_result_cleaned_token_cleaned.txt', 'r') as f:
        for line in f:
            i = line.strip().split('\t')
            text, tag = make_text_tag(i[2], i[0], i[1], i[3])
            stcs.append(text)
            tags.append(tag)

    # build vocabulary
    full_vocab = set(w for sent in stcs for w in sent)
    word_count = {w: 0 for w in full_vocab}

    # 统计字出现的次数
    for stc in stcs:
        for word in stc:
            word_count[word] += 1

    ignored_word_count = 0
    full_vocab = []
    for word, count in word_count.items():
        if word not in word2idx:
            if count >= min_word:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    for word in word2idx.keys():
        full_vocab.append(word)

    # 根据词典制作embedding层的权重
    print('[Info] 加载词向量')
    # glove = vocab.GloVe(name='6B', dim=300,
    #                     cache='/home/spman_x/liu_codes/tmp/pycharm_project_662/s7_relation/s10/.vector_cache')
    glove = gensim.models.KeyedVectors.load_word2vec_format('/home/spman_x/liu_codes/embed/sgns.sogounews.bigram-char')

    print('[Info] 加载词向量完毕')

    matrix_len = len(word2idx)
    weights_matrix = np.zeros((matrix_len, emb_dim))

    for word, i in word2idx.items():
        try:
            weights_matrix[i] = glove[word]
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word))
    print("[Info] Ignored word count = {}".format(ignored_word_count))

    split_ix = int(train_ratio * len(stcs))
    stcs = convert_instance_to_idx_seq(stcs, word2idx)

    c = list(zip(stcs, tags))
    random.shuffle(c)
    stcs[:], tags[:] = zip(*c)

    data = {
        'word2idx': word2idx,
        'tag2idx': Constants.tag_2_id,
        'vocab': full_vocab,
        'embedding_weight': weights_matrix,
        'train': {
            'text': stcs[:split_ix],
            'tag': tags[:split_ix]
        },
        'valid': {
            'text': stcs[split_ix:],
            'tag': tags[split_ix:]
        }
    }

    sample_data = {
        'word2idx': word2idx,
        'tag2idx': Constants.tag_2_id,
        'vocab': full_vocab,
        'embedding_weight': weights_matrix,
        'train': {
            'text': stcs[:100],
            'tag': tags[:100]
        },
        'valid': {
            'text': stcs[100:200],
            'tag': tags[100:200]
        }
    }

    torch.save(data, out_pkl)
    torch.save(sample_data, out_sample_pkl)
    #
    # with open('./data/clean_data.txt', 'w') as f:
    #     for i in range(5):
    #         for j in range(len(stcs[i])):
    #             print(stcs[i][j], tags[i][j])
    #         print()
    print('[Info] Finish.')

    print('Samples:')
    for i in range(5):
        print(''.join([full_vocab[idx] for idx in stcs[i]]))
        print(tags[i])
        print('-' * 10)
