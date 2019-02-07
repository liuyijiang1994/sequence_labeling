from common import Constants
import codecs
import random
import numpy as np
import torchtext.vocab as vocab
import torch
import gensim

file_path = './data/dh_msra.txt'
train_ratio = 0.7
min_word = 1
emb_dim = 300
english_embed_cache_path = '/home/spman_x/liu_codes/tmp/pycharm_project_662/s7_relation/s10/.vector_cache'
chinese_embed_cache_path = '/home/spman_x/liu_codes/embed/sgns.sogounews.bigram-char'


# 将文本转换为index，找不到则为unk
def convert_instance_to_idx_seq(stcs, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in stcs]


def load_iob2(file_path):
    '''加载 IOB2 格式的数据'''
    token_seqs = []
    label_seqs = []
    tokens = []
    labels = []
    with codecs.open(file_path) as f:
        for index, line in enumerate(f):
            items = line.strip().split()
            if len(items) == 2:
                token, label = items
                tokens.append(token)
                labels.append(label)
            elif len(items) == 0:
                if tokens:
                    token_seqs.append(''.join(tokens))
                    label_seqs.append(labels)
                    tokens = []
                    labels = []
            else:
                print('格式错误。行号：{} 内容：{}'.format(index, line))
                continue

    if tokens:  # 如果文件末尾没有空行，手动将最后一条数据加入序列的列表中
        token_seqs.append(''.join(tokens))
        label_seqs.append(labels)

    return token_seqs, label_seqs


def deal(file_path):
    stcs, label_seqs = load_iob2(file_path)
    word2idx = Constants.init_word_2_id
    tag2idx = Constants.init_tag_2_id
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

    label_set = set([label for labels in label_seqs for label in labels])
    for label in label_set:
        if label not in tag2idx:
            tag2idx[label] = len(tag2idx)
    print('tag2idx', tag2idx)
    for word in word2idx.keys():
        full_vocab.append(word)

    # 根据词典制作embedding层的权重
    print('[Info] 加载词向量')
    if 'l' in language:
        glove = vocab.GloVe(name='6B', dim=300, cache=english_embed_cache_path)
    else:
        glove = gensim.models.KeyedVectors.load_word2vec_format(
            chinese_embed_cache_path)

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
    label_seqs = [[tag2idx.get(w) for w in s] for s in label_seqs]
    data = {
        'word2idx': word2idx,
        'tag2idx': tag2idx,
        'vocab': full_vocab,
        'embedding_weight': weights_matrix,
        'train': {
            'text': stcs[:split_ix],
            'tag': label_seqs[:split_ix]
        },
        'valid': {
            'text': stcs[split_ix:],
            'tag': label_seqs[split_ix:]
        }
    }

    return data


if __name__ == '__main__':
    language = 'chinese'
    data = deal(file_path)
    print('[Info] Finish.')
    torch.save(data, Constants.msra_data_path)
