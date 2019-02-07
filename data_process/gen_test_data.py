import torch
import Constants
import stringUtil
from data_process.gen_data import convert_instance_to_idx_seq, make_text_tag
from chunck import Chunck


#
# # 将文本转换为index，找不到则为unk
# def convert_instance_to_idx_seq(stcs, word2idx):
#     ''' Mapping words to idx sequence. '''
#     return [[word2idx.get(w, Constants.UNK) for w in s] for s in stcs]
def gen_test_data(test_path):
    sentence_list = []
    tag_list = []
    with open(test_path, 'r') as f:
        for line in f:
            items = line.split('\t')
            line = stringUtil.clear_blank(items[3])
            text, tag = make_text_tag(items[0], items[1], items[2], line)
            sentence_list.append(text)
            tag_list.append(tag)
    data = torch.load(Constants.msra_data_path)
    sentence_list = convert_instance_to_idx_seq(sentence_list, data['word2idx'])
    test_data = {
        'word2idx': data['word2idx'],
        'tag2idx': data['tag2idx'],
        'vocab': data['vocab'],
        'test': {
            'text': sentence_list,
            'tag': tag_list
        }
    }
    torch.save(test_data, Constants.test_data_path)
