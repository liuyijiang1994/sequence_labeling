import re
from model import *
import random
import os
from common import Constants
from common.chunck import Chunck
import torch
import time

def get_cost_time(start,end):
    seconds=end - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)




def get_seq_len_batch(data):
    '''
    计算batch中每个句子的真实长度
    :param data:pad后的batch[batch_sz,seq_len]
    :return:[batch_sz]
    '''
    seq_lengths = data.gt(Constants.PAD).sum(1)
    return seq_lengths


def mk_current_dir():
    path = ''.join(random.sample('zyxwvutsrqponmlkjihgfedcbaQWERTYUIOPASDFGHJKLZXCVBNM1234567890', 6))
    cur_path = os.path.abspath(os.curdir)
    # 去除首位空格
    path = path.strip()
    path = path.rstrip("\\")
    path = cur_path + '/result/' + path
    log_path = path + '/logs'
    save_path = path + '/save'
    isExists = os.path.exists(cur_path + '/' + path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        os.makedirs(log_path)
        os.makedirs(save_path)
        return path, log_path, save_path
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return path, log_path, save_path


def normalize(x):
    # x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x


def tokenize(x, unit):
    x = normalize(x)
    if unit == "char":
        return re.sub(" ", "", x)
    if unit == "word":
        return x.split(" ")


def save_data(filename, data):
    fo = open(filename + ".csv", "w")
    for seq in data:
        fo.write(" ".join(seq) + "\n")
    fo.close()


def load_word_to_idx(filename):
    print("loading word_to_idx...")
    word_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        word_to_idx[line] = len(word_to_idx)
    fo.close()
    return word_to_idx


def save_word_to_idx(filename, word_to_idx):
    fo = open(filename + ".word_to_idx", "w")
    for word, _ in sorted(word_to_idx.items(), key=lambda x: x[1]):
        fo.write("%s\n" % word)
    fo.close()


def load_tag_to_idx(filename):
    print("loading tag_to_idx...")
    tag_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        tag_to_idx[line] = len(tag_to_idx)
    fo.close()
    return tag_to_idx


def save_tag_to_idx(filename, tag_to_idx):
    fo = open(filename + ".tag_to_idx", "w")
    for tag, _ in sorted(tag_to_idx.items(), key=lambda x: x[1]):
        fo.write("%s\n" % tag)
    fo.close()


def load_checkpoint(filename, model=None):
    checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("loaded saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch


def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        print("saving model...")
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch %d" % epoch)


def iob_to_txt(txt, tags, unit):
    y = ""
    txt = tokenize(txt, unit)
    for i, j in enumerate(tags):
        if i and j[0] == "B":
            y += " "
        y += txt[i]
    return y


def f1(p, r):
    if p + r:
        return 2 * p * r / (p + r)
    return 0


# Turn a sequence of IOBES chunks into single tokens
def bio_to_chuncks(id_seq, idx_2_tag):
    """
    convert to bio to chunck
    """
    current_bidx = -1
    current_len = 0
    current_type = ''
    chunck_list = []
    if not isinstance(id_seq, list):
        id_seq = id_seq.tolist()
    id_seq.append(3)
    for idx, id in enumerate(id_seq):
        tag = idx_2_tag[id]

        if tag.startswith('B'):
            if current_bidx >= 0:
                chk = Chunck(current_type, current_bidx, current_len)
                chunck_list.append(chk)

            current_bidx = idx
            current_type = tag.split('-')[1]
            current_len = 1

        if tag.startswith('I'):
            if current_bidx >= 0 and tag.split('-')[1] == current_type:
                current_len += 1
            else:
                current_bidx = -1
                current_len = 0
                current_type = ''

        if tag.startswith('O'):
            if current_bidx >= 0:
                chk = Chunck(current_type, current_bidx, current_len)
                chunck_list.append(chk)
                current_bidx = -1
                current_len = 0
                current_type = ''
        # print(tag)
        # print(current_bidx,current_len, current_type)
    return chunck_list
