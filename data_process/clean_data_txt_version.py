# -*- coding: UTF-8 -*-

from utils import stringUtil
import re
from utils.langconv import *

max_sub_len = 80


# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line


# 转换简体到繁体
def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line


def split_fn(re_str, text):
    text = stringUtil.clear_blank(text)
    substcs = re.split(re_str, text)
    new_sub = []

    for i in range(0, len(substcs), 2):
        if i + 1 >= len(substcs):
            new_sub.append(substcs[i])
        else:
            new_sub.append(substcs[i] + substcs[i + 1])
    return new_sub


def split_sub_stc(text):
    re_str = r"([,，])"
    return split_fn(re_str, text)


# 以句号等分割句子
def split_stc(text):
    re_str = r"([.。!！?？；；;])"
    return split_fn(re_str, text)


def get_relavent_part(sub, s0):
    idxs = [s0]
    s = sub[s0]
    for i in range(1, len(sub)):
        if s0 - i >= 0:
            if len(s + sub[s0 - i]) < max_sub_len:
                s = s + sub[s0 - i]
                idxs.append(s0 - i)
        if s0 + i <= len(sub) - 1:
            if len(s + sub[s0 + i]) < max_sub_len:
                s = s + sub[s0 + i]
                idxs.append(s0 + i)

    return s, idxs


def sort_num(n1, n2):
    if n1 > n2:
        t = n2
        n2 = n1
        n1 = t
    return n1, n2


#
# def get_relavent_part(sub, s1, s2):
#     s = ''.join(sub[s1:s2 + 1])
#
#     if len(s) > max_sub_len:
#         if s1 == s2:
#             return s
#         else:
#             s = ''
#             for i in range(s1, s2, 1):


def get_sub_stc(words, text):
    text = stringUtil.clear_blank(text)
    substcs = split_stc(text)
    new_sub = substcs

    result = []

    find1 = -1
    find2 = -1
    find1_idx = []
    for idx, sub in enumerate(new_sub):
        if find1 > 0 and find2 > 0:
            break
        if find1 < 0 and sub.find(words[2]) >= 0 and sub not in result:
            if len(sub) > max_sub_len:
                sub_t = split_sub_stc(sub)
                s0 = -1
                for i in range(len(sub_t)):
                    if s0 < 0 and sub_t[i].find(words[2]) >= 0:
                        s0 = i
                        break

                assert s0 >= 0
                a, sub_idxs = get_relavent_part(sub_t, s0)
                result.append(a)
                for sub_idx in sub_idxs:
                    find1_idx.append(str(idx) + '-' + str(sub_idx))

            else:
                result.append(sub)
                find1_idx.append(str(idx))
            find1 = 1

        if find2 < 0 and sub.find(words[0]) >= 0 and sub.find(words[1]) >= 0 and abs(
                sub.find(words[0]) - sub.find(words[1])) < 10 and sub not in result:

            if len(sub) > max_sub_len:
                sub_t = split_sub_stc(sub)
                s1 = -1
                s2 = -1
                for i in range(len(sub_t)):
                    if s1 < 0 and sub_t[i].find(words[0]) >= 0:
                        s1 = i
                    if s2 < 0 and sub_t[i].find(words[1]) >= 0:
                        s2 = i
                    if s1 >= 0 and s2 >= 0:
                        break
                s1, s2 = sort_num(s1, s2)
                sss = ''.join(sub_t[s1:s2 + 1])
                # print(find1_idx)
                # print(str(idx) + '-' + str(s1))
                if str(idx) + '-' + str(s1) not in find1_idx:
                    result.append(sss)
            else:
                if idx not in find1_idx:
                    result.append(sub)
            find2 = 1
        else:
            continue
    if find1 > 0 and find2 > 0:
        str1 = ''.join('%s' % s for s in result)
        if len(str1) < 180:
            return words, str1
        else:
            return None, None
    else:
        return None, None


cont = 0
with open('./data/fusion_test.txt', 'r', encoding='utf-8') as seed:
    with open('./data/fusion_test_cleaned.txt', 'w', encoding='utf-8') as f:
        for line in seed:
            cont += 1
            if cont % 1000 == 0:
                print(cont)
            items = line.strip().split('\t')
            words, text = [items[0], items[1], items[2]], items[3]
            if words is not None and text is not None:
                words, text = get_sub_stc(words, text)
                if words is not None:
                    check = 1
                    text = text[:-1] + '。'
                    for word in words:
                        if text.find(word) < 0:
                            check = -1
                    if check == 1:
                        f.write(words[0] + '\t' + words[1] + '\t' + words[2] + '\t' + text + '\n')  # 加\n换行显示
