import re
import stringUtil
import xlrd

MAX_SEQ_LEN = 150


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


def stc_idx_by_words(words, sentence_list):
    '''
    找到list中的词语都在的那个句子的idx的list
    '''
    stc_idx_list = []
    for idx, stc in enumerate(sentence_list):
        words_in_flag = True
        for word in words:
            if stc.find(word) < 0:
                words_in_flag = False
                break
        if words_in_flag:
            stc_idx_list.append(idx)
    return stc_idx_list


def sort_num(n1, n2):
    if n1 > n2:
        t = n2
        n2 = n1
        n1 = t
    return n1, n2


def clip_sentence(p_idx, f_idx, sentence_list):
    '''
    输入候选的part_fusion对，和stc_lsit
    返回阉割后的句子
    '''
    if p_idx == f_idx:
        stc = sentence_list[p_idx]
    else:
        s1, s2 = sort_num(p_idx, f_idx)
        stc = sentence_list[s1] + sentence_list[s2]
    return stc


def make_instance(part1, part2, fusion, text):
    '''
    输入格式：得到的fusion和两个part token，原始文本
    输出：最后长度在max_seq_len左右，两个part词语同在的一个子句，和最近的fusion所在的子句。
    '''
    sentence_list = split_stc(text)
    fusion_idx_list = stc_idx_by_words([fusion], sentence_list)
    part_idx_list = stc_idx_by_words([part1, part2], sentence_list)
    p_f_map = {}
    for par_idx in part_idx_list:
        idx = -1
        min_dis = 99999
        for fu_idx in fusion_idx_list:
            if abs(fu_idx - par_idx) < min_dis:
                idx = fu_idx
                min_dis = abs(fu_idx - par_idx)

        p_f_map[par_idx] = idx
    cliped_sentence_list = []
    for key, value in p_f_map.items():
        stc = clip_sentence(key, value, sentence_list)
        cliped_sentence_list.append(stc)
    return cliped_sentence_list


if __name__ == '__main__':
    text = '夫妻麦客忙麦收。6月4日，在山东省临沂市郯城县郯城街道米顶村麦田间，范加江驾驶收割机在收获小麦。'
    part1, part2, fusion = '小麦', '收获', '麦收'
    stc_list = make_instance(part1, part2, fusion, text)
    print(stc_list)


    xls_path = '../data/down_data.xls'
    save_path = '../data/clean_download_data.txt'
    book = xlrd.open_workbook(xls_path)
    sheet = book.sheet_by_index(0)
    with open(save_path, 'w') as f:
        for i in sheet.get_rows():
            part1, part2, fusion, text = i[1].value, i[2].value, i[3].value, i[4].value
            stc_list = make_instance(part1, part2, fusion, text)
            for stc in stc_list:
                f.write(f'{part1}\t{part2}\t{fusion}\t{stc}\n')  # 加\n换行显示

