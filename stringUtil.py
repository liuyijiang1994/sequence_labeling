# -*- coding: utf-8 -*-

import re


def is_empty(string):
    return string == None or string.strip() == ''


def clear_blank(string):
    a = string.replace('\n', '。').replace('\r', '').replace('　', '。').replace('\t', '').replace('　　', '').replace(
        '	', '').replace(
        ' ', '').replace(
        '。。。', '。').replace('。。', '。').replace('', '').replace('“', '').replace('”', '').replace('\s+', '。').replace(
        '〞', '').replace('．', '。').replace('＂', '').replace('…', '。')

    a = ''.join(a.split())

    return a

    # a='asd as das das d\n \r \n'
    # # print(a.replace(' ','').replace('\n','').replace('\r',''

    # 为了缩短句子，将str中与三个词语无关的部分去掉
    def sub_corp(str, w_list):
        result = []
        str = str.replace('【', '').replace('】', '')
        sten_list = re.split(r"。", str)
        for sten in sten_list:
            for w in w_list:
                if sten.find(w) >= 0:
                    result.append(sten)
        return ''.join(result)
