import configparser
from utils import *
from mydataasets import seqlabel_dataset, paired_collate_fn
from train import sort_by_seq_len
import utils
from model.evalator import eval_w
import torch
from model.model import lstm_crf
import sys, getopt


def load_model():
    print(f'[INFO] loading model...{model_path}')
    word_to_idx = test_set.word2idx
    tag_to_idx = test_set.tag2idx
    idx_to_tag = [tag for tag, _ in sorted(tag_to_idx.items(), key=lambda x: x[1])]
    conf = configparser.ConfigParser()
    conf.read(cfg_path)
    torch.manual_seed(conf.getint('model', 'MANUAL_SEED'))
    model = lstm_crf(len(word_to_idx), len(tag_to_idx), conf).to(device)
    print(model)
    model.eval()
    load_checkpoint(model_path, model)
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


def get_test_tag():
    model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    guess_tag_list = []
    gold_tag_list = []
    for text, tag in test_iter:
        if text.shape[0] == BATCH_SZ:
            text = text.to(device)
            indices, desorted_indices = sort_by_seq_len(text)
            text = text[indices]
            tag = tag[indices]
            result = model.decode(text)
            tag = clean_pad(tag)
            guess_tag_list.extend(result)
            gold_tag_list.extend(tag)
    return guess_tag_list, gold_tag_list


if __name__ == "__main__":

    model_path = '/tmp/pycharm_project_229/sequence_labeling/result/XZnt1s/save/model_18.pt'
    cfg_path = './config/fusion_word2vec.cfg'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SZ = 32

    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:c:t:", ['model_path=', 'conf_path=', 'test_path'])

    except getopt.GetoptError:
        print('参数错误')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-c", "--conf_path"):
            cfg_path = arg
        if opt in ("-m", "--model_path"):
            model_path = arg
        if opt in ("-t", "--test_path"):
            test_path = arg
    assert cfg_path is not None
    assert model_path is not None
    assert test_path is not None

    print(f'[INFO]MODEL PATH = {model_path}, CONFIG PATH = {cfg_path}, TEST DATA = {test_path}')
    # test_path = './data/test.txt'
    # print('generating test data...')
    # gen_test_data(test_path)
    eva_matrix = 'fa'
    # test_set = seqlabel_dataset(Constants.fusion_test, stage='test')
    test_set = seqlabel_dataset(test_path, stage='test')

    test_iter = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SZ, num_workers=0,
                                            collate_fn=paired_collate_fn)
    print(test_set.tag2idx)
    evaluator = eval_w(test_set.tag2idx, eva_matrix)
    with torch.no_grad():
        guess_tag_list, gold_tag_list = get_test_tag()

    print(len(guess_tag_list), len(gold_tag_list))
    if 'f' in eva_matrix:
        test_f1 = evaluator.calc_score(guess_tag_list, gold_tag_list)
        for k, v in test_f1.items():
            print(k, v)
    else:
        test_acc = evaluator.calc_score(guess_tag_list, gold_tag_list)
        print(' test_acc: %.4f\n' % (test_acc))
