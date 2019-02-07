import configparser
from utils.utils import *
from mydataasets import seqlabel_dataset, paired_collate_fn
from train import sort_by_seq_len
import utils
from model.evalator import eval_w
from data_process.gen_test_data import gen_test_data
import torch
from model.model import lstm_crf

model_path = '/tmp/pycharm_project_229/sequence_labeling/result/rLYkSi/save/model_5.pt'
cfg_path = './config/embed1.cfg'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SZ = 256


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


def predict():
    data = []
    model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    for text, tag in test_iter:
        if text.shape[0] == BATCH_SZ:
            text = text.to(device)
            indices, desorted_indices = sort_by_seq_len(text)
            text = text[indices]
            tag = tag[indices]
            result = model.decode(text)
            tag = clean_pad(tag)
            for i in range(len(result)):
                print(result[i])
                print(tag[i])
                print('-' * 10)
            data.append(result)
    return result


def get_test_score():
    model, word_to_idx, tag_to_idx, idx_to_tag = load_model()
    guess_tag_list = []
    gold_tag_list = []
    sentence_list = []
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
            sentence_list.extend(text.tolist())
    return guess_tag_list, gold_tag_list, sentence_list


def convert_idx_to_word(seq_idx, vocab):
    return ''.join([vocab[idx] for idx in seq_idx])


def visual_chunck_list(chunck_list, sentence, vocab):
    visual_chuncks = []
    for chunck in chunck_list:
        chunck_name = convert_idx_to_word(sentence[chunck.b_idx:chunck.b_idx + chunck.length], vocab)
        visual_chuncks.append(chunck_name)
    return visual_chuncks


if __name__ == "__main__":
    test_path = './data/test.txt'
    print('generating test data...')
    gen_test_data(test_path)
    eva_matrix = 'fa'
    test_set = seqlabel_dataset(Constants.msra_data_path, stage='train')
    test_iter = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SZ, num_workers=0,
                                            collate_fn=paired_collate_fn)
    print(test_set.tag2idx)
    idx2tag = {v: k for k, v in test_set.tag2idx.items()}
    evaluator = eval_w(test_set.tag2idx, eva_matrix)
    with torch.no_grad():
        guess_tag_list, gold_tag_list, sentence_list = get_test_score()

    print(len(guess_tag_list), len(gold_tag_list), len(sentence_list))
    pre_seq = {
        'guess_tag_list': guess_tag_list,
        'gold_tag_list': gold_tag_list,
        'sentence_list': sentence_list
    }
    torch.save(pre_seq, 'predict_seq.pt')

    with open('predict.txt', 'w') as f:
        for i in range(len(sentence_list)):
            visual_sentence = convert_idx_to_word(sentence_list[i], test_set.vocab)
            guess_chunck_list = utils.bio_to_chuncks(guess_tag_list[i], idx2tag)
            gold_chunck_list = utils.bio_to_chuncks(gold_tag_list[i], idx2tag)
            guess_visual = visual_chunck_list(guess_chunck_list, sentence_list[i], test_set.vocab)
            gold_visual = visual_chunck_list(gold_chunck_list, sentence_list[i], test_set.vocab)
            f.write(f'{visual_sentence}\n')
            f.write(f'guess:{guess_visual}\n')
            f.write(f'gold:{gold_visual}\n')
            f.write('-' * 20)
    print(len(guess_tag_list), len(sentence_list))
    #
    # print(len(guess_tag_list), len(gold_tag_list))
    # if 'f' in eva_matrix:
    #     test_f1 = evaluator.calc_score(guess_tag_list, gold_tag_list)
    #     for k, v in test_f1.items():
    #         print(k, v)
    # else:
    #     test_acc = evaluator.calc_score(guess_tag_list, gold_tag_list)
    #     print(' test_acc: %.4f\n' % (test_acc))
