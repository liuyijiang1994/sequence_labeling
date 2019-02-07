import torch
import Constants
import numpy as np

from torch.utils import data


def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(len(inst) for inst in insts)
    batch_text = np.array([inst + [Constants.PAD] * (max_len - len(inst))
                           for inst in insts])
    batch_text = torch.LongTensor(batch_text)

    return batch_text


def collate_test_fn(insts):
    ''' Pad the instance to the max seq length in batch '''
    text, chuncks = list(zip(*insts))
    text = collate_fn(text)

    return text, list(chuncks)


def collate_tag_fn(insts):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(len(inst) for inst in insts)
    batch_text = np.array([[Constants.BOS] + inst + [Constants.PAD] * (max_len - len(inst))
                           for inst in insts])
    batch_text = torch.LongTensor(batch_text)

    return batch_text


def paired_collate_fn(insts):
    text, tag = list(zip(*insts))
    text = collate_fn(text)
    tag = collate_tag_fn(tag)
    return (text, tag)


class seqlabel_dataset(data.Dataset):
    def __init__(self, path, stage='train'):
        self.data = torch.load(path)
        self.word2idx = self.data['word2idx']
        self.tag2idx=self.data['tag2idx']
        self.vocab = self.data['vocab']
        self.data = self.data[stage]
        self.text = self.data['text']
        self.tag = self.data['tag']

    def __getitem__(self, index):
        return self.text[index], self.tag[index]

    def __len__(self):
        return len(self.text)


class seqlabel_test_dataset(data.Dataset):
    def __init__(self, path):
        self.data = torch.load(path)
        self.word2idx = self.data['word2idx']
        self.tag2idx=self.data['tag2idx']
        self.vocab = self.data['vocab']
        self.text = self.data['text']

    def __getitem__(self, index):
        return self.text[index], self.chuncks[index]

    def __len__(self):
        return len(self.text)


if __name__ == "__main__":
    valid_set = seqlabel_dataset(Constants.msra_data_path, stage='train')

    valid_iter = torch.utils.data.DataLoader(dataset=valid_set,
                                             batch_size=4,
                                             num_workers=0,
                                             collate_fn=paired_collate_fn,
                                             drop_last=True)
    for text,tag in valid_iter:
        print(text,tag)
        break
    print(valid_set.tag2idx)