import torch
import torch.nn as nn
from common import Constants
from elmoformanylangs import Embedder


class embedding_layer(nn.Module):
    def __init__(self, vocab_sz, embed_sz):
        super(embedding_layer, self).__init__()
        self.vocab_sz = vocab_sz
        self.embed_sz = embed_sz

    def _forward_alg(self, x):
        return

    def forward(self, x):
        x = self._forward_alg(x)
        return x


class normal_embedding_layer(nn.Module):
    def __init__(self, vocab_sz, embed_sz, pretrained=False, pretrained_weight='', frozen=False):
        super(normal_embedding_layer, self).__init__()
        self.vocab_sz = vocab_sz
        self.embed_sz = embed_sz
        self.embed = nn.Embedding(self.vocab_sz, self.embed_sz, padding_idx=Constants.PAD)
        if pretrained:
            assert pretrained_weight is not None
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
            if frozen:
                self.embeds.weight.requires_grad = False
            print('[INFO][embedding layer]使用预加载的词向量')

    def forward(self, x):
        x = self.embed(x)
        return x


class elmo_embedding_layer(nn.Module):
    def __init__(self, vocab):
        super(elmo_embedding_layer, self).__init__()
        self.embed = Embedder('./embed_module/elmo/chinese')
        self.vocab = vocab

    def forward(self, x):
        '''
        保证调用时代码的一致性，这里的输入为batch_sz x seq_len的tensor
        步骤：
        1.使用vocab转换为sentence list
        2.forward
        :param x:tensor [batch_sz x seq_len]
        :return:tensor [batch_sz x seq_len x dim]
        '''
        seq_batch = [[self.vocab[i] for i in seq] for seq in x]
        seq_embed = self.embed.sents2elmo(seq_batch)
        return torch.Tensor(seq_embed)
