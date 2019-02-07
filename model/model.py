import torch
import torch.nn as nn
import Constants

from model.sub_layer import normal_embedding_layer, elmo_embedding_layer

PAD_IDX = Constants.PAD
SOS_IDX = Constants.BOS
EOS_IDX = Constants.EOS
UNK_IDX = Constants.UNK

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class lstm_crf(nn.Module):
    def __init__(self, vocab_size, num_tags, cfg):
        super().__init__()
        self.lstm = lstm(vocab_size, num_tags, cfg)
        self.crf = crf(num_tags, cfg)

    def forward(self, x, y):  # for training
        '''
        gt：
        逐元素比较input和other ，
        不一样的元素位置的值为1，一样的为0
        '''

        mask = x.data.gt(PAD_IDX).float()  # 获得input的mask（填充位置为0）
        h = self.lstm(x, mask)
        Z = self.crf.forward(h, mask)
        score = self.crf.score(h, y, mask)
        return Z - score  # NLL loss

    def decode(self, x):  # for prediction
        mask = x.data.gt(PAD_IDX).float()
        h = self.lstm(x, mask)
        return self.crf.decode(h, mask)


class lstm(nn.Module):
    def __init__(self, vocab_size, num_tags, cfg):
        super().__init__()
        self.embed_sz = cfg.getint('model', 'EMBED_SZ')
        self.num_layer = cfg.getint('model', 'NUM_LAYER')
        self.hidden_sz = cfg.getint('model', 'HIDDEN_SZ')
        self.bidirectional = cfg.getboolean('model', 'BIDIRECTIONAL')
        self.num_dir = cfg.getint('model', 'NUM_DIRS')
        self.dropout = cfg.getfloat('model', 'DROPOUT')
        data = torch.load(cfg.get('data', 'DATA_PATH'))
        # architecture
        # pretrained / random / elmo

        embed_type = cfg.get('model', 'EMBEDDING')
        if 'random' == embed_type:
            self.embed = normal_embedding_layer(vocab_size, self.embed_sz)

        if 'pretrained' == embed_type:
            self.embed = normal_embedding_layer(vocab_size, self.embed_sz, pretrained=True,
                                                pretrained_weight=data['embedding_weight'], frozen=False)

        if 'elmo' == embed_type:
            self.embed = elmo_embedding_layer(data['vocab'])
            self.embed_sz = 1024
        assert self.embed is not None

        self.lstm = nn.LSTM(
            input_size=self.embed_sz,
            hidden_size=self.hidden_sz // self.num_layer,
            num_layers=self.num_layer,
            bias=True,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        self.out = nn.Linear(self.hidden_sz, num_tags)  # LSTM output to tag

    def init_hidden(self, batch_sz):  # initialize hidden states
        h = zeros(self.num_dir * self.num_layer, batch_sz, self.hidden_sz // self.num_layer)  # hidden states
        c = zeros(self.num_dir * self.num_layer, batch_sz, self.hidden_sz // self.num_layer)  # cell states
        return (h.to(device), c.to(device))

    def forward(self, x, mask):
        batch_sz = x.shape[0]
        self.hidden = self.init_hidden(batch_sz)
        x = self.embed(x).to(device)
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first=True)
        h, _ = self.lstm(x, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h = self.out(h)
        h *= mask.unsqueeze(2)
        return h


class crf(nn.Module):
    def __init__(self, num_tags, cfg):
        super().__init__()
        self.num_tags = num_tags

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[SOS_IDX, :] = -10000.  # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000.  # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000.  # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000.  # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0.
        self.trans.data[PAD_IDX, PAD_IDX] = 0.

    def forward(self, h, mask):  # forward algorithm
        batch_sz = mask.shape[0]
        # initialize forward variables in log space
        score = Tensor(batch_sz, self.num_tags).fill_(-10000.)  # [B, C]
        score[:, SOS_IDX] = 0.
        trans = self.trans.unsqueeze(0)  # [1, C, C]
        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans  # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[EOS_IDX])
        return score  # partition function

    def score(self, h, y, mask):  # calculate the score of a given sequence
        batch_sz = mask.shape[0]

        score = Tensor(batch_sz).fill_(0.)
        h = h.unsqueeze(3)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t]
            emit_t = torch.cat([h[t, y[t + 1]] for h, y in zip(h, y)])
            trans_t = torch.cat([trans[y[t + 1], y[t]] for y in y])
            score += (emit_t + trans_t) * mask_t
        last_tag = y.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[EOS_IDX, last_tag]
        return score

    def decode(self, h, mask):  # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        batch_sz = mask.shape[0]
        bptr = LongTensor()
        score = Tensor(batch_sz, self.num_tags).fill_(-10000.)
        score[:, SOS_IDX] = 0.

        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans  # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2)  # best previous scores and tags
            score_t += h[:, t]  # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[EOS_IDX]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(batch_sz):
            x = best_tag[b]  # best tag
            y = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:y]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path


def Tensor(*args):
    x = torch.Tensor(*args)
    return x.to(device)


def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.to(device)


def randn(*args):
    x = torch.randn(*args)
    return x.to(device)


def zeros(*args):
    x = torch.zeros(*args)
    return x.to(device)


def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))
