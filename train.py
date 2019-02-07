from mydataasets import seqlabel_dataset
from mydataasets import paired_collate_fn
from tensorboardX import SummaryWriter
import configparser
import utils
import torch
from model.model import lstm_crf
import time
import sys, getopt


def sort_by_seq_len(data):
    # 对text 和 tag 按seq_len排序，便于使用pack_padded_sequence
    # 对序列长度进行排序(降序), sorted_seq_lengths = [5, 3, 2]
    # indices 为 [1, 0, 2], indices 的值可以这么用语言表述
    # 原来 batch 中在 0 位置的值, 现在在位置 1 上.
    # 原来 batch 中在 1 位置的值, 现在在位置 0 上.
    # 原来 batch 中在 2 位置的值, 现在在位置 2 上.
    seq_lengths = utils.get_seq_len_batch(data)
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
    # 如果我们想要将计算的结果恢复排序前的顺序的话,
    # 只需要对 indices 再次排序(升序),会得到 [0, 1, 2],
    # desorted_indices 的结果就是 [1, 0, 2]
    # 使用 desorted_indices 对计算结果进行索引就可以了.
    _, desorted_indices = torch.sort(indices, descending=False)
    return indices, desorted_indices


def vld():
    model.eval()
    total_loss = 0
    for idx, (text, tag) in enumerate(valid_iter):
        text, tag = text.to(device), tag.to(device)
        # 对原始序列进行排序
        indices, desorted_indices = sort_by_seq_len(text)
        text = text[indices]
        tag = tag[indices]
        loss = torch.mean(model(text, tag))  # forward pass and compute loss
        total_loss += loss.item()

    return total_loss / len(valid_iter)


def train(epoch):
    model.train()
    loss_sum = 0
    loss_all = 0
    total_len = len(training_iter)
    for idx, (text, tag) in enumerate(training_iter):
        text, tag = text.to(device), tag.to(device)

        # 对原始序列进行排序
        indices, desorted_indices = sort_by_seq_len(text)
        text = text[indices]
        tag = tag[indices]

        model.zero_grad()

        loss = torch.mean(model(text, tag))  # forward pass and compute loss
        loss_sum += loss.item()
        loss_all += loss.item()

        loss.backward()  # compute gradients
        optim.step()  # update parameters

        if idx % print_every == 0 and idx != 0:
            end = time.clock()
            used_time = utils.get_cost_time(start, end)
            ctime = time.strftime('%H:%M:%S')
            print(f"[INFO][{ctime}][cost:{used_time}][epoch:{epoch}][{idx}/{total_len}][loss:{loss_sum / print_every}]")
            loss_sum = 0

    return loss_all / len(training_iter)


if __name__ == "__main__":
    final_model_path = ''
    conf = configparser.ConfigParser()
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:", ['conf_path='])

    except getopt.GetoptError:
        print('参数错误')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-c", "--conf_path"):
            conf_path = arg
    assert conf_path is not None
    print(f'conf_path: {conf_path}')
    conf.read(conf_path)
    patience = conf.getint('train', 'PATIENCE')
    torch.manual_seed(conf.getint('model', 'MANUAL_SEED'))
    print_every = conf.getint('train', 'PRINT_EVERY')
    batch_sz = conf.getint('train', 'BATCH_SZ')
    data_path = conf.get('data', 'DATA_PATH')
    path, log_path, save_path = utils.mk_current_dir()
    print('current_path: ', path)
    print('log_path', log_path)
    print('save_path', save_path)

    writer = SummaryWriter(log_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_set = seqlabel_dataset(data_path)

    training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                                batch_size=batch_sz,
                                                num_workers=0,
                                                collate_fn=paired_collate_fn,
                                                drop_last=True)

    valid_set = seqlabel_dataset(data_path, stage='valid')

    valid_iter = torch.utils.data.DataLoader(dataset=valid_set,
                                             batch_size=batch_sz,
                                             num_workers=0,
                                             collate_fn=paired_collate_fn,
                                             drop_last=True)

    word_to_idx = training_set.word2idx
    tag_to_idx = training_set.tag2idx
    model = lstm_crf(len(word_to_idx), len(tag_to_idx), conf).to(device)
    print({section: dict(conf[section]) for section in conf.sections()})
    print(model)
    print(f'[INFO] traing data length : {len(training_set)}')
    print(f'[INFO] vld data length : {len(valid_set)}')

    optim_SGD = torch.optim.SGD(model.parameters(), lr=conf.getfloat('train', 'LEARNING_RATE'),
                                weight_decay=conf.getfloat('train', 'WEIGHT_DECAY'))
    optim_Adam = torch.optim.Adam(model.parameters(), lr=conf.getfloat('train', 'LEARNING_RATE'),
                                  weight_decay=conf.getfloat('train', 'WEIGHT_DECAY'))
    optims = {'SGD': optim_SGD, 'Adam': optim_Adam}
    optim = optims[conf.get('train', 'OPTIM')]
    # Train the model
    best_val_loss = None
    print(f"[INFO][{time.strftime('%H:%M:%S')}][==Strart Training==]")
    start = time.clock()
    for epoch in range(1, conf.getint('train', 'EPOCH') + 1):
        train_loss = train(epoch)
        print('[INFO][Epoch:%d] ---train_loss: %5.5f---' % (epoch, train_loss))
        vld_loss = vld()
        print('[INFO][Epoch:%d] ---valid_loss: %5.5f---' % (epoch, vld_loss,))

        if not best_val_loss or vld_loss < best_val_loss:
            print("[!] saving model...", f'{save_path}/model_{epoch}.pt ')
            checkpoint = {'state_dict': model.state_dict(), 'loss': vld_loss, 'epoch': epoch}
            torch.save(checkpoint, f'{save_path}/model_{epoch}.pt')
            best_val_loss = vld_loss
            final_model_path = f'{save_path}/model_{epoch}.pt'
            patience = conf.getint('train', 'PATIENCE')
        else:
            patience -= 1
        info = {'train_loss': train_loss, 'valid_loss': vld_loss}

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch)
            if value.grad is not None:
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        if patience <= 0:
            break

    writer.close()
    print(f'final_model_path:{final_model_path}')
