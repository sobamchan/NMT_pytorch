import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import LongTensor as LT
from torch.autograd import Variable
from libs import models
from libs import utils
from libs.dataset import get_dataloaders
from tqdm import tqdm


def train():
    epoch = 50
    batch_size = 128
    embedding_size = 256
    hidden_size = 256
    lr = 0.0001
    decoder_learning_ration = 5.0
    # rescheduled = False
    use_cuda = True
    bidirec = False

    train_dataloader, test_dataloader =\
        get_dataloaders('../DATA/small_parallel_enja',
                        'en',
                        'ja',
                        batch_size,
                        30000,
                        30000)
    sw2i = train_dataloader.dataset.sw2i
    # si2w = train_dataloader.dataset.si2w
    tw2i = train_dataloader.dataset.tw2i
    ti2w = train_dataloader.dataset.ti2w

    encoder = models.Encoder(len(sw2i),
                             embedding_size,
                             hidden_size,
                             1,
                             bidirec,
                             use_cuda)
    decoder = models.Decoder(len(tw2i),
                             embedding_size,
                             hidden_size * 2 if bidirec else hidden_size,
                             n_layers=1,
                             use_cuda=use_cuda)
    encoder.init_weight()
    decoder.init_weight()

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    enc_optim = optim.Adam(encoder.parameters(), lr=lr)
    dec_optim = optim.Adam(decoder.parameters(),
                           lr=lr * decoder_learning_ration)

    for i_epoch in range(epoch):
        losses = []
        for batch in tqdm(train_dataloader):
            batch = utils.prepare_batch(batch, sw2i, tw2i)
            inputs, targets, input_lengths, target_lengths =\
                utils.pad_to_batch(batch, sw2i, tw2i)

            start_decode =\
                Variable(LT([[tw2i['<s>']] * targets.size(0)])).transpose(0, 1)
            encoder.zero_grad()
            decoder.zero_grad()

            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                start_decode = start_decode.cuda()

            output, hidden_c = encoder(inputs, input_lengths)

            preds = decoder(start_decode,
                            hidden_c,
                            targets.size(1),
                            output,
                            None,
                            True)
            loss = loss_func(preds, targets.view(-1))
            losses.append(loss.data[0])
            loss.backward()
            nn.utils.clip_grad_norm(encoder.parameters(), 50.0)
            nn.utils.clip_grad_norm(decoder.parameters(), 50.0)
            enc_optim.step()
            dec_optim.step()
        print(np.mean(losses))
        preds = preds.view(inputs.size(0), targets.size(1), -1)
        preds_max = torch.max(preds, 2)[1]
        print(' '.join([ti2w[p] for p in preds_max.data[0].tolist()]))
        print(' '.join([ti2w[p] for p in preds_max.data[1].tolist()]))


if __name__ == '__main__':
    train()
