from pathlib import Path
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


class Trainer:

    def __init__(self, args):
        self.args = args
        train_dataloader, test_dataloader =\
            get_dataloaders(args.train_src,
                            args.train_tgt,
                            args.valid_src,
                            args.valid_tgt,
                            args.batch_size,
                            args.src_vocab_size,
                            args.tgt_vocab_size)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.sw2i = train_dataloader.dataset.sw2i
        self.si2w = train_dataloader.dataset.si2w
        self.tw2i = train_dataloader.dataset.tw2i
        self.ti2w = train_dataloader.dataset.ti2w

        encoder = models.Encoder(len(self.sw2i),
                                 args.src_embedding_size,
                                 args.encoder_hidden_n,
                                 n_layers=args.encoder_num_layers,
                                 bidirec=args.encoder_bidirectional,
                                 use_cuda=args.use_cuda)
        if args.decoder_bidirectional:
            decoder_hidden_size = args.decoder_hidden_n * 2
        else:
            decoder_hidden_size = args.decoder_hidden_n
        decoder = models.Decoder(len(self.tw2i),
                                 args.tgt_embedding_size,
                                 decoder_hidden_size,
                                 n_layers=args.decoder_num_layers,
                                 use_cuda=args.use_cuda)
        src_embedder = models.Embedder(len(self.sw2i),
                                       args.src_embedding_size,
                                       args.use_cuda)
        tgt_embedder = models.Embedder(len(self.tw2i),
                                       args.tgt_embedding_size,
                                       args.use_cuda)
        encoder.init_weight()
        decoder.init_weight()
        if args.use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            src_embedder = src_embedder.cuda()
            tgt_embedder = tgt_embedder.cuda()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder

        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        self.enc_optim = optim.Adam(encoder.parameters(), lr=args.lr)
        self.dec_optim = optim.Adam(decoder.parameters(), lr=args.lr)
        self.src_embedder_optim = optim.Adam(self.src_embedder.parameters(),
                                             lr=args.lr)
        self.tgt_embedder_optim = optim.Adam(self.tgt_embedder.parameters(),
                                             lr=args.lr)

    def train_one_epoch(self, log_dict):
        sw2i = self.sw2i
        tw2i = self.tw2i
        losses = []
        for batch in tqdm(self.train_dataloader):
            batch = utils.prepare_batch(batch, sw2i, tw2i)
            inputs, targets, input_lengths, target_lengths =\
                utils.pad_to_batch(batch, sw2i, tw2i)

            start_decode =\
                Variable(LT([[tw2i['<s>']] * targets.size(0)])).transpose(0, 1)
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            self.src_embedder.zero_grad()
            self.tgt_embedder.zero_grad()

            if self.args.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                start_decode = start_decode.cuda()

            output, hidden_c = self.encoder(self.src_embedder,
                                            inputs,
                                            input_lengths)

            preds = self.decoder(self.tgt_embedder,
                                 start_decode,
                                 hidden_c,
                                 targets.size(1),
                                 output,
                                 None,
                                 True)
            loss = self.loss_func(preds, targets.view(-1))
            losses.append(loss.data[0])
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 50.0)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), 50.0)
            self.enc_optim.step()
            self.dec_optim.step()
            self.src_embedder_optim.step()
            self.tgt_embedder_optim.step()
        log_dict['train_loss'] = np.mean(losses)

    def dump_model(self, output_dir, fname=None):
        if fname:
            opath = Path(output_dir) / fname
        else:
            opath = Path(output_dir) / 'checkpoint.pth'

        cp = {
                'encoder_state_dict': self.encoder.to('cpu').state_dict(),
                'decoder_state_dict': self.decoder.to('cpu').state_dict(),
                'src_embedder': self.src_embedder.to('cpu').state_dict(),
                'tgt_embedder': self.tgt_embedder.to('cpu').state_dict(),
                'sw2i': self.sw2i,
                'si2w': self.si2w,
                'tw2i': self.tw2i,
                'ti2w': self.ti2w,
                'args': self.args,
                }

        print('dumping model to %s' % opath)

        with open(opath, 'wb') as f:
            torch.save(cp, f)

        if self.args.use_cuda:
            self.encoder.to('cuda')
            self.decoder.to('cuda')
            self.src_embedder.to('cuda')
            self.tgt_embedder.to('cuda')
