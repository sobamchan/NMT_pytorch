import numpy as np
import torch
from torch import LongTensor as LT
from torch.autograd import Variable
from libs import utils


class Evaluator:

    def __init__(self, trainer):
        self.trainer = trainer
        self.args = trainer.args
        self.sw2i = trainer.sw2i
        self.si2w = trainer.si2w
        self.tw2i = trainer.tw2i
        self.ti2w = trainer.ti2w
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.src_embedder = trainer.src_embedder
        self.tgt_embedder = trainer.tgt_embedder
        self.loss_func = trainer.loss_func
        self.test_dataloader = trainer.test_dataloader

    def calc_test_loss(self, log_dict):
        sw2i = self.sw2i
        tw2i = self.tw2i
        losses = []
        for batch in self.test_dataloader:
            batch = utils.prepare_batch(batch, sw2i, tw2i)
            inputs, targets, input_lengths, target_lengths =\
                utils.pad_to_batch(batch, sw2i, tw2i)

            start_decode =\
                Variable(LT([[tw2i['<s>']] * targets.size(0)]),
                         requires_grad=False)\
                .transpose(0, 1)

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

        log_dict['test_loss'] = np.mean(losses)

        log_dict['sample_translation'] = {}
        log_dict['sample_translation']['src'] =\
            self.test_dataloader.dataset.src[0]
        log_dict['sample_translation']['tgt'] =\
            self.test_dataloader.dataset.tgt[0]
        preds = preds.view(inputs.size(0), targets.size(1), -1)
        preds_max = torch.max(preds, 2)[1]
        log_dict['sample_translation']['prediction'] =\
            ' '.join([self.ti2w[p] for p in preds_max.data[0].tolist()])

        return log_dict
