import os
import argparse
from distutils.util import strtobool
import torch

from libs import models
from libs import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=0)

    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)

    parser.add_argument('--use-cuda', type=strtobool, default='1')
    return parser.parse_args()


def main(args):
    cp = torch.load(args.checkpoint)
    cargs = cp['args']
    sw2i = cp['sw2i']
    tw2i = cp['tw2i']

    encoder = models.Encoder(
            len(sw2i),
            cargs.src_embedding_size,
            cargs.encoder_hidden_n,
            n_layers=cargs.encoder_num_layers,
            bidirec=cargs.encoder_bidirectional,
            use_cuda=args.use_cuda
            )

    if cargs.decoder_bidirectional:
        decoder_hidden_size = cargs.decoder_hidden_n * 2
    else:
        decoder_hidden_size = cargs.decoder_hidden_n

    decoder = models.Decoder(
            len(tw2i),
            cargs.tgt_embedding_size,
            decoder_hidden_size,
            n_layers=cargs.decoder_num_layers,
            use_cuda=args.use_cuda
            )
    src_embedder = models.Embedder(
            len(sw2i),
            cargs.src_embedding_size,
            args.use_cuda
            )
    tgt_embedder = models.Embedder(
            len(tw2i),
            cargs.tgt_embedding_size,
            args.use_cuda
            )

    encoder.load_state_dict(cp['encoder_state_dict'])
    decoder.load_state_dict(cp['decoder_state_dict'])
    src_embedder.load_state_dict(cp['src_embedder'])
    tgt_embedder.load_state_dict(cp['tgt_embedder'])

    # data
    lines = open(args.src, 'r', encoding='utf-8').readlines()
    if args.src.find('ja') == -1:
        x = [utils.normalize_string(line).split() for line in lines]
    else:
        x = [line.lower.split() for line in lines]


if __name__ == '__main__':

    args = get_args()

    # GPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print('using GPU id: ', os.environ['CUDA_VISIBLE_DEVICES'])
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    main(args)
