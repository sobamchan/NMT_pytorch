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

    device = torch.device('cuda' if args.use_cuda else 'cpu')

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

    encoder.to(device)
    decoder.to(device)
    src_embedder.to(device)
    tgt_embedder.to(device)

    # data
    lines = open(args.src, 'r', encoding='utf-8').readlines()
    lines = [l.strip() + ' </s>' for l in lines]
    if args.src.find('ja') == -1:
        X = [utils.normalize_string(line).split() for line in lines]
    else:
        X = [line.lower().split() for line in lines]

    translated = []
    for x in X:
        idxs = list(map(lambda w: sw2i.get(w, sw2i['<UNK>']), x))
        print(x)
        idxs = torch.tensor([idxs], device=device)
        length = idxs.size(1)
        output, hidden_c = encoder(src_embedder, idxs, [length])
        start_decode =\
            torch.tensor([[tw2i['<s>']] * 1])

        # preds: 1, 50, V
        preds = decoder(
                tgt_embedder,
                start_decode,
                hidden_c,
                50,
                output,
                None,
                False
                )

        # preds_max: 1, 50
        preds_max = torch.max(preds, 2)[1]
        sent = ' '.join([tw2i[p] for p in preds_max.data[0].tolist()])
        translated.append(sent)

    return translated


if __name__ == '__main__':

    args = get_args()

    # GPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print('using GPU id: ', os.environ['CUDA_VISIBLE_DEVICES'])
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    lines = main(args)
    print(lines)
