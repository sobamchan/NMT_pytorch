import argparse
from distutils.util import strtobool
from libs.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../DATA/giga-fren')
    parser.add_argument('--src-lang', type=str, default='fr')
    parser.add_argument('--tgt-lang', type=str, default='en')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--src-vocab-size', type=int, default=30000)
    parser.add_argument('--tgt-vocab-size', type=int, default=30000)
    parser.add_argument('--src-embedding-size', type=int, default=256)
    parser.add_argument('--encoder-dropout-p', type=float, default=0.1)
    parser.add_argument('--encoder-hidden-n', type=int, default=256)
    parser.add_argument('--encoder-num-layers', type=int, default=1)
    parser.add_argument('--tgt-embedding-size', type=int, default=256)
    parser.add_argument('--decoder-dropout-p', type=float, default=0.1)
    parser.add_argument('--decoder-hidden-n', type=int, default=256)
    parser.add_argument('--decoder-num-layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use-cuda', type=strtobool, default='1')
    parser.add_argument('--encoder-bidirectional', type=strtobool, default='0')
    parser.add_argument('--decoder-bidirectional', type=strtobool, default='0')
    args = parser.parse_args()
    print(args)

    trainer = Trainer(args)
    for _ in range(50):
        trainer.train_one_epoch()
