import numpy as np


def pad_batch(src_sents, tgt_sents, src_pad_id, tgt_pad_id):
    '''
    in:
      src_sents: list of word ids
    '''

    src_max_len = max([len(src_sent) for src_sent in src_sents])
    tgt_max_len = max([len(tgt_sent) for tgt_sent in tgt_sents])
    pad_src_sents = []
    pad_tgt_sents = []
    src_lens = []
    tgt_lens = []
    for src_sent, tgt_sent in zip(src_sents, tgt_sents):
        src_sent_len = len(src_sent)
        src_lens.append(src_sent_len)
        if src_sent_len < src_max_len:
            pad_src_sent =\
                src_sent + [src_pad_id] * (src_max_len - src_sent_len)
        else:
            pad_src_sent = src_sent
        pad_src_sents.append(pad_src_sent)

        tgt_sent_len = len(tgt_sent)
        tgt_lens.append(tgt_sent_len)
        if tgt_sent_len < tgt_max_len:
            pad_tgt_sent =\
                tgt_sent + [tgt_pad_id] * (tgt_max_len - tgt_sent_len)
        else:
            pad_tgt_sent = tgt_sent
        pad_tgt_sents.append(pad_tgt_sent)

    pad_src_sents =\
        np.array(pad_src_sents)[np.argsort(src_lens)[::-1]].tolist()
    pad_tgt_sents =\
        np.array(pad_tgt_sents)[np.argsort(src_lens)[::-1]].tolist()
    src_lens = np.array(src_lens)[np.argsort(src_lens)[::-1]]
    tgt_lens = np.array(tgt_lens)[np.argsort(src_lens)[::-1]]

    return pad_src_sents, pad_tgt_sents, src_lens, tgt_lens


def convert_s2i(sent, w2i):
    '''
    convert string sentence to list of ids
    '''
    seq = []
    for word in sent.split():
        if word in w2i.keys():
            seq.append(w2i[word])
        else:
            seq.append(w2i['<UNK>'])
    return seq
