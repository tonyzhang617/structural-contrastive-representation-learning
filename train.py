import logging
import sys
from argparse import ArgumentParser
import pandas as pd
from os.path import join
import numpy as np
from multiprocessing import Pool
from functools import partial
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

from evaluate import evaluate


parser = ArgumentParser('structural_contrastive_learning')
parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v1')
parser.add_argument('--tokenizer', type=str, default='sentence-transformers/all-mpnet-base-v1')
parser.add_argument('--dataset', type=str, default='dataset/LF-Amazon-131K')
parser.add_argument('--min-len', type=int, default=40)
parser.add_argument('--max-len', type=int, default=80)
parser.add_argument('--tokenizer-max-len', type=int, default=None)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--eval-batch-size', type=int, default=None)
parser.add_argument('--temperature', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--min-lr', type=float, default=5e-6)
parser.add_argument('--scheduler', type=str, default='linear', choices=['linear', 'cosine'])
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--experiment', type=str, default='default')
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--eval-only', action='store_true')
parser.add_argument('--exclude-title-pairs', action='store_true')
parser.add_argument('--exclude-label-pairs', action='store_true')
parser.add_argument('--exclude-title-content-pairs', action='store_true')
parser.add_argument('--exclude-content-content-pairs', action='store_true')

args = parser.parse_args()

class StreamToLogger(object):
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


Path("out").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"out/{args.experiment}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger('structural_contrastive_learning')
sys.stdout = StreamToLogger(log,logging.INFO)
sys.stderr = StreamToLogger(log,logging.ERROR)

print(args)

device = torch.device(args.device)

tokenizer_max_len = args.tokenizer_max_len if isinstance(args.tokenizer_max_len, int) else 2 * args.max_len

model = AutoModel.from_pretrained(args.model)
model = nn.DataParallel(model.to(device))
tokenizer = AutoTokenizer.from_pretrained(args.model if args.tokenizer is None else args.tokenizer)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def val(self):
        return self.sum / self.count


class RandIndexDataset(Dataset):
    def __init__(self, range):
        self.range = range

    def __len__(self):
        return self.range

    def __getitem__(self, idx):
        return idx

def split_sentence(sentence, min_len, max_len, last_sent_min_len):
    if len(sentence) == 0:
        return []

    words = sentence.split()
    sent_len = len(words)
    if sent_len <= min_len:
        return [sentence]
    
    sentences = []
    idx = 0
    while idx < sent_len:
        new_len = np.random.randint(min_len, max_len + 1)
        if idx + new_len > sent_len and sent_len - idx < last_sent_min_len:
            sentences[-1] = sentences[-1] + ' ' + ' '.join(words[idx:sent_len])
        else:
            sentences.append(' '.join(words[idx:idx + new_len]))
        idx += new_len
    return sentences

def get_training_pairs_and_loader(titles, contents, min_len, max_len, last_sent_min_len, labels, batch_size):
    print('Generating training pairs...')
    np.random.seed(int(time.time()))
    training_pairs = []

    if not args.exclude_title_pairs:
        for t in titles:
            if len(t) > 0:
                training_pairs.append((t, t))

    if not args.exclude_label_pairs:
        for l in labels:
            if len(l) > 0:
                training_pairs.append((l, l))

    if not args.exclude_title_content_pairs:
        with Pool(processes=32) as pool:
            split_contents = pool.map(partial(split_sentence, min_len=min_len, max_len=max_len, last_sent_min_len=last_sent_min_len), contents)
            for i in range(len(titles)):
                for c in split_contents[i]:
                    if len(titles[i]) > 0:
                        training_pairs.append((titles[i], c))

    if not args.exclude_content_content_pairs:
        with Pool(processes=32) as pool:
            split_contents = pool.map(partial(split_sentence, min_len=min_len, max_len=max_len, last_sent_min_len=last_sent_min_len), contents)
            for i in range(len(contents)):
                len_split_content = len(split_contents[i])
                perm = np.random.permutation(len_split_content)
                for j in range(len_split_content // 2 + (len_split_content % 2 > 0)):
                    if j * 2 + 1 < len_split_content:
                        training_pairs.append((split_contents[i][perm[j * 2]], split_contents[i][perm[j * 2 + 1]]))
                    else:
                        training_pairs.append((split_contents[i][perm[j * 2]], split_contents[i][0]))
    
    training_pairs = np.array(training_pairs, dtype=object)
    dataset = RandIndexDataset(len(training_pairs))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return training_pairs, loader


trn_js = pd.read_json(join(args.dataset, 'trn.json'), lines=True)
tst_js = pd.read_json(join(args.dataset, 'tst.json'), lines=True)
lbl_js = pd.read_json(join(args.dataset, 'lbl.json'), lines=True)

tst_content = tst_js.content

if args.eval_only:
    evaluate(
        tokenizer, model, tst_js.title, tst_content, lbl_js.title, tst_js.target_ind,
        tst_js.uid, lbl_js.uid,
        batch_size=args.eval_batch_size if isinstance(args.eval_batch_size, int) else args.batch_size * 4,
        device=device, print_freq=args.print_freq,
    )
    exit(0)

sim = nn.CosineSimilarity(dim=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=args.min_lr / args.lr, total_iters=args.epochs, verbose=True) \
    if args.scheduler == 'linear' else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, verbose=True)

def get_similarity(a, b, temperature=0.05, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt / temperature

class PairLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x0, x1):
        sim = get_similarity(x0, x1)
        targets = torch.arange(x0.shape[0]).long().to(x0.device)
        return self.loss(sim, targets)

best_metric = 0
def save_best_model(model, metric):
    global best_metric
    if metric > best_metric:
        print('Saving model...')
        best_metric = metric
        model.module.save_pretrained(f'models/{args.experiment}')

loss_func = PairLoss()
loss_meter = AverageMeter()

for epoch in range(args.epochs):
    training_pairs, loader = get_training_pairs_and_loader(
        trn_js.title, trn_js.content,
        args.min_len, args.max_len, args.min_len // 2,
        lbl_js.title,
        batch_size=args.batch_size,
    )
    print(f"New Training Set Size: {len(training_pairs)}")

    total_iter = len(training_pairs) // args.batch_size + (len(training_pairs) % args.batch_size > 0)
    model.train()

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()

        pair_idx = batch.numpy()
        sentence_pairs = training_pairs[pair_idx]
        s0, s1 = list(zip(*sentence_pairs))
        t0 = tokenizer(list(s0), padding=True, truncation=True, max_length=tokenizer_max_len, return_tensors="pt")
        t1 = tokenizer(list(s1), padding=True, truncation=True, max_length=tokenizer_max_len, return_tensors="pt")
        for k in t0:
            t0[k] = t0[k].to(device)
        for k in t1:
            t1[k] = t1[k].to(device)

        o0 = model(**t0).last_hidden_state[:, 0]
        o1 = model(**t1).last_hidden_state[:, 0]
        loss = loss_func(o0, o1)
        loss_meter.update(loss.item())

        if batch_idx % args.print_freq == 0:
            print(f'epoch {epoch:3}/{args.epochs:3}, iter {batch_idx:5}/{total_iter:5}, loss {loss.item():7.4f} ({loss_meter.val:7.4f})')

        nn.utils.clip_grad_norm_(model.parameters(), 1)
        loss.backward()
        optimizer.step()

    loss_meter.reset()
    scheduler.step()

    prec = evaluate(
        tokenizer, model, tst_js.title, tst_content, lbl_js.title, tst_js.target_ind,
        tst_js.uid, lbl_js.uid,
        batch_size=args.eval_batch_size if isinstance(args.eval_batch_size, int) else args.batch_size * 4, 
        device=device, print_freq=args.print_freq,
    )
    save_best_model(model, prec)
