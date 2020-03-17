import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import Net
from dataloader import get_data_loader

import argparse

parser = argparse.ArgumentParser(description='Reading-VQA')
parser.add_argument('--data_root', required=True, type=str, help='root directory of dataset')
parser.add_argument('--stage', default='train', choices=('train', 'test'), help='model stage')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--epochs', default=20, type=int, help='total epochs to train')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--load', type=str, help='loading specific model checkpoint')
parser.add_argument('--save_dir', default='./checkpoints', type=str, help='directory for saving model checkpoints')
parser.add_argument('--save_freq', default=500, type=int, help='number of iterations between two saving actions')

args = parser.parse_args()

def train(model, data_loader, optimizer, criterion):
    model.train()
    iteration_count = 0
    for epoch in range(args.epochs):
        for data in tqdm(data_loader, desc='Epoch %s' % epoch):
            model.zero_grad()
            logits = model(data)
            loss = criterion(logits, data['answer']).mean()
            preds = logits.argmax(1)
            acc = (preds == data['answer']).float().mean()
            print(model.tokenizer.decode(preds[preds == data['answer']].unsqueeze(-1)))
            print(loss.data.item(), acc.data.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            iteration_count += 1
            if iteration_count % args.save_freq == 0:
                os.system('mkdir -p %s' % args.save_dir)
                torch.save(model.state_dict(), os.path.join(args.save_dir, '%x_%d.pt' % (int(time.time()), iteration_count)))

def main():
    use_cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Loading model ...')
    model = Net(device)
    if args.load is not None:
        print('Loading checkpoint ...')
        model.load_state_dict(torch.load(args.load))
    if use_cuda:
        model.cuda()
    print('Loading data ...')
    data_loader = get_data_loader(args.data_root, args.stage, args.batch_size)
    print('Preparation done')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    if args.stage == 'train':
        train(model, data_loader, optimizer, criterion)

if __name__ == '__main__':
    main()
