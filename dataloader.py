import os
import json
import h5py
import random
import torch
from torch.utils.data import Dataset, DataLoader

IGNORES = [
    'a', 'an', 'the', 'on', 'in', 'at', 'to', 'behind', 'under', 'during', 'it', 'is'
]

def process_answer(ans):
    ans = ans.lower()
    for word in IGNORES:
        ans = ans.replace(word + ' ', '')
    return ans

class VisualGenome(Dataset):
    def __init__(self, root, stage):
        if not stage in ['train', 'dev', 'test']:
            raise Exception('Unknown stage "%s"' % stage)

        with open(os.path.join(root, 'data.json'), 'r') as f:
            self.data = json.load(f)[stage]

        self.image_feats = h5py.File(os.path.join(root, 'image_feats.hdf5'), 'r')['data']

    def __getitem__(self, index):
        qa = self.data['qas'][index]
        image_id = qa['image_id']
        image = self.data['images'][str(image_id)]
        image_feats = torch.from_numpy(self.image_feats[image['index']])
        question = qa['question']
        answer = process_answer(qa['answer'])
        contexts = random.sample(image['desc'], min(5, len(image['desc'])))
        return image_feats, contexts, question, answer

    def __len__(self):
        return len(self.data['qas'])

def collate_fn(samples):
    image = []
    contexts = []
    question = []
    answer = []
    for img, ctx, q, a in samples:
        image.append(img)
        contexts.append(ctx)
        question.append(q)
        answer.append(a)
    return {
        'image': torch.stack(image),
        'contexts': contexts,
        'question': question,
        'answer': answer
    }

def get_data_loader(root, stage, batch_size):
    return DataLoader(VisualGenome(root, stage), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
