import os
import json
import h5py
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import argparse

parser = argparse.ArgumentParser(description='Extract image features for VisualGenome dataset')
parser.add_argument('--root', required=True, help='root directory of dataset')
parser.add_argument('--cuda', action='store_true', help='enable cuda')
parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default: 16)')
parser.add_argument('--image_size', default=320, type=int, help='resized image size (default: 320)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available and args.cuda
device = torch.device('cuda' if use_cuda else 'cpu')

class VisualGenomeImages(torch.utils.data.Dataset):
    def __init__(self):
        with open(os.path.join(args.root, 'image_data.json'), 'r') as f:
            self.image_data = json.load(f)
            self.image_data = [{
                'id': item['image_id'],
                'path': os.path.join(args.root, '/'.join(item['url'].split('/')[-2:]))
            } for item in self.image_data]

        self.preprocess = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img = Image.open(self.image_data[index]['path']).convert('RGB')
        img = img.resize((args.image_size, args.image_size))
        return self.preprocess(img)

    def __len__(self):
        return len(self.image_data)

def main():
    model = torchvision.models.resnet152(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-2])
    model.cuda()
    model.eval()

    hdf5_path = os.path.join(args.root, 'image_feats.hdf5')
    hdf5_file = h5py.File(hdf5_path, 'w')

    dataset = VisualGenomeImages()
    image_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    output = model(torch.ones(1, 3, args.image_size, args.image_size).to(device))
    output_shape = (len(dataset.image_data), output.size(1), output.size(2), output.size(3))
    print(output_shape)
    hdf5_data = hdf5_file.create_dataset('data', output_shape, dtype='f')

    idx = 0
    for images in tqdm(image_loader):
        batch_size = images.size(0)
        output = model(images.to(device))
        hdf5_data[idx:idx+batch_size] = output.cpu().detach().numpy()
        idx += batch_size

    hdf5_file.close()

if __name__ == '__main__':
    main()
