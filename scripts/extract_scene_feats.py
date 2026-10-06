"""
Extract the pooled scene feature s = GAP(F_scene) (paper Eq. 42-43) once, offline, for every image.

The baseline pipeline only has pre-extracted bottom-up features (no image loader), so the scene CNN is run
here as a frozen feature extractor and its 2048-d global-average-pooled output is stored as
<out_dir>/<image_id>.npy.  GAS-Net then trains the scene classifier head + scene attention on top of it
(--input_scene_dir).  Default backbone: torchvision ResNet-101 (ImageNet).  A scene-specific backbone
(e.g. a Places365 ResNet) can be used through --weights.

    python scripts/extract_scene_feats.py --input_json data/cocotalk_attr.json \
        --image_root /path/to/coco_images --out_dir data/cocobu_scene_2048
(image path = <image_root>/<file_path from the json>, e.g. val2014/COCO_val2014_000000391895.jpg)
"""
from __future__ import print_function

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image


class ImgSet(torch.utils.data.Dataset):
    def __init__(self, images, root, out_dir):
        self.items = [(im['id'], os.path.join(root, im['file_path'])) for im in images
                      if not os.path.isfile(os.path.join(out_dir, str(im['id']) + '.npy'))]   # resumable
        self.tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_id, path = self.items[i]
        return img_id, self.tf(Image.open(path).convert('RGB'))


def main(args):
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    try:
        net = torchvision.models.resnet101(weights=None if args.weights else torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
    except AttributeError:                      # older torchvision
        net = torchvision.models.resnet101(pretrained=not args.weights)
    if args.weights:
        net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    net.fc = nn.Identity()                      # output of the global average pool: [B, 2048]
    net = net.cuda().eval()

    images = json.load(open(args.input_json))['images']
    ds = ImgSet(images, args.image_root, args.out_dir)
    print('%d images to process' % len(ds))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    with torch.no_grad():
        for n, (ids, x) in enumerate(dl):
            feats = net(x.cuda()).cpu().numpy().astype('float32')
            for img_id, f in zip(ids.tolist(), feats):
                np.save(os.path.join(args.out_dir, str(img_id) + '.npy'), f)
            if n % 50 == 0:
                print('batch %d / %d' % (n, len(dl)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_json', default='data/cocotalk_attr.json')
    p.add_argument('--image_root', required=True)
    p.add_argument('--out_dir', default='data/cocobu_scene_2048')
    p.add_argument('--weights', default='', help='optional state_dict (.pth) of a ResNet-101 to use instead of ImageNet weights')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    main(p.parse_args())
