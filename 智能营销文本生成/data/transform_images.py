
import os
import sys
import torch.nn as nn
from torchvision import models, transforms
import torch
from PIL import Image
from torch.autograd import Variable

sys.path.append('..')

from data_utils import write_samples

def extractor(img_path, net):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    img = transform(img)
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    y = net(x).cpu()
    y = y.view(1, -1)
    y = y.data.numpy().tolist()
    return y

if __name__ == "__main__":
    device = torch.device('cpu')
    resnet = models.resnet34(pretrained=True).to(device)
    features = list(resnet.children())[:-2]
    model = nn.Sequential(*features)
    for param in resnet.parameters():
        param.requires_grad = False
    mappings = []
    img_folder = '../files/img'
    img_files = os.listdir(img_folder)
    count = 0
    for img_file in img_files:
        count += 1
        if count % 500 == 0:
            print(f'processing file no.{count}')
        img = os.path.join(img_folder, img_file)
        img_vec = extractor(img, resnet)
        mapping = img_file + '\t' + ' '.join(map(lambda x: str(x), img_vec))
        mappings.append(mapping)
    write_samples(mappings, '../files/img.txt', opt='a')


