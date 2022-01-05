from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import csv
import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Histology_data(Dataset):
    def __init__(self, root):
        self.image_path_list = []
        image_name_list = os.listdir(os.path.join(root, 'train'))
        image_name_list.sort()
        self.transform = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
                p = 0.5
            ),
            transforms.RandomGrayscale(p=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.GaussianBlur((23, 23), (0.1, 2.0))],
                p = 0.4
            ),
            Solarization(0.4),
            transforms.RandomResizedCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        for image_name in image_name_list:
            image_path = os.path.join(root, 'train', image_name)
            self.image_path_list.append(image_path)
    
    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path).convert('RGB')

        return self.transform(image), self.transform(image)
    
    def __len__(self):
        return len(self.image_path_list)

class query_dataset(Dataset):
    def __init__(self, data_root, image_size, mode):
        self.image_path_list = []
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        if mode == 'val':
            query_csv = np.loadtxt(os.path.join(data_root, 'validation_ground_truth.csv'), delimiter=',', dtype=np.str)[1:]
            for data in query_csv:
                label = data[1]
                image_query = data[0].split('_')
                image1, image2 = image_query[0] + '.png', image_query[1] + '.png'
                image1_path = os.path.join(data_root, 'train', image1)
                image2_path = os.path.join(data_root, 'train', image2)
                image_pair = [image1_path, image2_path]
                self.image_path_list.append([image_pair, int(label)])

        elif mode == 'test':
            query_csv = np.loadtxt(os.path.join(data_root, 'queries.csv'), delimiter=',', dtype=np.str)
            for data in query_csv:
                image1, image2 = data[0], data[1]
                public_name = image1.split('.')[0] + '_' + image2.split('.')[0]
                image1_path = os.path.join(data_root, 'test', image1)
                image2_path = os.path.join(data_root, 'test', image2)
                image_pair = [image1_path, image2_path]
                self.image_path_list.append([image_pair, public_name])
            
    
    def __getitem__(self, index):
        if self.mode == 'val':
            image_pair, label = self.image_path_list[index]
            
            image1, image2 = Image.open(image_pair[0]).convert('RGB'), Image.open(image_pair[1]).convert('RGB')
            image1, image2 = self.transform(image1), self.transform(image2)
            return image1, image2, label
        elif self.mode == 'test':
            image_pair, public_name = self.image_path_list[index]
            
            image1, image2 = Image.open(image_pair[0]).convert('RGB'), Image.open(image_pair[1]).convert('RGB')
            image1, image2 = self.transform(image1), self.transform(image2)
            return image1, image2, public_name

        

    def __len__(self):
        return len(self.image_path_list)
            
class Histology_testdata(Dataset):
    def __init__(self, root, image_size):
        self.image_path_list = []
        image_name_list = os.listdir(os.path.join(root, 'test'))
        image_name_list.sort()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        for image_name in image_name_list:
            image_path = os.path.join(root, 'test', image_name)
            self.image_path_list.append(image_path)
    
    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path).convert('RGB')
        pid = index
        return self.transform(image), pid, image_path
    
    def __len__(self):
        return len(self.image_path_list)
