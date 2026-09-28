import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np

def randomCrop(image, label, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    label = Image.fromarray(label)
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), np.array(label.crop(random_region)), depth.crop(
        random_region)

class Dataset(data.Dataset):
    
    def __init__(self, datasets, mode='train', transform=None, return_size=True, modality='Modality-Complete'):
        self.return_size = return_size
        if type(datasets) != list:
            datasets = [datasets]
        self.datas_id = []
        self.mode = mode
        self.modality = modality  
        
        # Configure this path with the RGBT_DATA_ROOT environment variable.
        base_path = os.environ.get('RGBT_DATA_ROOT', './data/rgbt_dataset')

        for (i, dataset) in enumerate(datasets):
            if mode == 'train':
                data_dir = '{}/{}'.format(base_path, dataset)
                imgset_path = data_dir + '/train.txt'
            else:
                data_dir = '{}/{}'.format(base_path, dataset)
                imgset_path = data_dir + '/test.txt'

            imgset_file = open(imgset_path)

            for line in imgset_file:
                data = {}
                
                raw_rgb_path = line.strip("\n").split(" ")[0]
                raw_gt_path = line.strip("\n").split(" ")[1]
                
                
                if self.modality == 'Modality-Only-T':
                    file_name = raw_rgb_path.split('/')[-1] 
                    name_no_ext = os.path.splitext(file_name)[0] 
                    
                    data['img_path'] = data_dir + '/black/' + name_no_ext + '.jpg'
                else:
                    data['img_path'] = data_dir + raw_rgb_path

                data['gt_path'] = data_dir + raw_gt_path

                if dataset == 'DUTS-TR':
                    data['split'] = 'Mamba'
                    data['img_path'] = data_dir +'/'+ raw_rgb_path
                    data['gt_path'] = data_dir +'/'+ raw_gt_path
                else:
                    raw_depth_path = line.strip("\n").split(" ")[2]
                    
                    
                    if self.modality == 'Modality-Only-RGB':
                        file_name = raw_depth_path.split('/')[-1]
                        name_no_ext = os.path.splitext(file_name)[0]
                        data['depth_path'] = data_dir + '/black/' + name_no_ext + '.jpg'
                    else:
                        data['depth_path'] = data_dir + raw_depth_path
                    
                    data['split'] = 'Mamba'
                
                data['dataset'] = dataset
                self.datas_id.append(data)
        self.transform = transform

    def __getitem__(self, item):
        
        assert os.path.exists(self.datas_id[item]['img_path']), (
            'Image path does not exist: {}'.format(self.datas_id[item]['img_path']))
        assert os.path.exists(self.datas_id[item]['gt_path']), (
            'GT path does not exist: {}'.format(self.datas_id[item]['gt_path']))
        
        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            pass
        else:
            assert os.path.exists(self.datas_id[item]['depth_path']), (
                'Depth/Thermal path does not exist: {}'.format(self.datas_id[item]['depth_path']))

        image = Image.open(self.datas_id[item]['img_path']).convert('RGB')
        label = Image.open(self.datas_id[item]['gt_path']).convert('L')
        label = np.array(label)

        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            depth = np.zeros((image.size[1], image.size[0], 3))
            depth = Image.fromarray(np.uint8(depth))
        else:
            depth = Image.open(self.datas_id[item]['depth_path']).convert('RGB')

        if label.max() > 0:
            label = label / 255

        w, h = image.size
        size = (h, w)

        sample = {'image': image, 'label': label, 'depth': depth}
        if self.mode == 'train':
            sample['image'], sample['label'], sample['depth'] = randomCrop(sample['image'],sample['label'],sample['depth'])
        
        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            sample['size'] = torch.tensor(size)
        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            sample['depth'] = torch.zeros((3, 448, 448))
        name = self.datas_id[item]['gt_path'].split('/')[-1]
        sample['dataset'] = self.datas_id[item]['dataset']
        sample['split'] = self.datas_id[item]['split']
        sample['name'] = name

        return sample

    def __len__(self):
        return len(self.datas_id)
