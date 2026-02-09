
import os
import glob
import time
import math

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
       
class MultilabelDataset(Dataset):
    def __init__(self, file_paths, img_transform):
        super(Dataset).__init__()
        self.file_paths = file_paths 
        self.preprocess = img_transform     
        
    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])
        # data = {
        #     'correct_multi_labels':correct_multi_labels,
        #     'image':image,
        #     'original_label':original_label,
        #     'unclear_multi_labels':unclear_multi_labels
        # }
        img, olabel, mlabel = data['image'], data['original_label'], torch.cat((data['correct_multi_labels'], data['unclear_multi_labels']), dim=0)
        
        img = img.to(torch.float32) / 255.0 # unit8 -> float32
        img = self.preprocess(img)


        mlabel = mlabel.to(torch.int64)       # int32 -> int64
        size = mlabel.shape[0]
        if size < 10:
            padding = torch.full((10 - size,), -1, dtype=mlabel.dtype)
            mlabel = torch.cat((mlabel, padding))
        elif size > 10:
            mlabel = mlabel[:10]
        else:
            pass

        # mlabel = torch.squeeze(mlabel)        # [batch_size, 1] -> [batch_size]
        print(mlabel.shape)
        
        return img, olabel, mlabel
                
    def __len__(self):
        return len(self.file_paths)
                    
if __name__ == "__main__":

    device = torch.device('cuda:0')

    data_dir = "/media/data_cifs/pfeng2/Harmoization/datasets/imagenet_multi_label"
    file_paths = glob.glob(os.path.join(data_dir, '*.pth')) 
    # print(file_paths)
    print(file_paths[0])

    model = timm.create_model('resnet50', pretrained=True, num_classes=1000).to(device)
    model.eval()
    
    data_config = timm.data.resolve_model_data_config(model)
    img_transform = create_transform(**data_config)
    img_transform = transforms.Compose([transforms.ToPILImage(), img_transform])
    print(img_transform)
    
    dataset = MultilabelDataset(file_paths, img_transform)
    dataloader = DataLoader(dataset, batch_size=3, num_workers=4, pin_memory=True)
    start = time.time()
    cnt = 0
    for imgs, olabels, mlabels in dataloader:
        # imgs, hmps, labels = preprocess(imgs, hmps, labels)
        
        if cnt == 0:
            print(imgs.shape, olabels.shape, mlabels.shape)
            print(imgs[0].max(), imgs[0].min())
            print(imgs[0].dtype, mlabels[0].dtype)
        cnt += 1
        break
        
    end = time.time()
    print(end-start) 
    

    
        