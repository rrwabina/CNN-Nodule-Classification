import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image

class NoduleDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.images_dir = data_dir/'images'
        self.labels_dir = data_dir/'labels'

        self.train_images_dir = self.images_dir   / 'train'
        self.val_images_dir   = self.images_dir   / 'val'
        self.test_images_dir  = self.images_dir   / 'test'

        self.train_labels_file = self.labels_dir  / 'trainlabels.txt'
        self.val_labels_file   = self.labels_dir  / 'vallabels.txt'
        self.test_labels_file  = self.labels_dir  / 'testlabels.txt'

        self.train_data = self._load_data(self.train_images_dir, self.train_labels_file)
        self.val_data   = self._load_data(self.val_images_dir,   self.val_labels_file)
        self.test_data  = self._load_data(self.test_images_dir,  self.test_labels_file)

    def __len__(self):
        return len(self.train_data) + len(self.val_data) + len(self.test_data)

    def __getitem__(self, index):
        if index < len(self.train_data):
            return self.train_data[index]
        elif index < len(self.train_data) + len(self.val_data):
            return self.val_data[index - len(self.train_data)]
        else:
            return self.test_data[index - len(self.train_data) - len(self.val_data)]
    
    def _load_data(self, images_dir, labels_file):
        with open(labels_file, 'r', newline = '\r\n') as f:
            labels = f.readlines()
        data = []
        for i, label in enumerate(labels):
            label = int(label.strip().split()[-1])
            image_path = images_dir / f'{i}.jpg'
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            data.append((image, label))
        return data

data_dir = Path('../data/')
dataset  = NoduleDataset(data_dir)


class NoduleDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.augment = transforms.Compose([
            transforms.Resize(50),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.transform = transforms.Compose([
            transforms.Resize(50),
            transforms.RandomCrop(32, padding = 2),
            transforms.ToTensor()])
        
        self.images_dir = data_dir / 'images'
        self.labels_dir = data_dir / 'labels'

        self.train_images_dir = self.images_dir 
        self.val_images_dir   = self.images_dir  
        self.test_images_dir  = self.images_dir  

        self.train_labels_file = self.labels_dir  / 'trainlabels.txt'
        self.val_labels_file   = self.labels_dir  / 'vallabels.txt'
        self.test_labels_file  = self.labels_dir  / 'testlabels.txt'

        self.train_data = self._load_data(self.train_images_dir, self.train_labels_file)
        self.val_data   = self._load_data(self.val_images_dir, self.val_labels_file)
        self.test_data  = self._load_data(self.test_images_dir, self.test_labels_file)

    def __getitem__(self, index):
        if index < len(self.train_data):
            images_dir = self.train_images_dir
            data = self.train_data
        elif index < len(self.train_data) + len(self.val_data):
            images_dir = self.val_images_dir
            data = self.val_data
            index -= len(self.train_data)
        else:
            images_dir = self.test_images_dir
            data = self.test_data
            index -= (len(self.train_data) + len(self.val_data))

        img_path = images_dir / data[index][0]
        with open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        label = data[index][1]
        return self.transform(image), label

    def __len__(self):
        return len(self.train_data) + len(self.val_data) + len(self.test_data)

    def _load_data(self, images_dir, labels_file):
        with open(labels_file, 'r') as f:
            lines = f.readlines()

        data = []
        for line in lines[1:]:
            filename, label = line.strip().split()
            filename = filename 
            label = int(label)
            data.append((filename, label))
        return data

    def get_datasets(self):
        train_dataset = Subset(self, range(len(self.train_data)))
        test_dataset  = Subset(self, range(len(self.train_data),  len(self.train_data) + len(self.test_data)))
        valid_dataset = Subset(self, range(len(self.train_data) + len(self.test_data),   len(self)))
        return train_dataset, test_dataset, valid_dataset

def GET_NODULEDATASET():
    train_indices = list(range(0, len(dataset.train_data)))
    valid_indices = list(range(len(dataset.train_data),  len(dataset.train_data) + len(dataset.val_data)))
    test_indices  = list(range(len(dataset.train_data) + len(dataset.val_data), len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset  = Subset(dataset, test_indices)
    return train_dataset, valid_dataset, test_dataset


# class NoduleDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir        
#         self.transform = transforms.Compose([
#             transforms.Resize((50, 50)),
#             transforms.RandomCrop(32, padding = 2),
#             transforms.RandomRotation(90),
#             transforms.CenterCrop(40),
#             transforms.ToTensor(), 
#             transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                                  std  = [0.229, 0.224, 0.225])])
        
#         self.images_dir = data_dir / 'images'
#         self.labels_dir = data_dir / 'labels'

#         self.train_images_dir = self.images_dir 
#         self.val_images_dir   = self.images_dir  
#         self.test_images_dir  = self.images_dir  

#         self.train_labels_file = self.labels_dir  / 'trainlabels.txt'
#         self.val_labels_file   = self.labels_dir  / 'vallabels.txt'
#         self.test_labels_file  = self.labels_dir  / 'testlabels.txt'

#         self.train_data = self._load_data(self.train_images_dir, self.train_labels_file)
#         self.val_data   = self._load_data(self.val_images_dir, self.val_labels_file)
#         self.test_data  = self._load_data(self.test_images_dir, self.test_labels_file)

#     def __getitem__(self, index):
#         if index < len(self.train_data):
#             images_dir = self.train_images_dir
#             data = self.train_data
#         elif index < len(self.train_data) + len(self.val_data):
#             images_dir = self.val_images_dir
#             data = self.val_data
#             index -= len(self.train_data)
#         else:
#             images_dir = self.test_images_dir
#             data = self.test_data
#             index -= (len(self.train_data) + len(self.val_data))

#         img_path = images_dir / data[index][0]
#         with open(img_path, 'rb') as f:
#             image = Image.open(f).convert('RGB')

#         label = data[index][1]
#         return self.transform(image), label

#     def __len__(self):
#         return len(self.train_data) + len(self.val_data) + len(self.test_data)

#     def _load_data(self, images_dir, labels_file):
#         with open(labels_file, 'r') as f:
#             lines = f.readlines()

#         data = []
#         for line in lines[1:]:
#             filename, label = line.strip().split()
#             filename = filename 
#             label = int(label)
#             data.append((filename, label))
#         return data

#     def get_datasets(self):
#         train_dataset = Subset(self, range(len(self.train_data)))
#         test_dataset  = Subset(self, range(len(self.train_data),  len(self.train_data) + len(self.test_data)))
#         valid_dataset = Subset(self, range(len(self.train_data) + len(self.test_data),   len(self)))
#         return train_dataset, test_dataset, valid_dataset

# def GET_NODULEDATASET():
#     train_indices = list(range(0, len(dataset.train_data)))
#     valid_indices = list(range(len(dataset.train_data),  len(dataset.train_data) + len(dataset.val_data)))
#     test_indices  = list(range(len(dataset.train_data) + len(dataset.val_data), len(dataset)))

#     train_dataset = Subset(dataset, train_indices)
#     valid_dataset = Subset(dataset, valid_indices)
#     test_dataset  = Subset(dataset, test_indices)
#     return train_dataset, valid_dataset, test_dataset

# data_dir = Path('D:/nodule/data/')
# dataset  = NoduleDataset(data_dir)
# train_dataset, valid_dataset, test_dataset = GET_NODULEDATASET()

# train_classes = [label for _, label in train_dataset]
# class_count = Counter(train_classes)
# class_weights = torch.Tensor([len(train_classes)/c for c in pd.Series(class_count).sort_index().values])
# class_samples = [0] * len(class_weights)

# for _, label in train_dataset:
#     class_samples[label] += 1
# weights = [class_weights[label] / class_samples[label] for _, label in train_dataset]
# sampler = WeightedRandomSampler(weights = weights, num_samples = len(weights), replacement = True)

# train_loader  = DataLoader(train_dataset, batch_size = 32, sampler = sampler)
# valid_loader  = DataLoader(valid_dataset, batch_size = 32, shuffle = True )
# test_loader   = DataLoader(test_dataset,  batch_size = 32, shuffle = False)
