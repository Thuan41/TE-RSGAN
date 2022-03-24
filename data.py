import torchvision.transforms.functional as F
import torch
from os import listdir
from os.path import join
from torch.utils.data import Dataset
from skimage.feature import canny
from PIL import Image, ImageFilter
from torchvision.transforms import Compose, ToTensor, ToPILImage, Resize, CenterCrop, Normalize

# Normalization parameters for pre-trained PyTorch models
#mean = torch.tensor([0.485, 0.456, 0.406])
#std = torch.tensor([0.229, 0.224, 0.225])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class TrainDataset(Dataset):
    def __init__(self, data_path, mode, lr_size):  # lr_size must be valid
        super(TrainDataset, self).__init__()
        self.thermal_filenames = get_image(data_path, 'lwir', mode)
        self.rgb_filenames = get_image(data_path, 'visible', mode)
        self.lr_transform = Compose(
            [Resize(lr_size, interpolation=Image.BICUBIC), ToTensor()])  # @Thuan: add normalize
        self.hr_transform = Compose([ToTensor()])  # @Thuan:
        self.edge_transform = Compose(
            [Resize(lr_size, interpolation=Image.BICUBIC), ToTensor()])  # @Thuan: add normalize

    def __getitem__(self, index):
        hr_image = Image.open(self.thermal_filenames[index])
        rgb_image = Image.open(self.rgb_filenames[index])
        rgb_image = rgb_image.convert("L")  # Convert to  gray scale
        edge_map = rgb_image.filter(ImageFilter.FIND_EDGES)
        edge_map = self.edge_transform(edge_map)

        lr_image = self.lr_transform(hr_image)
        target = self.hr_transform(hr_image)
        return lr_image, edge_map, target

    def __len__(self):
        return len(self.thermal_filenames)


def display_transform():
    return Compose([
        ToPILImage(),
        ToTensor()
    ])


# data_path = '/kaist-cvpr15/images'
def get_image(data_path, data_type, mode, max_set=12, train_set=6):
    '''
    data_type
    - 'lwir' -> thermal image
    - 'visible' -> rgb image
    mode
    - 'train'
    - 'val'
    '''
    if mode == 'train':
        data_set = ["set{:02d}".format(i) for i in range(train_set)]
    elif mode == 'val':
        data_set = ["set{:02d}".format(i) for i in range(train_set, max_set)]

    data_list = []
    for setxx in listdir(data_path):
        if setxx in data_set:
            for Vxxx in listdir(join(data_path, setxx)):
                for x in listdir(join(data_path, setxx, Vxxx, data_type)):
                    if is_image_file(x):
                        data_list.append(
                            join(data_path, setxx, Vxxx, data_type, x))
    return data_list
