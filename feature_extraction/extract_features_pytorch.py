import os

from PIL import Image
import numpy as np
from scipy.misc import imresize
from scipy.io import savemat
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


## Settings #############################################################################

# create your model here!
# Example
from torchvision.models import resnet18
model = resnet18.resnet18(pretrained=True)

# if you use custom transformer for preprocessing, define it.
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

network_name = 'ResNet-18'
image_dir = './imagenet_test_images'
use_gpu = True

output_dir = os.path.join('../data/features', network_name)

############################################################################################


class ImageNetDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform

        tmp = sorted(os.listdir(image_dir))
        tmp = [t for t in tmp if 'mat' not in t]
        self.images = tmp

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(path)
        image = np.asarray(image)
        image = imresize(image, (224, 224), interp='bicubic')
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis].repeat(3, axis=2)
        image = image.transpose(2, 0, 1)
        image = image / 255.0
        image = torch.Tensor(image)
        image = self.transform(image)

        return self.images[idx], image


def extract_features(image_dir, model, mean_image, output_dir, transform, layers=None, use_gpu=False):
    # automatically list up all layers (blocks)
    # if you want to get activation within each blocks, specify layer names by your self, for example
    # layers = [block1.conv1, block1.relu1, ...]
    layers = list(dict(model.named_children()).keys())

    dataset = ImageNetDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=False, num_workers=4)

    if use_gpu is True and torch.cuda.is_avairable():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for layer in layers:
            if not os.path.exists(os.path.join(output_dir, layer)):
                os.mkdir(os.path.join(output_dir, layer))

            # set hook
            def _store_feats(layer, inp, output):
                _model_feats.append(output.numpy())
            exec('model.' + layer + '.register_forward_hook(_store_feats)')

            # extract features by batch
            features = []
            image_names = []
            for imnames, images in dataloader:
                _model_feats = []
                features.append(_model_feats[0])
                image_names.extend(imnames)

            # save features for each image
            features = np.concatenate(features)
            for i, image_name in enumerate(image_names):
                feat = features[i:(i+1)]
                save_fname = os.path.join(
                    output_dir, layer, image_name.replace('.JPEG', '.mat'))
                savemat(save_fname, {'feat': feat})


if __name__ == '__main__':
    extract_features(image_dir, model, output_dir, transform, use_gpu)
