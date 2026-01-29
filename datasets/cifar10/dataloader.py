import os
import numpy as np
import torch
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def _make_tfms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


pwd = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(pwd, "cifar10_train.npz")
test_path = os.path.join(pwd, "cifar10_test.npz")

if not (os.path.exists(train_path) and os.path.exists(test_path)):
    raise FileNotFoundError(
        "CIFAR-10 arrays not found. Run datasets/cifar10/prepare_data.py first."
    )

train_data = np.load(train_path)
test_data = np.load(test_path)

X_train = train_data["X"]
y_train = train_data["y"].astype(np.int64)

X_test = test_data["X"]
y_test = test_data["y"].astype(np.int64)

train_tfms = _make_tfms(train=True)
test_tfms = _make_tfms(train=False)


def _apply_tfms(X, tfms):
    xs = [tfms(x) for x in X]
    return torch.stack(xs).numpy()

def load(indices, category="train"):
    int_indices = indices.astype(np.int64)
    if category == "train":
        X = _apply_tfms(X_train[int_indices], train_tfms)
        return X, y_train[int_indices]
    else:
        X = _apply_tfms(X_test[int_indices], test_tfms)
        return X, y_test[int_indices]
