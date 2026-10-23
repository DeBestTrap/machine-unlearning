import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
from split_support import configured_manifest
from dataset_splits import fingerprint_cifar


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2430, 0.2610)


def _make_tfms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])


def select_numpy(ds, indices, *, use_transforms=False, channels_first=False):
    idx = np.asarray(indices, dtype=np.int64)

    if not use_transforms and hasattr(ds, "data"):
        Xsrc = ds.data
        X = (Xsrc[idx].numpy() if isinstance(Xsrc, torch.Tensor)
             else np.asarray(Xsrc)[idx])

        ysrc = getattr(ds, "targets", getattr(ds, "labels", None))
        if ysrc is None:
            y = np.asarray([ds[i][1] for i in idx])
        else:
            y = (ysrc.numpy()[idx] if isinstance(ysrc, torch.Tensor)
                 else np.asarray(ysrc)[idx])
    else:
        # Go through __getitem__ to apply transforms; then convert to NumPy
        xs, ys = zip(*(ds[i] for i in idx))
        X = (torch.stack(xs).numpy() if torch.is_tensor(xs[0])
             else np.stack([np.array(x) for x in xs]))
        y = (torch.stack([torch.as_tensor(v) for v in ys]).numpy()
             if torch.is_tensor(ys[0]) else np.asarray(ys))

    if channels_first and X.ndim == 4 and X.shape[-1] in (1, 3):
        X = np.transpose(X, (0, 3, 1, 2))
    return X, y

def get_cifar10(
    data_dir: str,
    image_size: int = 224,
):
    download = not (Path(data_dir) / datasets.CIFAR10.base_folder).exists()
    train_set = datasets.CIFAR10(root=data_dir, train=True, transform=_make_tfms(image_size, True), download=download)
    val_set = datasets.CIFAR10(root=data_dir, train=True, transform=_make_tfms(image_size, False), download=download)

    test_set = datasets.CIFAR10(root=data_dir, train=False, transform=_make_tfms(image_size, False), download=download)

    # train_X = np.asarray(train_set.data)
    # train_y = np.asarray(train_set.targets)

    # val_X = np.asarray(val_set.data)
    # val_y = np.asarray(val_set.targets)

    # return train_X, train_y, val_X, val_y
    return train_set, val_set, test_set

# train_X, train_y, val_X, val_y = get_cifar10(".")
train_set, val_set, test_set = get_cifar10(".")
manifest = configured_manifest(
    Path(__file__).with_name("datasetfile"),
    dataset_id="cifar10", source_size=len(train_set),
    source_sha256=fingerprint_cifar(train_set.data, train_set.targets),
)
train_indices = np.asarray(manifest["train_indices"], dtype=np.int64)
val_indices = np.asarray(manifest["val_indices"], dtype=np.int64)


def load(indices, category="train"):
    indices = np.asarray(indices, dtype=np.int64)
    if category == "train":
        return select_numpy(train_set, train_indices[indices], use_transforms=True)
    if category in ("val", "validation"):
        return select_numpy(val_set, val_indices[indices], use_transforms=True)
    if category == "test":
        return select_numpy(test_set, indices, use_transforms=True)
    raise ValueError(f"Unknown dataset category: {category}")
