import requests, zipfile, io, os
from typing import Optional, Union, Dict, Tuple, List
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
    WeightedRandomSampler,
)
import torchvision.transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def download_data(url: str, name: str) -> None:
    api_url = (
        "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        f"?public_key={url}"
    )
    download_url = requests.get(api_url).json()["href"]
    
    with requests.get(download_url, stream=True) as r, open(f"{name}.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    os.makedirs(name, exist_ok=True)
    with zipfile.ZipFile(f"{name}.zip", "r") as z:
        z.extractall(name)


def _fft_amp_channel(img: Image.Image, eps: float = 1e-8) -> np.ndarray:
    x = np.array(img.convert("L"), dtype=np.float32) / 255.0
    fft = np.fft.fftshift(np.fft.fft2(x))
    h, w = fft.shape
    D = 16
    fft[h//2 - h//D:h//2 + h//D, w//2 - w//D:w//2 + w//D] = 0.
    amp = np.abs(np.fft.ifft2(fft))
    return (amp - amp.mean()) / (amp.std() + eps)


def build_transforms(img_size: int = 224, train: bool = True) -> T.Compose:
    if train:
        aug = [
            T.RandomResizedCrop((img_size, img_size), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(45),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
            T.ColorJitter(0.2, 0.2, 0.15, 0.05),
        ]
    else:
        aug = [T.Resize((img_size, img_size))]
    aug += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(aug)


def _stratified_indices(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    val_idx = []
    for _type, grp in df.groupby("fake_type"):
        n_val = max(1, int(len(grp) * val_frac))
        val_idx.extend(rng.choice(grp.index, size=n_val, replace=False))
    train_idx = df.index.difference(val_idx)
    return train_idx.tolist(), val_idx


def _balanced_sampler(subset: Subset, type_idx_full: List[int]) -> WeightedRandomSampler:
    type_list = [type_idx_full[i] for i in subset.indices]
    freq = Counter(type_list)
    weights = torch.DoubleTensor([1.0 / freq[t] for t in type_list])
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class FaceDeepfakeDataset(Dataset):
    def __init__(self,
                 csv_file: Union[str, os.PathLike],
                 root_dir: Union[str, os.PathLike] = "",
                 rgb_transform: Optional[T.Compose] = None,
                 img_size: int = 224,
                 is_test: bool = False):
        super().__init__()
        self.is_test = is_test
        self.root_dir = os.fspath(root_dir)
        data_path = os.path.join(self.root_dir, csv_file)
        self.df = pd.read_csv(data_path)
        self.rgb_transform = rgb_transform or build_transforms(img_size, train=not is_test)
        self.freq_resize = T.Resize((img_size, img_size))
        self.img_size = img_size
        if not self.is_test:
            types = sorted(self.df["fake_type"].unique())
            self.type2idx = {t: i for i, t in enumerate(types)}
            self.idx2type = {v: k for k, v in self.type2idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["crop_path"][int(self.is_test):])
        img = Image.open(img_path).convert("RGB")

        # RGB branch
        rgb = self.rgb_transform(img)

        # Frequency branch
        freq_np = _fft_amp_channel(img)
        freq = torch.from_numpy(freq_np).unsqueeze(0)
        freq = self.freq_resize(freq)

        if self.is_test:
            return rgb, freq
        
        label_bin = torch.tensor(row["label"], dtype=torch.float32)
        type_idx = torch.tensor(self.type2idx[row["fake_type"]], dtype=torch.long)

        return rgb, freq, label_bin, type_idx


def build_dataloaders(csv_file_train: Union[str, os.PathLike],
                      root_dir_train: Union[str, os.PathLike],
                      csv_file_test: Union[str, os.PathLike],
                      root_dir_test: Union[str, os.PathLike],
                      img_size: int = 224,
                      batch_size: int = 32,
                      val_frac: float = 0.2,
                      seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:

    full_ds = FaceDeepfakeDataset(csv_file=csv_file_train,
                                  root_dir=root_dir_train,
                                  img_size=img_size,
                                  is_test=False)

    # stratified split
    train_idx, val_idx = _stratified_indices(full_ds.df, val_frac, seed)
    train_set = Subset(full_ds, train_idx)
    val_set = Subset(full_ds, val_idx)

    # replace RGB transform for validation
    if isinstance(val_set.dataset, FaceDeepfakeDataset):
        val_set.dataset.rgb_transform = build_transforms(img_size, train=False)

    # balanced sampler on fake_type
    type_column = [full_ds.type2idx[t] for t in full_ds.df["fake_type"].tolist()]
    sampler = _balanced_sampler(train_set, type_column)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              sampler=sampler)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False)

    test_ds = FaceDeepfakeDataset(csv_file=csv_file_test,
                                  root_dir=root_dir_test,
                                  img_size=img_size,
                                  is_test=True)
    
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False)
    
    return train_loader, val_loader, test_loader