import torch
import functools
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import json
import pickle
from pathlib import Path, PurePosixPath
base_path = Path(__file__).absolute().parents[1].absolute()
import random
from models.utils.simple_tokenizer import SimpleTokenizer

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


def _relative_caption_key(img_path, dataset_dir):
    parts = PurePosixPath(Path(img_path).as_posix()).parts
    if dataset_dir not in parts:
        return None
    anchor = max(idx for idx, part in enumerate(parts) if part == dataset_dir)
    suffix = parts[anchor + 1:]
    return "/".join(("data", dataset_dir, *suffix))


def get_caption(cap_train, img_path, dataset_dir):
    normalized = Path(img_path).as_posix()
    candidates = [normalized]

    try:
        repo_relative = Path(normalized).resolve().relative_to(base_path.resolve())
        candidates.append(repo_relative.as_posix())
    except Exception:
        pass

    remapped = _relative_caption_key(normalized, dataset_dir)
    if remapped is not None:
        candidates.append(remapped)

    for key in candidates:
        if key in cap_train:
            return cap_train[key][0]

    raise KeyError(
        f"Caption not found for '{img_path}'. Tried keys: {candidates}"
    )


def _resolve_caption_path(caption_path):
    path = Path(caption_path)
    return path if path.is_absolute() else base_path / path


class ImageDatasetClipPRCCTrain(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, caption_path='data/captions/prcc.json'):
        self.dataset = dataset
        self.transform = transform
        with open(_resolve_caption_path(caption_path)) as f:
            self.cap_train = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        caption = get_caption(self.cap_train, img_path, 'prcc')
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, caption


class ImageDatasetClipLTCC(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        with open(base_path / 'data' / 'captions' / f'ltcc.json') as f:
            self.cap_train = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        caption = get_caption(self.cap_train, img_path, 'LTCC_ReID')
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, caption


class ImageDatasetClipVCTrain(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = SimpleTokenizer()
        with open(base_path / 'data' / 'captions' / f'vcclothes.json') as f:
            self.cap_train = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        caption = get_caption(self.cap_train, img_path, 'VC-Clothes')
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, caption


class ImageDatasetLastTrain(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = SimpleTokenizer()
        with open(base_path / 'data' / 'captions' / f'last.json') as f:
            self.cap_train = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        caption = get_caption(self.cap_train, img_path, 'last')
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, caption


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, clothes_id = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, clothes_id, img_path

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

