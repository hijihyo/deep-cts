import os
from glob import glob
from typing import Any, Callable, Optional, Tuple, Union

from PIL import Image as pil
from PIL.Image import Image
from torch.utils.data import Dataset


class CTSDiag(Dataset):
    """PyTorch Dataset for CTS diagnosis"""

    DIRECTORY = "CTSDiag"
    NUM_TRAIN_TOTAL = 527
    NUM_TEST_TOTAL = 100
    NUM_TOTAL = 627

    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(CTSDiag, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        assert self.split in {"train", "test", None}
        if self.split == "train":
            tmask_filepaths = glob(
                os.path.join(self.root, self.DIRECTORY, "train", "*/*/thenar-mask-*.tiff")
            )
            t_filepaths = list(map(lambda f: f.replace("mask-", ""), tmask_filepaths))
            assert len(t_filepaths) == self.NUM_TRAIN_TOTAL
        elif self.split == "test":
            tmask_filepaths = glob(
                os.path.join(self.root, self.DIRECTORY, "test", "*/*/thenar-mask-*.tiff")
            )
            t_filepaths = list(map(lambda f: f.replace("mask-", ""), tmask_filepaths))
            assert len(t_filepaths) == self.NUM_TEST_TOTAL
        else:
            tmask_filepaths = glob(
                os.path.join(self.root, self.DIRECTORY, "*/*/*/thenar-mask-*.tiff")
            )
            t_filepaths = list(map(lambda f: f.replace("mask-", ""), tmask_filepaths))
            assert len(t_filepaths) == self.NUM_TOTAL

        t_filepaths = list(sorted(t_filepaths, key=lambda f: self._get_sort_key(f)))
        assert all(os.path.exists(self._get_ht_filepath(t)) for t in t_filepaths)

        self.t_filepaths = t_filepaths
        self.patients = [self._get_patient(t) for t in self.t_filepaths]
        self.severities = [p.split("-")[0] for p in self.patients]
        self.labels = [0 if s == "normal" else 1 for s in self.severities]

    def __getitem__(
        self, index: int
    ) -> Tuple[Union[Image, Any], Union[Image, Any], Union[int, Any]]:
        t_filepath = self.t_filepaths[index]
        ht_filepath = self._get_ht_filepath(t_filepath)
        label = self.labels[index]

        t_img = pil.open(t_filepath).convert("L")
        ht_img = pil.open(ht_filepath).convert("L")

        if self.transform:
            t_img = self.transform(t_img)
            ht_img = self.transform(ht_img)

        if self.target_transform:
            label = self.target_transform(label)

        return {"inputs1": t_img, "inputs2": ht_img, "labels": label}

    def __len__(self) -> None:
        return len(self.t_filepaths)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} Dataset"

    def _get_sort_key(self, filepath: str) -> Tuple[str, str, str, str, int]:
        parent, filename = os.path.split(filepath)
        parent, patient = os.path.split(parent)
        parent, severity = os.path.split(parent)
        _, split = os.path.split(parent)

        splits = filename.replace(".tiff", "").split("-")
        datano = int(splits[-1])
        muscle = splits[0]
        return (split, severity, patient, muscle, datano)

    def _get_ht_filepath(self, t_filepath: str) -> str:
        return t_filepath.replace("thenar-", "hypothenar-")

    def _get_patient(self, filepath: str) -> str:
        parent, _ = os.path.split(filepath)
        _, patient = os.path.split(parent)
        return patient


if __name__ == "__main__":
    import argparse

    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    dataset = CTSDiag(args.root, args.split)
    print(dataset)
    for _ in tqdm(dataset, desc="Loading data", total=len(dataset)):
        continue

    t_img = dataset[args.index]["inputs1"]
    ht_img = dataset[args.index]["inputs2"]
    label = dataset[args.index]["labels"]
    print("Data Example")
    print("  Thenar image:", t_img.size, t_img.mode)
    print("  Hypothenar image:", ht_img.size, ht_img.mode)
    print("  Label:", label, (["normal", "abnormal"])[label])
    print("  Patient:", dataset.patients[args.index])
