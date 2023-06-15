from typing import Callable, Optional, Tuple

from torch.utils.data import Dataset


def get_ctsdiag(
    data_path: str,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Tuple[Dataset, Dataset]:
    from .ctsdiag import CTSDiag

    train_dataset = CTSDiag(
        data_path, split="train", transform=train_transform, target_transform=target_transform
    )
    test_dataset = CTSDiag(
        data_path, split="test", transform=eval_transform, target_transform=target_transform
    )
    return train_dataset, test_dataset


def get_kfold_ctsdiag(
    data_path: str,
    num_splits: int,
    num_repeats: int,
    nth_fold: int,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Tuple[Dataset, Dataset]:
    from torch.utils.data import Subset

    from .ctsdiag import CTSDiag
    from .utils import RepeatedStratifiedGroupKFold

    dataset = CTSDiag(data_path)

    X = dataset.t_filepaths
    y = dataset.labels
    groups = dataset.patients
    cv = RepeatedStratifiedGroupKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=42)

    count = 1
    for train_indices, test_indices in cv.split(X, y, groups):
        if count == nth_fold:
            break
        count += 1

    train_dataset = CTSDiag(
        data_path, transform=train_transform, target_transform=target_transform
    )
    train_dataset = Subset(train_dataset, train_indices)
    eval_dataset = CTSDiag(data_path, transform=eval_transform, target_transform=target_transform)
    test_dataset = Subset(eval_dataset, test_indices)
    return train_dataset, test_dataset


def get_ctssev(
    data_path: str,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Tuple[Dataset, Dataset]:
    from .ctssev import CTSSev

    train_dataset = CTSSev(
        data_path, split="train", transform=train_transform, target_transform=target_transform
    )
    test_dataset = CTSSev(
        data_path, split="test", transform=eval_transform, target_transform=target_transform
    )
    return train_dataset, test_dataset


def get_kfold_ctssev(
    data_path: str,
    num_splits: int,
    num_repeats: int,
    nth_fold: int,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> Tuple[Dataset, Dataset]:
    from torch.utils.data import Subset

    from .ctssev import CTSSev
    from .utils import RepeatedStratifiedGroupKFold

    dataset = CTSSev(data_path)

    X = dataset.t_filepaths
    y = dataset.labels  # it should be `severities` but use `labels` for identical kfold splits
    groups = dataset.patients
    cv = RepeatedStratifiedGroupKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=42)

    count = 1
    for train_indices, test_indices in cv.split(X, y, groups):
        if count == nth_fold:
            break
        count += 1

    train_dataset = CTSSev(data_path, transform=train_transform, target_transform=target_transform)
    train_dataset = Subset(train_dataset, train_indices)
    eval_dataset = CTSSev(data_path, transform=eval_transform, target_transform=target_transform)
    test_dataset = Subset(eval_dataset, test_indices)
    return train_dataset, test_dataset
