from typing import Optional

from numpy.random import RandomState
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection._split import _RepeatedSplits


class RepeatedStratifiedGroupKFold(_RepeatedSplits):
    def __init__(
        self, *, n_splits: int = 5, n_repeats: int = 10, random_state: Optional[RandomState] = None
    ) -> None:
        super().__init__(
            StratifiedGroupKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )
