import itertools

from typing import Dict, List

from hypertune.matrix.utils import to_numpy
from hypertune.search_managers.base import BaseManager
from polyaxon.schemas import V1GridSearch


class GridSearchManager(BaseManager):
    """Grid search strategy manager for hyperparameter optimization."""

    CONFIG = V1GridSearch

    def get_suggestions(self, params: List[Dict] = None) -> List[Dict]:
        suggestions = []
        keys = list(self.config.params.keys())
        values = [to_numpy(v) for v in self.config.params.values()]
        for v in itertools.product(*values):
            suggestions.append(dict(zip(keys, v)))

        if self.config.num_runs:
            return suggestions[: self.config.num_runs]
        return suggestions
