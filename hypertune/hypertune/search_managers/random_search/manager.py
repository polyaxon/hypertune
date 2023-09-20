import copy

from functools import reduce
from operator import mul
from typing import Dict, List, Optional

from hypertune.matrix.utils import get_length, sample
from hypertune.search_managers.base import BaseManager
from hypertune.search_managers.spec import SuggestionSpec
from hypertune.search_managers.utils import get_random_generator
from polyaxon.schemas import V1RandomSearch


class RandomSearchManager(BaseManager):
    """Random search strategy manager for hyperparameter optimization."""

    CONFIG = V1RandomSearch

    def get_suggestions(self, params: Optional[Dict] = None) -> List[Dict]:
        if not self.config.num_runs:
            raise ValueError("This search strategy requires `num_runs`.")
        suggestions = []
        params = params or {}
        rand_generator = get_random_generator(seed=self.config.seed)
        # Validate number of suggestions and total space
        all_discrete = True
        for v in self.config.params.values():
            if v.is_continuous:
                all_discrete = False
                break
        num_runs = self.config.num_runs
        if all_discrete:
            space = reduce(mul, [get_length(v) for v in self.config.params.values()])
            num_runs = self.config.num_runs if self.config.num_runs <= space else space

        while num_runs > 0:
            suggestion_params = copy.deepcopy(params)
            suggestion_params.update(
                {
                    k: sample(v, rand_generator=rand_generator)
                    for k, v in self.config.params.items()
                }
            )
            suggestion = SuggestionSpec(params=suggestion_params)
            if suggestion not in suggestions:
                suggestions.append(suggestion)
                num_runs -= 1
        return [suggestion.params for suggestion in suggestions]
