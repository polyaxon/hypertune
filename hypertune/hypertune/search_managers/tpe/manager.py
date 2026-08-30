from typing import Dict, List

from hypertune.search_managers.base import BaseManager
from hypertune.search_managers.tpe.sampler import TPESampler
from polyaxon.schemas import V1TPE


class TPEManager(BaseManager):
    """Generate suggestions with a Tree-structured Parzen Estimator.

    TPE splits completed trials into better and worse groups, fits a probability
    model for each parameter in both groups, and selects candidates that are more
    likely under the better model. It uses random sampling until enough observations
    are available.
    """

    CONFIG = V1TPE

    def __init__(self, config):
        super().__init__(config)
        self.max_iterations = self.config.max_iterations

    def get_suggestions(
        self, configs: List[Dict] = None, metrics: List[float] = None
    ) -> List[Dict]:
        if not self.config.num_runs:
            raise ValueError("This search strategy requires `num_runs`.")
        sampler = TPESampler(
            params=self.config.params,
            optimization=self.config.metric.optimization,
            seed=self.config.seed,
        )
        return sampler.suggest(
            num_suggestions=self.config.num_runs,
            configs=configs,
            metrics=metrics,
        )
