from typing import Dict, List

from hypertune.search_managers.base import BaseManager
from hypertune.search_managers.tpe import TPESampler
from polyaxon.schemas import V1Hyperopt


class HyperoptManager(BaseManager):
    """Native TPE search manager using the legacy Hyperopt configuration name."""

    CONFIG = V1Hyperopt

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
