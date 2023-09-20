from typing import Dict, List

from hypertune.search_managers.base import BaseManager
from hypertune.search_managers.bayesian_optimization.optimizer import BOOptimizer
from hypertune.search_managers.random_search.manager import RandomSearchManager
from polyaxon.schemas import V1Bayes, V1RandomSearch


class BayesSearchManager(BaseManager):
    """Bayesian optimization strategy manager for hyperparameter optimization."""

    CONFIG = V1Bayes

    def __init__(self, config):
        super().__init__(config=config)
        self.num_initial_runs = self.config.num_initial_runs
        self.max_iterations = self.config.max_iterations

    def get_suggestions(
        self, configs: List[Dict] = None, metrics: List[float] = None
    ) -> List[Dict]:
        if not configs or not metrics:
            config = V1RandomSearch(
                params=self.config.params,
                num_runs=self.num_initial_runs,
                seed=self.config.seed,
            )
            return RandomSearchManager(config=config).get_suggestions()

        optimizer = BOOptimizer(config=self.config)
        optimizer.add_observations(configs=configs, metrics=metrics)
        suggestion = optimizer.get_suggestion()
        return [suggestion] if suggestion else None
