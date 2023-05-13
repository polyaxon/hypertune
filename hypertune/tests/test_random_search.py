import pytest

from unittest.mock import patch

from hypertune.search_managers.random_search.manager import RandomSearchManager
from polyaxon.polyflow.matrix import V1RandomSearch
from polyaxon.utils.test_utils import BaseTestCase


@pytest.mark.tuninig_mark
class TestRandomSearch(BaseTestCase):
    def test_random_search_config(self):
        assert RandomSearchManager.CONFIG == V1RandomSearch

    def test_get_suggestions(self):
        config = V1RandomSearch.from_dict(
            {
                "concurrency": 2,
                "numRuns": 10,
                "params": {
                    "feature1": {"kind": "choice", "value": [1, 2]},
                    "feature3": {"kind": "range", "value": [1, 3, 1]},
                },
            }
        )

        assert len(RandomSearchManager(config).get_suggestions()) == 4

        config = V1RandomSearch.from_dict(
            {
                "concurrency": 2,
                "numRuns": 10,
                "params": {
                    "feature1": {"kind": "pchoice", "value": [(1, 0.1), (2, 0.6)]},
                    "feature3": {"kind": "range", "value": [1, 3, 1]},
                },
            }
        )

        assert len(RandomSearchManager(config).get_suggestions()) == 4

        config = V1RandomSearch.from_dict(
            {
                "concurrency": 2,
                "numRuns": 10,
                "params": {
                    "feature1": {"kind": "choice", "value": [1, 2, 3]},
                    "feature2": {"kind": "linspace", "value": [1, 2, 5]},
                    "feature3": {"kind": "range", "value": [1, 5, 1]},
                },
            }
        )
        assert len(RandomSearchManager(config).get_suggestions()) == 10

        config = V1RandomSearch.from_dict(
            {
                "concurrency": 2,
                "numRuns": 10,
                "params": {
                    "feature1": {
                        "kind": "pchoice",
                        "value": [(1, 0.3), (2, 0.3), (3, 0.3)],
                    },
                    "feature2": {"kind": "uniform", "value": [0, 1]},
                    "feature3": {"kind": "qlognormal", "value": [0, 0.5, 0.51]},
                },
            }
        )
        assert len(RandomSearchManager(config).get_suggestions()) == 10

    def test_get_suggestions_calls_sample(self):
        config = V1RandomSearch.from_dict(
            {
                "concurrency": 2,
                "numRuns": 1,
                "params": {
                    "feature1": {"kind": "choice", "value": [1, 2, 3]},
                    "feature2": {"kind": "linspace", "value": [1, 2, 5]},
                    "feature3": {"kind": "range", "value": [1, 5, 1]},
                },
            }
        )
        with patch(
            "hypertune.search_managers.random_search.manager.sample"
        ) as sample_mock:
            RandomSearchManager(config).get_suggestions()

        assert sample_mock.call_count == 3

        config = V1RandomSearch.from_dict(
            {
                "concurrency": 2,
                "numRuns": 1,
                "params": {
                    "feature1": {
                        "kind": "pchoice",
                        "value": [(1, 0.3), (2, 0.3), (3, 0.3)],
                    },
                    "feature2": {"kind": "uniform", "value": [0, 1]},
                    "feature3": {"kind": "qlognormal", "value": [0, 0.5, 0.51]},
                    "feature4": {"kind": "range", "value": [1, 5, 1]},
                },
            }
        )
        with patch(
            "hypertune.search_managers.random_search.manager.sample"
        ) as sample_mock:
            RandomSearchManager(config).get_suggestions()

        assert sample_mock.call_count == 4
