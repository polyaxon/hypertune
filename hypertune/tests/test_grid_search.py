import pytest

from unittest.mock import patch

from hypertune.search_managers.grid_search.manager import GridSearchManager
from polyaxon._utils.test_utils import BaseTestCase
from polyaxon.schemas import V1GridSearch


@pytest.mark.tuninig_mark
class TestGridSearch(BaseTestCase):
    def test_grid_search_config(self):
        assert GridSearchManager.CONFIG == V1GridSearch

    def test_get_suggestions(self):
        config = V1GridSearch.from_dict(
            {
                "concurrency": 2,
                "numRuns": 10,
                "params": {"feature": {"kind": "choice", "value": [1, 2, 3]}},
            }
        )
        assert len(GridSearchManager(config).get_suggestions()) == 3

        config = V1GridSearch.from_dict(
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
        assert len(GridSearchManager(config).get_suggestions()) == 10

    def test_get_suggestions_calls_to_numpy(self):
        config = V1GridSearch.from_dict(
            {
                "concurrency": 2,
                "numRuns": 10,
                "params": {"feature": {"kind": "choice", "value": [1, 2, 3]}},
            }
        )
        with patch(
            "hypertune.search_managers.grid_search.manager.to_numpy"
        ) as to_numpy_mock:
            GridSearchManager(config).get_suggestions()

        assert to_numpy_mock.call_count == 1

        config = V1GridSearch.from_dict(
            {
                "concurrency": 2,
                "params": {
                    "feature1": {"kind": "choice", "value": [1, 2, 3]},
                    "feature2": {"kind": "logspace", "value": "0.01:0.1:5"},
                },
            }
        )
        with patch(
            "hypertune.search_managers.grid_search.manager.to_numpy"
        ) as to_numpy_mock:
            GridSearchManager(config).get_suggestions()

        assert to_numpy_mock.call_count == 2
