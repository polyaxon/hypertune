import math
import pytest

from hypertune.search_managers.hyperopt.manager import HyperoptManager
from polyaxon._utils.test_utils import BaseTestCase
from polyaxon.schemas import V1Hyperopt


@pytest.mark.tuninig_mark
class TestHyperoptSearch(BaseTestCase):
    def test_hyperopt_search_config(self):
        assert HyperoptManager.CONFIG == V1Hyperopt

    def test_search_space(self):
        config = V1Hyperopt.from_dict(
            {
                "concurrency": 2,
                "numRuns": 1,
                "metric": {"name": "loss", "optimization": "minimize"},
                "params": {
                    "param1": {"kind": "uniform", "value": [0.01, 0.5]},
                    "param2": {"kind": "quniform", "value": [0.01, 0.99, 0.1]},
                    "param3": {"kind": "normal", "value": [0, 0.99]},
                    "param4": {"kind": "choice", "value": [32, 64, 126, 256]},
                    "param5": {
                        "kind": "choice",
                        "value": ["sgd", "adagrad", "adam", "ftrl"],
                    },
                    "param6": {"kind": "linspace", "value": [0, 10, 1]},
                    "param7": {"kind": "geomspace", "value": [0.1, 1, 1]},
                },
            }
        )
        suggestion = HyperoptManager(config).get_suggestions()[0]

        assert set(suggestion) == set(config.params)
        assert 0.01 <= suggestion["param1"] <= 0.5
        assert 0.01 <= suggestion["param2"] <= 0.99
        assert math.isfinite(suggestion["param3"])
        assert suggestion["param4"] in [32, 64, 126, 256]
        assert suggestion["param5"] in ["sgd", "adagrad", "adam", "ftrl"]
        assert suggestion["param6"] == 0
        assert suggestion["param7"] == 0.1

    def test_get_tpe_suggestions_basic(self):
        config = V1Hyperopt.from_dict(
            {
                "concurrency": 2,
                "numRuns": 1,
                "metric": {"name": "loss", "optimization": "minimize"},
                "params": {
                    "lr": {"kind": "uniform", "value": [0.01, 0.5]},
                    "dropout": {"kind": "uniform", "value": [0.01, 0.99]},
                    "batch": {"kind": "choice", "value": [32, 64, 126, 256]},
                    "optimizer": {
                        "kind": "choice",
                        "value": ["sgd", "adagrad", "adam", "ftrl"],
                    },
                },
            }
        )

        suggestions = HyperoptManager(config).get_suggestions()
        assert len(suggestions) == 1
        assert len({tuple(suggestion.items()) for suggestion in suggestions}) == 1
        suggestion = suggestions[0]

        self.assertTrue(0.99 >= suggestion["dropout"] >= 0.01)
        self.assertTrue(0.5 >= suggestion["lr"] >= 0.01)
        self.assertTrue(suggestion["batch"] in [32, 64, 126, 256])
        self.assertTrue(suggestion["optimizer"] in ["sgd", "adagrad", "adam", "ftrl"])

        config = V1Hyperopt.from_dict(
            {
                "concurrency": 2,
                "numRuns": 10,
                "metric": {"name": "loss", "optimization": "minimize"},
                "params": {
                    "lr": {"kind": "uniform", "value": [0.01, 0.5]},
                    "dropout": {"kind": "uniform", "value": [0.01, 0.99]},
                    "batch": {"kind": "choice", "value": [32, 64, 126, 256]},
                    "optimizer": {
                        "kind": "choice",
                        "value": ["sgd", "adagrad", "adam", "ftrl"],
                    },
                },
            }
        )

        assert len(HyperoptManager(config).get_suggestions()) == 10

    def test_get_tpe_suggestions(self):
        config = V1Hyperopt.from_dict(
            {
                "concurrency": 2,
                "numRuns": 10,
                "seed": 11,
                "metric": {"name": "loss", "optimization": "minimize"},
                "params": {
                    "param1": {"kind": "uniform", "value": [0.01, 0.5]},
                    "param2": {"kind": "quniform", "value": [0.01, 0.99, 0.1]},
                    "param3": {"kind": "normal", "value": [0, 0.99]},
                    "param4": {"kind": "choice", "value": [32, 64, 126, 256]},
                    "param5": {
                        "kind": "choice",
                        "value": ["sgd", "adagrad", "adam", "ftrl"],
                    },
                    "param6": {"kind": "linspace", "value": [0, 10, 1]},
                    "param7": {"kind": "geomspace", "value": [0.1, 1, 1]},
                },
            }
        )

        suggestions = HyperoptManager(config).get_suggestions()

        assert len(suggestions) == 10
        assert len({tuple(suggestion.items()) for suggestion in suggestions}) == 10
        for suggestion in suggestions:
            assert 0.01 <= suggestion["param1"] <= 0.5
            assert 0.01 <= suggestion["param2"] <= 0.99
            assert suggestion["param4"] in [32, 64, 126, 256]
            assert suggestion["param5"] in ["sgd", "adagrad", "adam", "ftrl"]
            assert suggestion["param6"] in [0]
            assert suggestion["param7"] in [0.1]

    def test_seed_zero_is_deterministic(self):
        config = V1Hyperopt.from_dict(
            {
                "numRuns": 4,
                "seed": 0,
                "metric": {"name": "loss", "optimization": "minimize"},
                "params": {"value": {"kind": "uniform", "value": [0, 1]}},
            }
        )

        assert (
            HyperoptManager(config).get_suggestions()
            == HyperoptManager(config).get_suggestions()
        )

    def test_tpe_learns_from_observations(self):
        for optimization, expected in [("minimize", 0), ("maximize", 1)]:
            config = V1Hyperopt.from_dict(
                {
                    "numRuns": 1,
                    "seed": 7,
                    "metric": {"name": "score", "optimization": optimization},
                    "params": {"value": {"kind": "choice", "value": [0, 1]}},
                }
            )
            configs = [{"value": value} for value in [0, 1] * 10]
            metrics = [item["value"] for item in configs]

            suggestion = HyperoptManager(config).get_suggestions(
                configs=configs, metrics=metrics
            )[0]

            assert suggestion["value"] == expected

    def test_tpe_learns_continuous_observations(self):
        cases = [("minimize", (0, 0.5)), ("maximize", (0.5, 1))]
        for optimization, expected_range in cases:
            config = V1Hyperopt.from_dict(
                {
                    "numRuns": 1,
                    "seed": 3,
                    "metric": {"name": "score", "optimization": optimization},
                    "params": {"value": {"kind": "uniform", "value": [0, 1]}},
                }
            )
            configs = [{"value": value} for value in [0.1, 0.9] * 10]
            metrics = [item["value"] for item in configs]

            suggestion = HyperoptManager(config).get_suggestions(
                configs=configs, metrics=metrics
            )[0]

            assert expected_range[0] <= suggestion["value"] <= expected_range[1]

    def test_quantized_and_log_distributions(self):
        config = V1Hyperopt.from_dict(
            {
                "numRuns": 5,
                "seed": 5,
                "metric": {"name": "loss", "optimization": "minimize"},
                "params": {
                    "quniform": {"kind": "quniform", "value": [0, 1, 0.1]},
                    "loguniform": {"kind": "loguniform", "value": [-3, -1]},
                    "qnormal": {"kind": "qnormal", "value": [0, 1, 0.25]},
                    "lognormal": {"kind": "lognormal", "value": [0, 0.5]},
                },
            }
        )

        suggestions = HyperoptManager(config).get_suggestions()

        for suggestion in suggestions:
            assert suggestion["quniform"] == pytest.approx(
                round(suggestion["quniform"] / 0.1) * 0.1
            )
            assert math.exp(-3) <= suggestion["loguniform"] <= math.exp(-1)
            assert suggestion["qnormal"] == pytest.approx(
                round(suggestion["qnormal"] / 0.25) * 0.25
            )
            assert suggestion["lognormal"] > 0

    def test_accepts_zero_from_quantized_log_history(self):
        config = V1Hyperopt.from_dict(
            {
                "numRuns": 1,
                "seed": 9,
                "metric": {"name": "loss", "optimization": "minimize"},
                "params": {"value": {"kind": "qlognormal", "value": [0, 1, 1]}},
            }
        )
        configs = [{"value": value} for value in [0, 1] * 5]

        suggestion = HyperoptManager(config).get_suggestions(
            configs=configs,
            metrics=[item["value"] for item in configs],
        )[0]

        assert suggestion["value"] >= 0

    def test_invalid_observations(self):
        config = V1Hyperopt.from_dict(
            {
                "numRuns": 1,
                "metric": {"name": "loss", "optimization": "minimize"},
                "params": {"value": {"kind": "uniform", "value": [0, 1]}},
            }
        )

        with pytest.raises(ValueError, match="provided together"):
            HyperoptManager(config).get_suggestions(
                configs=[{"value": 0.5}], metrics=None
            )

        with pytest.raises(ValueError, match="same length"):
            HyperoptManager(config).get_suggestions(
                configs=[{"value": 0.5}], metrics=[]
            )

        with pytest.raises(ValueError, match="missing parameters"):
            HyperoptManager(config).get_suggestions(configs=[{}], metrics=[1.0])

        with pytest.raises(ValueError, match="finite numbers"):
            HyperoptManager(config).get_suggestions(
                configs=[{"value": 0.5}], metrics=[float("inf")]
            )

    def test_unsupported_parameter_kind(self):
        config = V1Hyperopt.from_dict(
            {
                "numRuns": 1,
                "metric": {"name": "loss", "optimization": "minimize"},
                "params": {
                    "value": {
                        "kind": "pchoice",
                        "value": [["a", 0.5], ["b", 0.5]],
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="not supported by TPE"):
            HyperoptManager(config).get_suggestions()
