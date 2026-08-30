import math
from numbers import Real
import numpy as np
from typing import Dict, List, Optional, Sequence

from hypertune.matrix.utils import to_numpy
from polyaxon.schemas import (
    V1HpChoice,
    V1HpGeomSpace,
    V1HpLinSpace,
    V1HpLogNormal,
    V1HpLogSpace,
    V1HpLogUniform,
    V1HpNormal,
    V1HpQLogNormal,
    V1HpQLogUniform,
    V1HpQNormal,
    V1HpQUniform,
    V1HpRange,
    V1HpUniform,
    V1Optimization,
)


_EPS = 1e-12
_SQRT_TWO = math.sqrt(2.0)
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)

_DISCRETE_KINDS = {
    V1HpChoice._IDENTIFIER,
    V1HpRange._IDENTIFIER,
    V1HpLinSpace._IDENTIFIER,
    V1HpLogSpace._IDENTIFIER,
    V1HpGeomSpace._IDENTIFIER,
}
_UNIFORM_KINDS = {
    V1HpUniform._IDENTIFIER,
    V1HpQUniform._IDENTIFIER,
    V1HpLogUniform._IDENTIFIER,
    V1HpQLogUniform._IDENTIFIER,
}
_NORMAL_KINDS = {
    V1HpNormal._IDENTIFIER,
    V1HpQNormal._IDENTIFIER,
    V1HpLogNormal._IDENTIFIER,
    V1HpQLogNormal._IDENTIFIER,
}
_LOG_KINDS = {
    V1HpLogUniform._IDENTIFIER,
    V1HpQLogUniform._IDENTIFIER,
    V1HpLogNormal._IDENTIFIER,
    V1HpQLogNormal._IDENTIFIER,
}
_QUANTIZED_KINDS = {
    V1HpQUniform._IDENTIFIER,
    V1HpQLogUniform._IDENTIFIER,
    V1HpQNormal._IDENTIFIER,
    V1HpQLogNormal._IDENTIFIER,
}
_QUANTIZED_LOG_KINDS = {
    V1HpQLogUniform._IDENTIFIER,
    V1HpQLogNormal._IDENTIFIER,
}
_SUPPORTED_KINDS = _DISCRETE_KINDS | _UNIFORM_KINDS | _NORMAL_KINDS


def _normal_cdf(values):
    values = np.asarray(values, dtype=float)
    return np.asarray(
        [0.5 * (1.0 + math.erf(value / _SQRT_TWO)) for value in values.flat]
    ).reshape(values.shape)


def _find_value(values, target):
    for index, value in enumerate(values):
        if value == target:
            return index
    raise ValueError(
        "Observed categorical value {!r} is not in the search space.".format(target)
    )


class _CategoricalParzenEstimator:
    def __init__(self, values: Sequence, observations: Sequence):
        self.values = list(values)
        counts = np.ones(len(self.values), dtype=float)
        for observation in observations:
            counts[_find_value(self.values, observation)] += 1.0
        self.probabilities = counts / counts.sum()

    def sample(self, size, random_state):
        indices = random_state.choice(len(self.values), size=size, p=self.probabilities)
        return [self.values[index] for index in indices]

    def log_pdf(self, values):
        return np.asarray(
            [
                math.log(self.probabilities[_find_value(self.values, value)])
                for value in values
            ]
        )


class _NumericalParzenEstimator:
    def __init__(
        self,
        observations: Sequence[float],
        prior_mean: float,
        prior_scale: float,
        low: Optional[float] = None,
        high: Optional[float] = None,
    ):
        self.low = low
        self.high = high
        observations = np.asarray(observations, dtype=float)
        self.means = np.append(observations, prior_mean)

        min_scale = prior_scale / min(100.0, len(self.means) + 1.0)
        sigmas = []
        for index, mean in enumerate(self.means):
            distances = np.abs(np.delete(self.means, index) - mean)
            sigma = distances.min() if len(distances) else prior_scale
            sigmas.append(max(min_scale, min(prior_scale, sigma)))
        sigmas[-1] = prior_scale

        self.sigmas = np.asarray(sigmas)
        self.weights = np.ones(len(self.means), dtype=float) / len(self.means)

    def sample(self, size, random_state):
        samples = []
        while len(samples) < size:
            component = random_state.choice(len(self.means), p=self.weights)
            value = random_state.normal(
                loc=self.means[component], scale=self.sigmas[component]
            )
            if self.low is not None and value < self.low:
                continue
            if self.high is not None and value > self.high:
                continue
            samples.append(value)
        return np.asarray(samples)

    def log_pdf(self, values):
        values = np.asarray(values, dtype=float)[:, np.newaxis]
        normalized = (values - self.means) / self.sigmas
        densities = np.exp(-0.5 * normalized**2) / (_SQRT_TWO_PI * self.sigmas)

        if self.low is not None and self.high is not None:
            upper = (self.high - self.means) / self.sigmas
            lower = (self.low - self.means) / self.sigmas
            normalizers = np.maximum(_normal_cdf(upper) - _normal_cdf(lower), _EPS)
            densities /= normalizers

        mixture_density = densities.dot(self.weights)
        return np.log(np.maximum(mixture_density, _EPS))


class TPESampler:
    """Independent Tree-structured Parzen Estimator sampler."""

    def __init__(
        self,
        params: Dict,
        optimization,
        seed: Optional[int] = None,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
    ):
        self.params = params
        self.optimization = optimization
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.random_state = np.random.RandomState(seed)
        self._validate_search_space()

    def _validate_search_space(self):
        for name, param in self.params.items():
            kind = param._IDENTIFIER
            if kind not in _SUPPORTED_KINDS:
                raise ValueError(
                    "Parameter kind `{}` is not supported by TPE for `{}`.".format(
                        kind, name
                    )
                )
            if kind in _DISCRETE_KINDS and not len(to_numpy(param)):
                raise ValueError(
                    "Parameter `{}` has an empty search space.".format(name)
                )
            if kind in _UNIFORM_KINDS and param.value.low >= param.value.high:
                raise ValueError(
                    "Parameter `{}` requires `low` to be smaller than `high`.".format(
                        name
                    )
                )
            if kind in _NORMAL_KINDS and param.value.scale <= 0:
                raise ValueError(
                    "Parameter `{}` requires a positive `scale`.".format(name)
                )
            if kind in _QUANTIZED_KINDS and param.value.q <= 0:
                raise ValueError("Parameter `{}` requires a positive `q`.".format(name))

    def _validate_observations(self, configs, metrics):
        if configs is None and metrics is None:
            return [], []
        if configs is None or metrics is None:
            raise ValueError("Both `configs` and `metrics` must be provided together.")
        if len(configs) != len(metrics):
            raise ValueError("`configs` and `metrics` must have the same length.")

        required_params = set(self.params)
        for index, config in enumerate(configs):
            missing = required_params - set(config)
            if missing:
                raise ValueError(
                    "Observation {} is missing parameters: {}.".format(
                        index, ", ".join(sorted(missing))
                    )
                )
        for metric in metrics:
            if not isinstance(metric, Real) or not math.isfinite(metric):
                raise ValueError("Observation metrics must be finite numbers.")
        return list(configs), list(metrics)

    def _is_discrete(self, param):
        return param._IDENTIFIER in _DISCRETE_KINDS

    def _to_internal(self, param, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Observed numerical values must be finite.")
        if param._IDENTIFIER in _LOG_KINDS:
            if value <= 0:
                if value == 0 and param._IDENTIFIER in _QUANTIZED_LOG_KINDS:
                    value = param.value.q / 4.0
                    if param._IDENTIFIER == V1HpQLogUniform._IDENTIFIER:
                        value = min(
                            math.exp(param.value.high),
                            max(math.exp(param.value.low), value),
                        )
                else:
                    raise ValueError(
                        "Observed values for log distributions must be positive."
                    )
            value = math.log(value)
        if param._IDENTIFIER in _UNIFORM_KINDS:
            if not param.value.low <= value <= param.value.high:
                raise ValueError("Observed value is outside the configured bounds.")
        return value

    def _to_external(self, param, value):
        kind = param._IDENTIFIER
        if kind in _LOG_KINDS:
            value = math.exp(value)
        if kind in _QUANTIZED_KINDS:
            value = round(value / param.value.q) * param.value.q
        if kind in {V1HpUniform._IDENTIFIER, V1HpQUniform._IDENTIFIER}:
            value = min(param.value.high, max(param.value.low, value))
        if kind in {V1HpLogUniform._IDENTIFIER, V1HpQLogUniform._IDENTIFIER}:
            value = min(
                math.exp(param.value.high),
                max(math.exp(param.value.low), value),
            )
        return value

    def _prior(self, param):
        if param._IDENTIFIER in _UNIFORM_KINDS:
            low = float(param.value.low)
            high = float(param.value.high)
            return (low + high) / 2.0, high - low, low, high
        return float(param.value.loc), float(param.value.scale), None, None

    def _sample_prior(self, param):
        if self._is_discrete(param):
            values = list(to_numpy(param))
            return values[self.random_state.randint(len(values))]

        prior_mean, prior_scale, low, high = self._prior(param)
        if low is not None:
            value = self.random_state.uniform(low=low, high=high)
        else:
            value = self.random_state.normal(loc=prior_mean, scale=prior_scale)
        return self._to_external(param, value)

    def _split_observations(self, configs, metrics):
        losses = np.asarray(metrics, dtype=float)
        if self.optimization == V1Optimization.MAXIMIZE:
            losses = -losses
        order = np.argsort(losses, kind="mergesort")
        num_good = min(max(1, int(math.ceil(0.2 * len(order)))), len(order) - 1)
        return order[:num_good], order[num_good:]

    def _sample_tpe_param(self, param, good_values, bad_values):
        if self._is_discrete(param):
            values = list(to_numpy(param))
            good = _CategoricalParzenEstimator(values, good_values)
            bad = _CategoricalParzenEstimator(values, bad_values)
            candidates = good.sample(self.n_ei_candidates, self.random_state)
        else:
            prior_mean, prior_scale, low, high = self._prior(param)
            good = _NumericalParzenEstimator(
                [self._to_internal(param, value) for value in good_values],
                prior_mean,
                prior_scale,
                low,
                high,
            )
            bad = _NumericalParzenEstimator(
                [self._to_internal(param, value) for value in bad_values],
                prior_mean,
                prior_scale,
                low,
                high,
            )
            internal_candidates = good.sample(self.n_ei_candidates, self.random_state)
            scores = good.log_pdf(internal_candidates) - bad.log_pdf(
                internal_candidates
            )
            return self._to_external(param, internal_candidates[int(np.argmax(scores))])

        scores = good.log_pdf(candidates) - bad.log_pdf(candidates)
        return candidates[int(np.argmax(scores))]

    def _sample_tpe(self, configs, metrics):
        good_indices, bad_indices = self._split_observations(configs, metrics)
        suggestion = {}
        for name, param in self.params.items():
            good_values = [configs[index][name] for index in good_indices]
            bad_values = [configs[index][name] for index in bad_indices]
            suggestion[name] = self._sample_tpe_param(param, good_values, bad_values)
        return suggestion

    def _sample(self, configs, metrics):
        if len(configs) < self.n_startup_trials:
            return {
                name: self._sample_prior(param) for name, param in self.params.items()
            }
        return self._sample_tpe(configs, metrics)

    def suggest(
        self,
        num_suggestions: int,
        configs: Optional[List[Dict]] = None,
        metrics: Optional[List[float]] = None,
    ) -> List[Dict]:
        configs, metrics = self._validate_observations(configs, metrics)
        suggestions = []

        for _ in range(num_suggestions):
            for _ in range(64):
                suggestion = self._sample(configs, metrics)
                if suggestion not in suggestions:
                    break
            suggestions.append(suggestion)

            configs.append(suggestion)
            if metrics:
                if self.optimization == V1Optimization.MAXIMIZE:
                    metrics.append(min(metrics))
                else:
                    metrics.append(max(metrics))
            else:
                metrics.append(0.0)

        return suggestions
