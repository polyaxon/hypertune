import numpy as np

from datetime import date, datetime

from clipped.compact.pydantic import ValidationError

from hypertune.matrix import dist
from polyaxon.schemas import (
    V1HpChoice,
    V1HpDateRange,
    V1HpDateTimeRange,
    V1HpGeomSpace,
    V1HpLinSpace,
    V1HpLogNormal,
    V1HpLogSpace,
    V1HpLogUniform,
    V1HpNormal,
    V1HpPChoice,
    V1HpQLogNormal,
    V1HpQLogUniform,
    V1HpQNormal,
    V1HpQUniform,
    V1HpRange,
    V1HpUniform,
    validate_pchoice,
)


def pchoice(values, size=None, rand_generator=None):
    rand_generator = rand_generator or np.random
    keys = [v[0] for v in values]
    dists = [v[1] for v in values]
    validate_pchoice(dists)
    indices = rand_generator.multinomial(1, dists, size=size)
    if size is None:
        return keys[indices.argmax()]
    return [keys[ind.argmax()] for ind in indices]


def space_sample(value, size, rand_generator):
    size = None if size == 1 else size
    rand_generator = rand_generator or np.random
    try:
        return rand_generator.choice(value, size=size)
    except ValueError:
        idx = rand_generator.randint(0, len(value))
        return value[idx]


def space_get_index(array, value):
    try:
        return array.index(value)
    except (ValueError, AttributeError):
        return int(np.where(array == value)[0][0])


def dist_sample(fct, value, size, rand_generator):
    size = None if size == 1 else size
    rand_generator = rand_generator or np.random
    value["size"] = size
    value["rand_generator"] = rand_generator
    return fct(**value)


def get_length(matrix):
    if matrix._IDENTIFIER == V1HpChoice._IDENTIFIER:
        return len(matrix.value)

    if matrix._IDENTIFIER == V1HpPChoice._IDENTIFIER:
        return len(matrix.value)

    if matrix._IDENTIFIER == V1HpDateRange._IDENTIFIER:
        return len(np.arange(**matrix.value.to_dict()))

    if matrix._IDENTIFIER == V1HpDateTimeRange._IDENTIFIER:
        return len(np.arange(**matrix.value.to_dict()))

    if matrix._IDENTIFIER == V1HpRange._IDENTIFIER:
        return len(np.arange(**matrix.value.to_dict()))

    if matrix._IDENTIFIER == V1HpLinSpace._IDENTIFIER:
        return len(np.linspace(**matrix.value.to_dict()))

    if matrix._IDENTIFIER == V1HpLogSpace._IDENTIFIER:
        return len(np.logspace(**matrix.value.to_dict()))

    if matrix._IDENTIFIER == V1HpGeomSpace._IDENTIFIER:
        return len(np.geomspace(**matrix.value.to_dict()))

    if matrix._IDENTIFIER in {
        V1HpUniform._IDENTIFIER,
        V1HpQUniform._IDENTIFIER,
        V1HpLogUniform._IDENTIFIER,
        V1HpQLogUniform._IDENTIFIER,
        V1HpNormal._IDENTIFIER,
        V1HpQNormal._IDENTIFIER,
        V1HpLogNormal._IDENTIFIER,
        V1HpQLogNormal._IDENTIFIER,
    }:
        raise ValidationError(
            ["Distribution should not call `length`"], matrix.__class__
        )


def get_min(matrix):
    if matrix._IDENTIFIER == V1HpChoice._IDENTIFIER:
        if matrix.is_categorical:
            return None
        return min(to_numpy(matrix))

    if matrix._IDENTIFIER == V1HpPChoice._IDENTIFIER:
        return None

    if matrix._IDENTIFIER in {
        V1HpDateRange._IDENTIFIER,
        V1HpDateTimeRange._IDENTIFIER,
        V1HpRange._IDENTIFIER,
        V1HpLinSpace._IDENTIFIER,
        V1HpLogSpace._IDENTIFIER,
        V1HpGeomSpace._IDENTIFIER,
    }:
        return matrix.value.start

    if matrix._IDENTIFIER == V1HpUniform._IDENTIFIER:
        return matrix.value.low

    if matrix._IDENTIFIER in {
        V1HpQUniform._IDENTIFIER,
        V1HpLogUniform._IDENTIFIER,
        V1HpQLogUniform._IDENTIFIER,
        V1HpNormal._IDENTIFIER,
        V1HpQNormal._IDENTIFIER,
        V1HpLogNormal._IDENTIFIER,
        V1HpQLogNormal._IDENTIFIER,
    }:
        return None


def get_max(matrix):
    if matrix._IDENTIFIER == V1HpChoice._IDENTIFIER:
        if matrix.is_categorical:
            return None
        return max(to_numpy(matrix))

    if matrix._IDENTIFIER == V1HpPChoice._IDENTIFIER:
        return None

    if matrix._IDENTIFIER in {
        V1HpDateRange._IDENTIFIER,
        V1HpDateTimeRange._IDENTIFIER,
        V1HpRange._IDENTIFIER,
        V1HpLinSpace._IDENTIFIER,
        V1HpLogSpace._IDENTIFIER,
        V1HpGeomSpace._IDENTIFIER,
    }:
        return matrix.value.stop

    if matrix._IDENTIFIER == V1HpUniform._IDENTIFIER:
        return matrix.value.high

    if matrix._IDENTIFIER in {
        V1HpQUniform._IDENTIFIER,
        V1HpLogUniform._IDENTIFIER,
        V1HpQLogUniform._IDENTIFIER,
        V1HpNormal._IDENTIFIER,
        V1HpQNormal._IDENTIFIER,
        V1HpLogNormal._IDENTIFIER,
        V1HpQLogNormal._IDENTIFIER,
    }:
        return None


def to_numpy(matrix):
    if matrix._IDENTIFIER == V1HpChoice._IDENTIFIER:
        return matrix.value

    if matrix._IDENTIFIER == V1HpPChoice._IDENTIFIER:
        raise ValidationError(  # TODO: Fix error message
            [
                "Distribution should not call `to_numpy`, "
                "instead it should call `sample`."
            ],
            matrix.__class__,
        )

    if matrix._IDENTIFIER == V1HpDateRange._IDENTIFIER:
        return np.arange(**matrix.value.to_dict()).astype(date)

    if matrix._IDENTIFIER == V1HpDateTimeRange._IDENTIFIER:
        return np.arange(**matrix.value.to_dict()).astype(datetime)

    if matrix._IDENTIFIER == V1HpRange._IDENTIFIER:
        return np.arange(**matrix.value.to_dict())

    if matrix._IDENTIFIER == V1HpLinSpace._IDENTIFIER:
        return np.linspace(**matrix.value.to_dict())

    if matrix._IDENTIFIER == V1HpLogSpace._IDENTIFIER:
        return np.logspace(**matrix.value.to_dict())

    if matrix._IDENTIFIER == V1HpGeomSpace._IDENTIFIER:
        return np.geomspace(**matrix.value.to_dict())

    if matrix._IDENTIFIER in {
        V1HpUniform._IDENTIFIER,
        V1HpQUniform._IDENTIFIER,
        V1HpLogUniform._IDENTIFIER,
        V1HpQLogUniform._IDENTIFIER,
        V1HpNormal._IDENTIFIER,
        V1HpQNormal._IDENTIFIER,
        V1HpLogNormal._IDENTIFIER,
        V1HpQLogNormal._IDENTIFIER,
    }:
        raise ValidationError(
            [
                "Distribution should not call `to_numpy`, "
                "instead it should call `sample`."
            ],
            matrix.__class__,
        )


def _sample(matrix, size=None, rand_generator=None):
    size = None if size == 1 else size

    if matrix._IDENTIFIER == V1HpChoice._IDENTIFIER:
        return space_sample(
            value=to_numpy(matrix), size=size, rand_generator=rand_generator
        )
    if matrix._IDENTIFIER == V1HpPChoice._IDENTIFIER:
        return pchoice(values=matrix.value, size=size, rand_generator=rand_generator)

    if matrix._IDENTIFIER == V1HpDateRange._IDENTIFIER:
        return space_sample(
            value=to_numpy(matrix), size=size, rand_generator=rand_generator
        )

    if matrix._IDENTIFIER == V1HpDateTimeRange._IDENTIFIER:
        return space_sample(
            value=to_numpy(matrix), size=size, rand_generator=rand_generator
        )

    if matrix._IDENTIFIER == V1HpRange._IDENTIFIER:
        return space_sample(
            value=to_numpy(matrix), size=size, rand_generator=rand_generator
        )

    if matrix._IDENTIFIER == V1HpLinSpace._IDENTIFIER:
        return space_sample(
            value=to_numpy(matrix), size=size, rand_generator=rand_generator
        )

    if matrix._IDENTIFIER == V1HpLogSpace._IDENTIFIER:
        return space_sample(
            value=to_numpy(matrix), size=size, rand_generator=rand_generator
        )

    if matrix._IDENTIFIER == V1HpGeomSpace._IDENTIFIER:
        return space_sample(
            value=to_numpy(matrix), size=size, rand_generator=rand_generator
        )

    if matrix._IDENTIFIER == V1HpUniform._IDENTIFIER:
        return dist_sample(dist.uniform, matrix.value.to_dict(), size, rand_generator)

    if matrix._IDENTIFIER == V1HpQUniform._IDENTIFIER:
        return dist_sample(dist.quniform, matrix.value.to_dict(), size, rand_generator)

    if matrix._IDENTIFIER == V1HpLogUniform._IDENTIFIER:
        return dist_sample(
            dist.loguniform, matrix.value.to_dict(), size, rand_generator
        )

    if matrix._IDENTIFIER == V1HpQLogUniform._IDENTIFIER:
        return dist_sample(
            dist.qloguniform, matrix.value.to_dict(), size, rand_generator
        )

    if matrix._IDENTIFIER == V1HpNormal._IDENTIFIER:
        return dist_sample(dist.normal, matrix.value.to_dict(), size, rand_generator)

    if matrix._IDENTIFIER == V1HpQNormal._IDENTIFIER:
        return dist_sample(dist.qnormal, matrix.value.to_dict(), size, rand_generator)

    if matrix._IDENTIFIER == V1HpLogNormal._IDENTIFIER:
        return dist_sample(dist.lognormal, matrix.value.to_dict(), size, rand_generator)

    if matrix._IDENTIFIER == V1HpQLogNormal._IDENTIFIER:
        return dist_sample(
            dist.qlognormal, matrix.value.to_dict(), size, rand_generator
        )


def sample(matrix, size=None, rand_generator=None):
    try:
        return _sample(matrix, size=size, rand_generator=rand_generator)
    except Exception as e:
        raise ValidationError(
            [
                "Could not sample from matrix value: {} for kind: {} with size: {}".format(
                    matrix.value, matrix._IDENTIFIER, size
                )
            ],
            matrix.__class__,
        ) from e
