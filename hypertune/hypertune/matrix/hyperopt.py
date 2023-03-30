#!/usr/bin/python
#
# Copyright 2018-2023 Polyaxon, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hyperopt

from hypertune.matrix.utils import to_numpy
from polyaxon.polyflow import (
    V1HpChoice,
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
)


def to_hyperopt(name, matrix):
    if matrix._IDENTIFIER in {
        V1HpChoice._IDENTIFIER,
        V1HpRange._IDENTIFIER,
        V1HpLinSpace._IDENTIFIER,
        V1HpLogSpace._IDENTIFIER,
        V1HpGeomSpace._IDENTIFIER,
    }:
        return hyperopt.hp.choice(name, to_numpy(matrix))

    if matrix._IDENTIFIER == V1HpPChoice._IDENTIFIER:
        raise ValueError("{} is not supported by Hyperopt.".format(matrix._IDENTIFIER))

    if matrix._IDENTIFIER == V1HpUniform._IDENTIFIER:
        return hyperopt.hp.uniform(name, matrix.value.low, matrix.value.high)

    if matrix._IDENTIFIER == V1HpQUniform._IDENTIFIER:
        return hyperopt.hp.quniform(
            name,
            matrix.value.low,
            matrix.value.high,
            matrix.value.q,
        )

    if matrix._IDENTIFIER == V1HpLogUniform._IDENTIFIER:
        return hyperopt.hp.loguniform(name, matrix.value.low, matrix.value.high)

    if matrix._IDENTIFIER == V1HpQLogUniform._IDENTIFIER:
        return hyperopt.hp.qloguniform(
            name,
            matrix.value.low,
            matrix.value.high,
            matrix.value.q,
        )

    if matrix._IDENTIFIER == V1HpNormal._IDENTIFIER:
        return hyperopt.hp.normal(name, matrix.value.loc, matrix.value.scale)

    if matrix._IDENTIFIER == V1HpQNormal._IDENTIFIER:
        return hyperopt.hp.qnormal(
            name,
            matrix.value.loc,
            matrix.value.scale,
            matrix.value.q,
        )

    if matrix._IDENTIFIER == V1HpLogNormal._IDENTIFIER:
        return hyperopt.hp.lognormal(name, matrix.value.loc, matrix.value.scale)

    if matrix._IDENTIFIER == V1HpQLogNormal._IDENTIFIER:
        return hyperopt.hp.qlognormal(
            name,
            matrix.value.loc,
            matrix.value.scale,
            matrix.value.q,
        )
