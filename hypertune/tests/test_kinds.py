import pytest

from polyaxon._utils.test_utils import BaseTestCase
from polyaxon.schemas import V1MatrixKind


@pytest.mark.tuninig_mark
class TestKinds(BaseTestCase):
    def test_supported_kinds(self):
        assert len(V1MatrixKind.to_list()) == 7

    def test_iteration_values(self):
        assert V1MatrixKind.iteration_values() == {
            V1MatrixKind.HYPERBAND,
            V1MatrixKind.BAYES,
            V1MatrixKind.HYPEROPT,
            V1MatrixKind.ITERATIVE,
        }
