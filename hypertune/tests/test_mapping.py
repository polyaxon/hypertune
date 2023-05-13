import pytest

from hypertune.search_managers.mapping.manager import MappingManager
from polyaxon.polyflow import V1Mapping
from polyaxon.utils.test_utils import BaseTestCase


@pytest.mark.tuninig_mark
class TestMapping(BaseTestCase):
    def test_mapping_config(self):
        assert MappingManager.CONFIG == V1Mapping

    def test_get_suggestions(self):
        config = V1Mapping.from_dict(
            {"concurrency": 2, "values": [{"a": 1, "b": 2}, {"a": 1.3, "b": 3}]}
        )
        assert len(MappingManager(config).get_suggestions()) == 2

        config = V1Mapping.from_dict(
            {
                "concurrency": 2,
                "values": [
                    {"feature1": 1, "feature2": 1, "feature3": 1},
                    {"feature1": 2, "feature2": 2, "feature3": 2},
                    {"feature1": 3, "feature2": 3, "feature3": 3},
                ],
            }
        )
        assert len(MappingManager(config).get_suggestions()) == 3
