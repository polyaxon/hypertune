import copy

from typing import Dict, List, Optional

from hypertune.search_managers.base import BaseManager
from polyaxon.schemas import V1Mapping


class MappingManager(BaseManager):
    """Mapping strategy manager for running parallel operations."""

    CONFIG = V1Mapping

    def get_suggestions(self, params: Optional[Dict] = None) -> List[Dict]:
        suggestions = []
        params = params or {}
        for v in self.config.values:
            suggestion_params = copy.deepcopy(params)
            suggestion_params.update(copy.deepcopy(v))
            suggestions.append(suggestion_params)
        return suggestions
