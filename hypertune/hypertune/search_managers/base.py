from typing import Dict, List


class BaseManager:
    CONFIG = None

    def __init__(self, config):
        if not isinstance(config, self.CONFIG):
            raise ValueError(
                "The current search manager `{}` is not compatible "
                "with the search kind `{}` defined in the config.".format(
                    self.CONFIG._IDENTIFIER, config._IDENTIFIER
                )
            )
        self.config = config

    def get_suggestions(self, **kwargs) -> List[Dict]:
        raise NotImplementedError
