"""Storage component: private image files on disk."""

import logging

# One handler for all loggers of this package (`storage.api`, `storage.files`, ...).
_logger: logging.Logger = logging.getLogger(__name__)
if not _logger.handlers:
    _handler: logging.StreamHandler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s-%(asctime)s: %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
