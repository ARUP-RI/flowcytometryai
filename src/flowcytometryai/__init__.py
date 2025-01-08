import logging
from flowcytometryai.config import Settings, get_panel_config


__version__ = "0.0.1"

logger = logging.getLogger("flowcytometryai")

config = Settings()

PANEL_CONFIG = get_panel_config()
