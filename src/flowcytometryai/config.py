import os
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_panel_config():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "panel_conf.yaml")
    # logger.debug(f"Finding panel config at {path}")
    with open(path) as fh:
        return yaml.safe_load(fh)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLOWANALYSIS_")

    AML_MODEL_PATH: str = Field("/flow/AML/models", alias="AML_MODEL_PATH")
    VIABILITY_MODEL_PATH: str = Field("/flow/VIABILITY/models", alias="VIABILITY_MODEL_PATH")
    NORMAL_MODEL_PATH: str = Field("/flow/NORMAL/models", alias="NORMAL_MODEL_PATH")

    AML_CENTROID_DIR: str = Field("/flow/AML/centroids", alias="AML_CENTROID_DIR")
    VIABILITY_CENTROID_DIR: str = Field("/flow/VIABILITY/centroids", alias="VIABILITY_CENTROID_DIR")
    NORMAL_CENTROID_DIR: str = Field("/flow/NORMAL/centroids", alias="NORMAL_CENTROID_DIR")
    PANEL_CONFIG: dict = Field(get_panel_config(), alias="PANEL_CONFIG")
