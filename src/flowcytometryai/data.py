from dataclasses import dataclass

from flowanalysis.utils.enums import Disease


@dataclass
class PredictionMetadata:
    modelVersion: str
    modelMD5Sum: str
    classifierLocation: str
    centroidLocation: str


@dataclass
class Prediction:
    disease: Disease
    prediction: bool
    pValue: float
    pValueThreshold: float
    metadata: PredictionMetadata
