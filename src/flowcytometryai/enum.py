from enum import Enum


class Disease(str, Enum):
    AML = ("AML",)


class ModelType(str, Enum):
    BOOST = ("boost",)


class TubeType(str, Enum):
    Myeloid = "Myeloid"
    B_cell = "B_cell"
    T_cell = "T_cell"
