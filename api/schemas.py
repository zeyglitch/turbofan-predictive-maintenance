"""What the API accepts and returns. pydantic checks every request against these classes and
answers 422 with the reason when a field is missing or has the wrong type."""

from pydantic import BaseModel, ConfigDict, Field


class Cycle(BaseModel):
    """One line of an engine's history: the cycle number, 3 settings and the 21 sensors."""

    cycle: int = Field(ge=1)
    setting_1: float
    setting_2: float
    setting_3: float
    s_1: float
    s_2: float
    s_3: float
    s_4: float
    s_5: float
    s_6: float
    s_7: float
    s_8: float
    s_9: float
    s_10: float
    s_11: float
    s_12: float
    s_13: float
    s_14: float
    s_15: float
    s_16: float
    s_17: float
    s_18: float
    s_19: float
    s_20: float
    s_21: float

    # last cycle of test engine 1, shown as an example on the /docs page
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "cycle": 31,
                    "setting_1": -0.0006,
                    "setting_2": 0.0004,
                    "setting_3": 100.0,
                    "s_1": 518.67,
                    "s_2": 642.58,
                    "s_3": 1581.22,
                    "s_4": 1398.91,
                    "s_5": 14.62,
                    "s_6": 21.61,
                    "s_7": 554.42,
                    "s_8": 2388.08,
                    "s_9": 9056.4,
                    "s_10": 1.3,
                    "s_11": 47.23,
                    "s_12": 521.79,
                    "s_13": 2388.06,
                    "s_14": 8130.11,
                    "s_15": 8.4024,
                    "s_16": 0.03,
                    "s_17": 393.0,
                    "s_18": 2388.0,
                    "s_19": 100.0,
                    "s_20": 38.81,
                    "s_21": 23.3552,
                }
            ]
        }
    )


class EngineHistory(BaseModel):
    """Body of POST /predict: one engine and its cycles so far, in any order."""

    unit_id: int
    cycles: list[Cycle] = Field(min_length=1)


class Prediction(BaseModel):
    """Answer of POST /predict, for the last cycle received."""

    unit_id: int
    cycle: int
    rul: float
    rul_low: float
    rul_high: float
    alert: bool
    model_version: str


class ModelInfo(BaseModel):
    """Answer of GET /model: which model is served and how good it was."""

    model_type: str
    model_version: str
    created_at: str
    sensors: list[str]
    rul_cap: int
    alert_threshold: int
    val_rmse: float
    test_rmse: float


class Health(BaseModel):
    """Answer of GET /health."""

    status: str
    model_loaded: bool
