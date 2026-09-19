from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import date

class GymWorkoutEnum(str, Enum):
    NONE = "NONE"
    MORNING = "MORNING"
    AFTERNOON = "AFTERNOON"
    EVENING = "EVENING"

class DailyLogCreate(BaseModel):
    entry_date: date
    am_cleanser: bool = False
    am_azid: bool = False
    am_rederma: bool = False
    am_spf: bool = False
    pm_cleanser: bool = False
    pm_differin: bool = False
    pm_rederma: bool = False
    stinging_index: int = Field(..., ge=0, le=3)
    gym_workout: GymWorkoutEnum = GymWorkoutEnum.NONE
    notes: Optional[str] = Field(None, max_length=500)

class DailyLogResponse(DailyLogCreate):
    has_photo: bool
    photo_url: Optional[str] = None

class SkinAnalysisMetrics(BaseModel):
    analysis_version: Optional[str] = None
    active_spots_count: int = 0
    active_area_pct: float = 0.0
    affected_area_pct: float = 0.0
    residual_area_pct: float = 0.0
    discromia_score: float = 0.0
    peak_p98: float = 0.0
    pore_prominence_pct: float = 0.0
    roughness_index: float = 0.0
    smoothness_score: float = 100.0

class SkinTrendPoint(SkinAnalysisMetrics):
    entry_date: date

class TimelineItem(BaseModel):
    index: int
    entry_date: date
    days_from_start: int
    photo_url: str
    pih_url: Optional[str] = None
    texture_url: Optional[str] = None
    analysis: Optional[SkinAnalysisMetrics] = None
    stinging_index: int
    am_compliance_rate: float
    pm_compliance_rate: float

class AxisMetric(BaseModel):
    current_pct: float
    target_pct: float
    completed_slots: int
    total_slots: int
    is_compliant: bool

class TherapeuticAxisMetric(AxisMetric):
    id: str
    label: str

class TherapeuticPressureResponse(BaseModel):
    window_start: str
    window_end: str
    days_recorded: int
    window_days: int
    axes: List[TherapeuticAxisMetric]
