"""Request and response schema for the scoring API.

Field names follow the PPRA release columns the pipeline reads. Types, ranges and the
batch size are checked here; procurement methods and categories are checked against
the frozen pipeline's own vocabulary in vocab_errors(). Text is passed on exactly as sent,
spaces included: the frozen pipeline does its own name normalisation and matching.
"""
from datetime import datetime, timezone
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_BATCH = 20
MAX_BODY_BYTES = 64_000
EARLIEST_CLOSING = datetime(2018, 1, 1, tzinfo=timezone.utc)  # start of the reference window
DATE_RANGE = (datetime(1990, 1, 1, tzinfo=timezone.utc), datetime(2100, 1, 1, tzinfo=timezone.utc))


def _utc(v):
    """Dates without a timezone are read as UTC, like the release."""
    return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")  # no stripping: values reach the pipeline unchanged

    ocid: Optional[str] = Field(None, max_length=200, description="contract id, echoed back")
    buyer_id: Union[str, int] = Field(..., description="PPRA buyer id")
    buyer_name: str = Field(..., min_length=1, max_length=300)
    # blank text is kept: the frozen pipeline treats it as a reported (not missing) supplier
    supplier_name: Optional[str] = Field(None, max_length=300, description="null if not reported")
    value_amount: float = Field(..., ge=0, allow_inf_nan=False, description="award value, 0 allowed")
    value_currency: Optional[str] = Field(None, max_length=10, description="echoed only, not used")
    tender_mainProcurementCategory: Optional[str] = Field(None, max_length=50)
    tender_procurementMethod: str = Field(..., max_length=50)
    status: Optional[str] = Field(..., min_length=1, max_length=50,
                                  description="contract status; required key, null if not reported")
    tender_tenderPeriod_endDate: datetime = Field(..., description="tender closing date, ISO 8601")
    dateSigned: Optional[datetime] = Field(None, description="signing date, null if not reported")

    @field_validator("buyer_id")
    @classmethod
    def _buyer(cls, v):
        v = str(v)
        if not v.strip() or len(v) > 200:
            raise ValueError("buyer_id must have 1 to 200 characters and not be blank")
        return v

    @field_validator("buyer_name")
    @classmethod
    def _buyer_name(cls, v):
        if not v.strip():
            raise ValueError("buyer_name must not be blank")
        return v

    @field_validator("tender_tenderPeriod_endDate")
    @classmethod
    def _closing(cls, v):
        v = _utc(v)
        if not EARLIEST_CLOSING <= v < DATE_RANGE[1]:
            raise ValueError("closing date must be on or after 2018-01-01 (the reference window) and before 2100")
        return v

    @field_validator("dateSigned")
    @classmethod
    def _signed(cls, v):
        if v is None:
            return v
        v = _utc(v)
        if not DATE_RANGE[0] <= v < DATE_RANGE[1]:
            raise ValueError("signing date must be between 1990 and 2100")
        return v


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contracts: list[Contract] = Field(..., min_length=1, max_length=MAX_BATCH)


class ContractScore(BaseModel):
    ocid: Optional[str]
    score_eif: float
    pct_eif: float
    flag_eif: bool
    score_if: float
    pct_if: float
    flag_if: bool
    R: Optional[float]
    f1_direct_award: int
    f3_decision_period: Optional[float]
    f4_supplier_concentration: Optional[float]
    f5_benford_deviation: Optional[float]
    days_to_signing: Optional[float]
    method_level: Optional[float]
    is_signed_pre_closure: int
    zero_value_award: int
    above_p99: int
    below_p1: int
    is_terminated: int


class ScoreResponse(BaseModel):
    deployed_model: str
    comparison_model: str
    thresholds: dict[str, float]
    note: str
    latency_ms: float
    results: list[ContractScore]


def vocab_errors(req, vocab):
    """Field errors for methods and categories the frozen pipeline does not know."""
    errs = []
    for i, c in enumerate(req.contracts):
        if c.tender_procurementMethod not in vocab["methods"]:
            errs.append({"field": f"contracts.{i}.tender_procurementMethod",
                         "message": f"unknown method; allowed: {sorted(vocab['methods'])}"})
        cat = c.tender_mainProcurementCategory
        if cat is not None and cat not in vocab["categories"]:
            errs.append({"field": f"contracts.{i}.tender_mainProcurementCategory",
                         "message": f"unknown category; allowed: {sorted(vocab['categories'])} or null"})
    return errs
