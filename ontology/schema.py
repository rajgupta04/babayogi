"""
Pydantic v2 schema models for the Ayurvedic ontology JSON files.

Provides strict validation for dosha_rules.json, viruddha_rules.json, and
ritu_charya.json.  Any structural change that breaks a contract is caught at
load time, making it safe to hand-edit the JSON and verify with::

    python -m ontology.verify

All models use ``from __future__ import annotations`` for modern type-hint
syntax and Pydantic v2 ``model_config`` settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared / primitive models
# ---------------------------------------------------------------------------

class DoshaEffect(BaseModel):
    """Numeric effect a substance has on each dosha (-2 … +2)."""

    vata: float = Field(..., ge=-2, le=2)
    pitta: float = Field(..., ge=-2, le=2)
    kapha: float = Field(..., ge=-2, le=2)


# ---------------------------------------------------------------------------
# Rasa (taste) models
# ---------------------------------------------------------------------------

class Rasa(BaseModel):
    """One of the six tastes (Shad-Rasa)."""

    english: str
    sanskrit: str
    elements: list[str]
    virya: str
    vipaka: str
    dosha_effect: DoshaEffect
    qualities: list[str]
    therapeutic_actions: list[str]
    excess_effects: list[str]
    common_foods: list[str]
    source: str


# ---------------------------------------------------------------------------
# Guna (quality) models
# ---------------------------------------------------------------------------

class GunaQuality(BaseModel):
    """A single quality within a Guna pair."""

    sanskrit: str
    english: str
    id: str
    effects: str
    increases: list[str]
    decreases: list[str]

    model_config = {"extra": "allow"}  # pitta_effect / vata_effect / kapha_effect etc.


class GunaPair(BaseModel):
    """A pair of opposing gunas (e.g. guru ↔ laghu)."""

    pair_name: str
    quality_1: GunaQuality
    quality_2: GunaQuality


# ---------------------------------------------------------------------------
# Agni models
# ---------------------------------------------------------------------------

class AgniType(BaseModel):
    """One of the four digestive-fire types."""

    english: str
    sanskrit: str
    associated_dosha: str
    description: str
    signs: list[str]
    dietary_approach: str
    recommended_foods: list[str]
    avoid: list[str]
    herbs_for_correction: list[str] | None = None
    source: str


# ---------------------------------------------------------------------------
# Dosha model
# ---------------------------------------------------------------------------

class Dosha(BaseModel):
    """Full representation of a single dosha (Vata / Pitta / Kapha)."""

    sanskrit: str
    meaning: str
    elements: list[str]
    seat_in_body: str
    functions: list[str]
    qualities: list[str]
    pacifying_qualities: list[str]
    pacifying_rasas: list[str]
    aggravating_rasas: list[str]
    peak_time_of_day: list[str]
    peak_season: str
    peak_life_stage: str
    balanced_signs: list[str]
    imbalanced_signs: list[str]
    dietary_principles: list[str]
    source: str


# ---------------------------------------------------------------------------
# Dhatu model
# ---------------------------------------------------------------------------

class Dhatu(BaseModel):
    """One of the seven bodily tissues (Sapta Dhatu)."""

    name: str
    english: str
    sanskrit: str
    nourished_by_rasa: str
    function: str
    depleted_signs: list[str]
    excess_signs: list[str]


# ---------------------------------------------------------------------------
# Prakriti dual-type model
# ---------------------------------------------------------------------------

class PrakritiDualType(BaseModel):
    """A dual-dosha constitution type (Dvandvaja Prakriti)."""

    primary_traits: str
    dietary_focus: str
    key_rasas: list[str]
    avoid_rasas: list[str]
    priority_dosha_to_watch: str


# ---------------------------------------------------------------------------
# Viruddha (incompatibility) models
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = ("critical", "high", "medium", "low")
SeverityLiteral = Literal["critical", "high", "medium", "low"]


class ViruddhaRule(BaseModel):
    """A specific food-combination incompatibility rule."""

    id: str
    foods: list[str]
    condition: str
    category: str | None = None
    severity: SeverityLiteral
    effects: list[str]
    note: str | None = None
    source: str
    food_aliases: dict[str, list[str]] = Field(default_factory=dict)


class SeasonalIncompatibility(BaseModel):
    """Foods / qualities to avoid during a specific season."""

    id: str
    season: list[str]
    season_english: list[str]
    avoid_qualities: list[str]
    avoid_examples: list[str]
    reason: str
    prefer_instead: str
    source: str


class DoshaIncompatibility(BaseModel):
    """Foods / qualities to avoid when a specific dosha is aggravated."""

    id: str
    applicable_when: str
    avoid_foods: list[str]
    avoid_tastes: list[str]
    avoid_qualities: list[str]
    reason: str
    source: str


# ---------------------------------------------------------------------------
# Ritu Charya (seasonal regimen) models
# ---------------------------------------------------------------------------

class DietaryRecommendations(BaseModel):
    """Dietary recommendations for a given season."""

    favor_tastes: list[str]
    reduce_tastes: list[str]
    favor_qualities: list[str] = Field(default_factory=list)
    recommended_foods: list[str]
    avoid_foods: list[str]
    beverages: list[str] = Field(default_factory=list)


class Season(BaseModel):
    """Full seasonal (Ritu) description with dosha state and dietary advice."""

    english: str
    sanskrit: str
    months_indian: list[str]
    approximate_gregorian: dict[str, str]
    month_numbers: list[int]
    dominant_element: str
    dosha_state: dict[str, str]
    agni_strength: str
    body_strength: str
    dietary_recommendations: DietaryRecommendations
    lifestyle_recommendations: list[str]
    source: str


class Ayana(BaseModel):
    """Uttarayana / Dakshinayana ayana description."""

    english: str
    sanskrit: str
    description: str
    seasons: list[str]
    body_strength: str
    agni_strength: str
    general_principle: str


class SeasonMapping(BaseModel):
    """Maps English season names and month numbers to Ayurvedic Ritu names."""

    english_to_ritu: dict[str, list[str]]
    month_to_ritu: dict[str, str]

    model_config = {"extra": "allow"}  # _description field


# ---------------------------------------------------------------------------
# Top-level file models
# ---------------------------------------------------------------------------

class DoshaRulesFile(BaseModel):
    """Validates the entire ``dosha_rules.json`` file."""

    doshas: dict[str, Dosha]
    rasas: dict[str, Rasa]
    gunas: dict[str, Any]  # validated more deeply below
    agni_types: dict[str, AgniType]
    dhatus: dict[str, Any]  # validated more deeply below
    prakriti_dual_types: dict[str, PrakritiDualType]

    model_config = {"extra": "allow"}  # _meta field

    @model_validator(mode="before")
    @classmethod
    def _strip_description_keys(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Remove internal ``_description`` keys from sections before validation."""
        for section in ("rasas", "agni_types", "prakriti_dual_types"):
            if section in data and isinstance(data[section], dict):
                data[section] = {
                    k: v for k, v in data[section].items() if not k.startswith("_")
                }
        return data

    @model_validator(mode="after")
    def _validate_gunas_and_dhatus(self) -> DoshaRulesFile:
        """Deeply validate gunas.pairs and dhatus.sequence after initial parse."""
        # --- gunas ---
        raw_gunas = self.gunas
        pairs_raw = raw_gunas.get("pairs", [])
        parsed_pairs: list[GunaPair] = []
        for item in pairs_raw:
            parsed_pairs.append(GunaPair.model_validate(item))
        self._guna_pairs = parsed_pairs

        # --- dhatus ---
        raw_dhatus = self.dhatus
        seq_raw = raw_dhatus.get("sequence", [])
        parsed_dhatus: list[Dhatu] = []
        for item in seq_raw:
            parsed_dhatus.append(Dhatu.model_validate(item))
        self._dhatu_sequence = parsed_dhatus

        return self

    @property
    def guna_pairs(self) -> list[GunaPair]:
        """Return parsed guna pair models."""
        return self._guna_pairs  # type: ignore[return-value]

    @property
    def dhatu_sequence(self) -> list[Dhatu]:
        """Return parsed dhatu models."""
        return self._dhatu_sequence  # type: ignore[return-value]


class ViruddhaRulesFile(BaseModel):
    """Validates the entire ``viruddha_rules.json`` file."""

    specific_food_incompatibilities: list[ViruddhaRule]
    seasonal_food_incompatibilities: list[SeasonalIncompatibility]
    dosha_specific_incompatibilities: list[DoshaIncompatibility]

    model_config = {"extra": "allow"}  # _meta, categories_of_viruddha


class RituCharyaFile(BaseModel):
    """Validates the entire ``ritu_charya.json`` file."""

    seasons: dict[str, Season]
    season_mapping: SeasonMapping
    ayanas: dict[str, Ayana] | None = None

    model_config = {"extra": "allow"}  # _meta


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_all(ontology_dir: str | Path | None = None) -> list[str]:
    """Load and validate all three ontology JSON files.

    Parameters
    ----------
    ontology_dir:
        Directory containing the JSON files.  Defaults to the directory
        containing this module.

    Returns
    -------
    list[str]
        Empty list if all files pass validation; otherwise a list of
        human-readable error strings.
    """
    if ontology_dir is None:
        ontology_dir = Path(__file__).resolve().parent
    else:
        ontology_dir = Path(ontology_dir)

    errors: list[str] = []

    file_model_map: dict[str, type[BaseModel]] = {
        "dosha_rules.json": DoshaRulesFile,
        "viruddha_rules.json": ViruddhaRulesFile,
        "ritu_charya.json": RituCharyaFile,
    }

    for filename, model_cls in file_model_map.items():
        filepath = ontology_dir / filename
        if not filepath.exists():
            errors.append(f"File not found: {filepath}")
            continue
        try:
            with open(filepath, encoding="utf-8") as fh:
                raw = json.load(fh)
            model_cls.model_validate(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"{filename}: Invalid JSON — {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{filename}: Validation error — {exc}")

    return errors
