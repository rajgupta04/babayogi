"""
Runtime query interface for the Ayurvedic ontology.

Loads the three ontology JSON files, validates them against Pydantic schemas,
and exposes a rich query API used by the RAG pipeline — query_router,
guardrails, context builder, and the generation layer.

Usage::

    from ontology.loader import AyurvedicOntology

    onto = AyurvedicOntology()                # loads from default dir
    onto.get_pacifying_rasas("vata")          # ["madhura", "amla", "lavana"]
    context = onto.build_context(user_payload) # full ontology context for LLM
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .schema import (
    DoshaRulesFile,
    RituCharyaFile,
    ViruddhaRulesFile,
)

logger = logging.getLogger(__name__)

_VALID_DOSHAS = frozenset({"vata", "pitta", "kapha"})


class AyurvedicOntology:
    """Loads the Ayurvedic ontology JSON files and provides query methods.

    Designed for use in the RAG pipeline — called by query_router,
    guardrails, etc.
    """

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def __init__(self, ontology_dir: str | Path | None = None) -> None:
        """Load all three ontology JSONs and validate with Pydantic schemas.

        Parameters
        ----------
        ontology_dir:
            Path to the directory containing the JSON files.
            Defaults to the directory that contains *this* module.
        """
        if ontology_dir is None:
            self._dir = Path(__file__).resolve().parent
        else:
            self._dir = Path(ontology_dir)

        # Load raw JSON data
        self._dosha_raw: dict[str, Any] = self._load_json("dosha_rules.json")
        self._viruddha_raw: dict[str, Any] = self._load_json("viruddha_rules.json")
        self._ritu_raw: dict[str, Any] = self._load_json("ritu_charya.json")

        # Validate through Pydantic models
        self._dosha_model = DoshaRulesFile.model_validate(self._dosha_raw)
        self._viruddha_model = ViruddhaRulesFile.model_validate(self._viruddha_raw)
        self._ritu_model = RituCharyaFile.model_validate(self._ritu_raw)

        # Build fast-lookup indexes
        self._build_indexes()

        logger.info(
            "AyurvedicOntology loaded: %d doshas, %d rasas, %d guna pairs, "
            "%d viruddha rules, %d seasons",
            len(self._doshas),
            len(self._rasas),
            len(self._guna_pairs),
            len(self._viruddha_rules),
            len(self._seasons),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_json(self, filename: str) -> dict[str, Any]:
        """Read and parse a JSON file from the ontology directory."""
        filepath = self._dir / filename
        with open(filepath, encoding="utf-8") as fh:
            return json.load(fh)

    def _build_indexes(self) -> None:
        """Pre-compute look-up dicts for O(1) access."""
        # --- doshas (raw dicts, keyed lowercase) ---
        self._doshas: dict[str, dict[str, Any]] = {
            k.lower().strip(): v
            for k, v in self._dosha_raw["doshas"].items()
        }

        # --- rasas (strip _description) ---
        self._rasas: dict[str, dict[str, Any]] = {
            k.lower().strip(): v
            for k, v in self._dosha_raw["rasas"].items()
            if not k.startswith("_")
        }

        # --- guna pairs ---
        self._guna_pairs: list[dict[str, Any]] = self._dosha_raw["gunas"]["pairs"]
        self._guna_by_id: dict[str, dict[str, Any]] = {}
        for pair in self._guna_pairs:
            for side in ("quality_1", "quality_2"):
                q = pair[side]
                self._guna_by_id[q["id"].lower().strip()] = q

        # --- agni types (strip _description) ---
        self._agni_types: dict[str, dict[str, Any]] = {
            k.lower().strip(): v
            for k, v in self._dosha_raw["agni_types"].items()
            if not k.startswith("_")
        }

        # --- dhatus ---
        self._dhatus: list[dict[str, Any]] = self._dosha_raw["dhatus"]["sequence"]

        # --- prakriti dual types ---
        self._prakriti_dual: dict[str, dict[str, Any]] = {
            k.lower().strip(): v
            for k, v in self._dosha_raw["prakriti_dual_types"].items()
            if not k.startswith("_")
        }

        # --- viruddha rules ---
        self._viruddha_rules: list[dict[str, Any]] = self._viruddha_raw[
            "specific_food_incompatibilities"
        ]
        self._seasonal_incomp: list[dict[str, Any]] = self._viruddha_raw[
            "seasonal_food_incompatibilities"
        ]
        self._dosha_incomp: list[dict[str, Any]] = self._viruddha_raw[
            "dosha_specific_incompatibilities"
        ]

        # --- seasons ---
        self._seasons: dict[str, dict[str, Any]] = {
            k.lower().strip(): v
            for k, v in self._ritu_raw["seasons"].items()
        }
        self._season_mapping: dict[str, Any] = self._ritu_raw["season_mapping"]

    @staticmethod
    def _norm(key: str) -> str:
        """Normalize a lookup key."""
        return key.lower().strip()

    # ------------------------------------------------------------------ #
    # Dosha queries
    # ------------------------------------------------------------------ #

    def get_dosha(self, name: str) -> dict[str, Any]:
        """Return full dosha info for *name* (vata / pitta / kapha).

        Raises
        ------
        KeyError
            If the dosha name is not recognized.
        """
        key = self._norm(name)
        if key not in self._doshas:
            raise KeyError(f"Unknown dosha: {name!r}. Valid: {sorted(self._doshas)}")
        return self._doshas[key]

    def get_pacifying_rasas(self, dosha: str) -> list[str]:
        """Return the list of rasa names that pacify *dosha*."""
        return list(self.get_dosha(dosha)["pacifying_rasas"])

    def get_aggravating_rasas(self, dosha: str) -> list[str]:
        """Return the list of rasa names that aggravate *dosha*."""
        return list(self.get_dosha(dosha)["aggravating_rasas"])

    def get_pacifying_qualities(self, dosha: str) -> list[str]:
        """Return quality IDs that pacify *dosha*.

        The JSON stores values like ``"ushna (hot)"``; this method returns
        only the ID portion (e.g. ``"ushna"``).
        """
        raw_qualities = self.get_dosha(dosha)["pacifying_qualities"]
        return [q.split("(")[0].strip().split()[0] for q in raw_qualities]

    # ------------------------------------------------------------------ #
    # Rasa queries
    # ------------------------------------------------------------------ #

    def get_rasa_info(self, rasa: str) -> dict[str, Any]:
        """Return full details for a single rasa (taste).

        Raises
        ------
        KeyError
            If the rasa name is not recognized.
        """
        key = self._norm(rasa)
        if key not in self._rasas:
            raise KeyError(f"Unknown rasa: {rasa!r}. Valid: {sorted(self._rasas)}")
        return self._rasas[key]

    def get_rasa_dosha_effect(self, rasa: str) -> dict[str, float]:
        """Return the dosha-effect dict for *rasa*.

        Example: ``get_rasa_dosha_effect("madhura")`` →
        ``{"vata": -2, "pitta": -1, "kapha": 2}``
        """
        return dict(self.get_rasa_info(rasa)["dosha_effect"])

    # ------------------------------------------------------------------ #
    # Agni queries
    # ------------------------------------------------------------------ #

    def get_agni_info(self, agni_type: str) -> dict[str, Any]:
        """Return full info for an agni type (sama / vishama / tikshna / manda).

        Raises
        ------
        KeyError
            If the agni type is not recognized.
        """
        key = self._norm(agni_type)
        if key not in self._agni_types:
            raise KeyError(
                f"Unknown agni type: {agni_type!r}. "
                f"Valid: {sorted(self._agni_types)}"
            )
        return self._agni_types[key]

    # ------------------------------------------------------------------ #
    # Season / Ritu queries
    # ------------------------------------------------------------------ #

    def get_season(self, season: str) -> dict[str, Any]:
        """Return full season (Ritu) info by ritu name.

        Raises
        ------
        KeyError
            If the season name is not recognized.
        """
        key = self._norm(season)
        if key not in self._seasons:
            raise KeyError(
                f"Unknown season: {season!r}. Valid: {sorted(self._seasons)}"
            )
        return self._seasons[key]

    def resolve_season(self, input: str) -> list[str]:  # noqa: A002
        """Convert an English season name or month number to ritu name(s).

        Parameters
        ----------
        input:
            An English name (``"winter"``, ``"summer"``), a ritu name
            (``"hemanta"``), or a month number string (``"1"``–``"12"``).

        Returns
        -------
        list[str]
            One or more ritu names.  Empty list if no mapping found.
        """
        key = self._norm(input)

        # Direct ritu name?
        if key in self._seasons:
            return [key]

        # English name mapping?
        eng_map: dict[str, list[str]] = {
            k.lower().strip(): v
            for k, v in self._season_mapping.get("english_to_ritu", {}).items()
        }
        if key in eng_map:
            return list(eng_map[key])

        # Month number mapping?
        month_map: dict[str, str] = self._season_mapping.get("month_to_ritu", {})
        if key in month_map:
            return [month_map[key]]

        return []

    def get_season_diet(self, season: str) -> dict[str, Any]:
        """Return dietary recommendations for *season* (ritu name)."""
        return dict(self.get_season(season)["dietary_recommendations"])

    # ------------------------------------------------------------------ #
    # Collection accessors
    # ------------------------------------------------------------------ #

    def get_all_dosha_names(self) -> list[str]:
        """Return all valid dosha names (e.g. ['kapha', 'pitta', 'vata'])."""
        return sorted(self._doshas.keys())

    def get_all_rasa_names(self) -> list[str]:
        """Return all six Shad-Rasa names."""
        return sorted(self._rasas.keys())

    def get_all_season_names(self) -> list[str]:
        """Return all six Ritu names."""
        return sorted(self._seasons.keys())

    def get_all_agni_types(self) -> list[str]:
        """Return all four Agni types."""
        return sorted(self._agni_types.keys())

    def get_dhatu_sequence(self) -> list[dict[str, Any]]:
        """Return the sequence of seven bodily tissues (Sapta Dhatu)."""
        return list(self._dhatus)

    def get_guna_pairs(self) -> list[dict[str, Any]]:
        """Return all ten pairs of opposite qualities (Vimshati Guna)."""
        return list(self._guna_pairs)

    def get_viruddha_rules(self) -> list[dict[str, Any]]:
        """Return all specific food incompatibility rules."""
        return list(self._viruddha_rules)

    # ------------------------------------------------------------------ #
    # Viruddha (incompatibility) queries
    # ------------------------------------------------------------------ #

    @staticmethod
    def _food_matches_alias(food: str, alias: str) -> bool:
        """Check if a food name matches an alias (exact, substring, or word match)."""
        f = food.strip().lower()
        a = alias.strip().lower()
        if not f or not a:
            return False
        if f == a:
            return True
        import re
        if re.search(r"\b" + re.escape(a) + r"\b", f):
            return True
        if re.search(r"\b" + re.escape(f) + r"\b", a):
            return True
        return False

    def check_viruddha(self, food_list: list[str]) -> list[dict[str, Any]]:
        """Check a list of food names against ALL viruddha rules.

        Matching uses ``food_aliases`` for fuzzy matching.  All comparisons
        are case-insensitive and support word/substring matches.

        Parameters
        ----------
        food_list:
            List of food names the user is eating / planning to eat.

        Returns
        -------
        list[dict]
            Each dict contains: ``id``, ``foods``, ``condition``, ``severity``,
            ``effects``, ``note``, ``source``, and the matched aliases.
        """
        violations: list[dict[str, Any]] = []

        for rule in self._viruddha_rules:
            # Build a set of all known aliases for each logical food in the rule
            aliases_map: dict[str, set[str]] = {}
            raw_aliases: dict[str, list[str]] = rule.get("food_aliases", {})
            for logical_food, alias_list in raw_aliases.items():
                aliases_map[logical_food] = {self._norm(a) for a in alias_list}
                aliases_map[logical_food].add(self._norm(logical_food))

            # For rules with no aliases, fall back to the rule's foods list
            if not aliases_map:
                aliases_map = {
                    f: {self._norm(f)} for f in rule["foods"]
                }

            # Check which logical foods are present in user's food_list
            matched_logical: dict[str, str] = {}
            for logical_food, aliases in aliases_map.items():
                for food in food_list:
                    norm_food = self._norm(food)
                    for alias in aliases:
                        if self._food_matches_alias(norm_food, alias):
                            matched_logical[logical_food] = food
                            break
                    if logical_food in matched_logical:
                        break

            # A rule with N distinct logical foods triggers if all logical foods matched
            rule_foods = rule["foods"]
            if len(rule_foods) == 1:
                # Single-food rule (e.g. "honey heated")
                if matched_logical:
                    violations.append(self._format_violation(rule, matched_logical))
            else:
                # Multi-food rule — all logical foods must be present
                if len(matched_logical) >= len(aliases_map):
                    violations.append(self._format_violation(rule, matched_logical))

        return violations

    @staticmethod
    def _format_violation(
        rule: dict[str, Any], matched: dict[str, str]
    ) -> dict[str, Any]:
        """Format a viruddha rule violation for API output."""
        return {
            "id": rule["id"],
            "foods": rule["foods"],
            "condition": rule["condition"],
            "severity": rule["severity"],
            "effects": rule["effects"],
            "note": rule.get("note", ""),
            "source": rule.get("source", ""),
            "matched_foods": matched,
        }

    def get_dosha_incompatibilities(self, dosha: str) -> dict[str, Any]:
        """Return foods / tastes / qualities to avoid for an aggravated dosha.

        Returns
        -------
        dict
            With keys: ``avoid_foods``, ``avoid_tastes``, ``avoid_qualities``,
            ``reason``, ``source``.  Empty dict if no specific rules found.
        """
        key = self._norm(dosha)
        for rule in self._dosha_incomp:
            if key in self._norm(rule.get("id", "")):
                return dict(rule)
        return {}

    def get_seasonal_incompatibilities(self, season: str) -> list[dict[str, Any]]:
        """Return incompatibilities for a given season (ritu name)."""
        key = self._norm(season)
        results: list[dict[str, Any]] = []
        for rule in self._seasonal_incomp:
            seasons_in_rule = [self._norm(s) for s in rule.get("season", [])]
            if key in seasons_in_rule:
                results.append(dict(rule))
        return results

    # ------------------------------------------------------------------ #
    # Context builder (main pipeline entry point)
    # ------------------------------------------------------------------ #

    def build_context(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        """Build a complete ontology context dict from a user payload.

        This is the main method called by the generation pipeline.

        Parameters
        ----------
        user_payload:
            Standard payload with structure::

                {
                    "profile": {"vikriti": "vata", "prakriti": "vata_pitta"},
                    "health":  {"agni": "vishama"},
                    "environment": {"season": "winter"},
                    "goals": [...],
                    "dietPreferences": {...}
                }

        Returns
        -------
        dict
            Merged ontology context ready for the generation layer.
        """
        profile = user_payload.get("profile", {})
        health = user_payload.get("health", {})
        environment = user_payload.get("environment", {})

        # --- primary dosha (vikriti) ---
        vikriti = self._norm(profile.get("vikriti", ""))
        primary_dosha: dict[str, Any] | None = None
        pacifying_rasas: list[str] = []
        aggravating_rasas: list[str] = []
        pacifying_qualities: list[str] = []
        dosha_incomp: dict[str, Any] = {}

        if vikriti and vikriti in self._doshas:
            primary_dosha = self.get_dosha(vikriti)
            pacifying_rasas = self.get_pacifying_rasas(vikriti)
            aggravating_rasas = self.get_aggravating_rasas(vikriti)
            pacifying_qualities = self.get_pacifying_qualities(vikriti)
            dosha_incomp = self.get_dosha_incompatibilities(vikriti)

        # --- prakriti (constitution) ---
        prakriti_key = self._norm(profile.get("prakriti", ""))
        dual_prakriti_advice: dict[str, Any] | None = None
        if prakriti_key in self._prakriti_dual:
            dual_prakriti_advice = self._prakriti_dual[prakriti_key]

        # --- agni ---
        agni_key = self._norm(health.get("agni", ""))
        agni_info: dict[str, Any] | None = None
        if agni_key and agni_key in self._agni_types:
            agni_info = self.get_agni_info(agni_key)

        # --- season ---
        season_input = self._norm(environment.get("season", ""))
        ritus = self.resolve_season(season_input) if season_input else []
        season_info: dict[str, Any] | None = None
        season_diet: dict[str, Any] | None = None
        seasonal_incomp: list[dict[str, Any]] = []

        if ritus:
            primary_ritu = ritus[0]
            try:
                season_info = self.get_season(primary_ritu)
                season_diet = self.get_season_diet(primary_ritu)
            except KeyError:
                pass
            for ritu in ritus:
                seasonal_incomp.extend(self.get_seasonal_incompatibilities(ritu))

        # --- viruddha categories applicable ---
        applicable_viruddha: list[str] = []
        if vikriti:
            applicable_viruddha.append(f"dosha_viruddha ({vikriti})")
        if ritus:
            applicable_viruddha.append(f"kala_viruddha ({', '.join(ritus)})")
        # samyoga/virya are always relevant
        applicable_viruddha.extend(["samyoga_viruddha", "virya_viruddha"])

        return {
            "primary_dosha": vikriti or None,
            "primary_dosha_info": primary_dosha,
            "prakriti_type": prakriti_key or None,
            "pacifying_rasas": pacifying_rasas,
            "aggravating_rasas": aggravating_rasas,
            "pacifying_qualities": pacifying_qualities,
            "agni_info": agni_info,
            "season_info": season_info,
            "season_diet": season_diet,
            "seasonal_incompatibilities": seasonal_incomp,
            "dosha_incompatibilities": dosha_incomp,
            "applicable_viruddha_categories": applicable_viruddha,
            "dual_prakriti_advice": dual_prakriti_advice,
            "goals": user_payload.get("goals", []),
            "diet_preferences": user_payload.get("dietPreferences", {}),
        }
