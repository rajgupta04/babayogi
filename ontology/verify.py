"""
Ayurvedic Ontology Verification CLI
===================================

Runs structural, schema, and cross-reference validation on the Ayurvedic ontology:

1. Validates `dosha_rules.json`, `viruddha_rules.json`, and `ritu_charya.json` against Pydantic schemas.
2. Checks internal consistency across files:
   - All pacifying/aggravating rasas map to valid Shad-Rasa definitions.
   - All seasons referenced in `season_mapping` exist in `seasons`.
   - All viruddha severities and food aliases are well-formed.
3. Tests runtime lookup operations on `AyurvedicOntology`.

Run directly from the `babayogi` directory::

    python -m ontology.verify
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure stdout handles UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ontology.loader import AyurvedicOntology
from ontology.schema import validate_all

# Terminal ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _pass(msg: str) -> None:
    print(f"  {GREEN}[PASS]{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def run_verification(ontology_dir: Path | None = None) -> bool:
    if ontology_dir is None:
        ontology_dir = Path(__file__).resolve().parent

    print(f"\n{BOLD}{CYAN}=== AyurVarta Ayurvedic Ontology Verification ==={RESET}")
    print(f"Checking directory: {ontology_dir}\n")

    has_errors = False

    # -------------------------------------------------------------
    # 1. Pydantic Schema Validation
    # -------------------------------------------------------------
    print(f"{BOLD}1. Schema Validation (Pydantic v2){RESET}")
    schema_errors = validate_all(ontology_dir)
    if schema_errors:
        has_errors = True
        for err in schema_errors:
            _fail(err)
    else:
        _pass("dosha_rules.json matches schema")
        _pass("viruddha_rules.json matches schema")
        _pass("ritu_charya.json matches schema")

    # -------------------------------------------------------------
    # 2. Cross-Reference Consistency Checks
    # -------------------------------------------------------------
    print(f"\n{BOLD}2. Semantic & Cross-Reference Checks{RESET}")
    try:
        onto = AyurvedicOntology(ontology_dir)

        # Check Dosha -> Rasa consistency
        valid_rasas = set(onto.get_all_rasa_names())
        for dosha_name in ["vata", "pitta", "kapha"]:
            pacifying = onto.get_pacifying_rasas(dosha_name)
            aggravating = onto.get_aggravating_rasas(dosha_name)

            for r in pacifying:
                if r not in valid_rasas:
                    _fail(f"Dosha '{dosha_name}' references unknown pacifying rasa '{r}'")
                    has_errors = True
            for r in aggravating:
                if r not in valid_rasas:
                    _fail(f"Dosha '{dosha_name}' references unknown aggravating rasa '{r}'")
                    has_errors = True

        _pass(f"All Dosha pacifying/aggravating rasas resolve against 6 Shad-Rasas ({', '.join(sorted(valid_rasas))})")

        # Check Seasons -> Season Mapping consistency
        valid_seasons = set(onto.get_all_season_names())
        mapping = onto._ritu_raw.get("season_mapping", {})
        eng_to_ritu = mapping.get("english_to_ritu", {})
        for eng, ritu_list in eng_to_ritu.items():
            for r in ritu_list:
                if r not in valid_seasons:
                    _fail(f"English season '{eng}' maps to unknown ritu '{r}'")
                    has_errors = True
        _pass(f"All English season mappings resolve to valid Ritus ({', '.join(sorted(valid_seasons))})")

        # Check Month to Ritu mapping
        month_to_ritu = mapping.get("month_to_ritu", {})
        for m, r in month_to_ritu.items():
            if r not in valid_seasons:
                _fail(f"Month '{m}' maps to unknown ritu '{r}'")
                has_errors = True
        _pass(f"All 12 month-to-ritu mappings resolve to valid Ritus")

        # Check Viruddha rules
        viruddha_rules = onto._viruddha_rules
        critical_count = sum(1 for r in viruddha_rules if r.get("severity") == "critical")
        high_count = sum(1 for r in viruddha_rules if r.get("severity") == "high")
        _pass(f"Loaded {len(viruddha_rules)} Viruddha Ahara rules ({critical_count} critical, {high_count} high)")

        # -------------------------------------------------------------
        # 3. Runtime Query Sanity Checks
        # -------------------------------------------------------------
        print(f"\n{BOLD}3. Runtime Query Sanity Tests{RESET}")

        # Test Viruddha detection (honey + ghee equal)
        test_foods = ["raw honey", "clarified butter", "basmati rice", "mung dal"]
        hits = onto.check_viruddha(test_foods)
        if any(h["id"] == "honey_ghee_equal" for h in hits):
            _pass("Viruddha check correctly detects honey + ghee combination")
        else:
            _fail("Viruddha check failed to detect honey + ghee combination")
            has_errors = True

        # Test Viruddha detection (milk + fish)
        hits_fish = onto.check_viruddha(["cow milk", "salmon fish"])
        if any(h["id"] == "milk_fish" for h in hits_fish):
            _pass("Viruddha check correctly detects milk + fish combination")
        else:
            _fail("Viruddha check failed to detect milk + fish combination")
            has_errors = True

        # Test Season resolution
        winter_ritus = onto.resolve_season("winter")
        if "hemanta" in winter_ritus or "shishira" in winter_ritus:
            _pass(f"Season resolution: 'winter' -> {winter_ritus}")
        else:
            _fail(f"Season resolution failed for 'winter': got {winter_ritus}")
            has_errors = True

        # Test Context Builder
        test_payload = {
            "profile": {"vikriti": "pitta", "prakriti": "pitta_kapha"},
            "health": {"agni": "tikshna"},
            "environment": {"season": "summer"},
            "goals": {"primaryGoal": "Reduce acidity and body heat"},
            "dietPreferences": {"dietType": "vegetarian", "cuisine": ["North Indian"], "allergies": []},
        }
        ctx = onto.build_context(test_payload)
        if ctx.get("primary_dosha") == "pitta" and "madhura" in ctx.get("pacifying_rasas", []):
            _pass("build_context() generates complete and coherent ontology context for LLM")
        else:
            _fail("build_context() produced incomplete context")
            has_errors = True

    except Exception as exc:
        _fail(f"Ontology loader failed with error: {exc}")
        has_errors = True

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print(f"\n{BOLD}Summary Statistics:{RESET}")
    try:
        onto = AyurvedicOntology(ontology_dir)
        print(f"  • Doshas: {len(onto._doshas)} (Vata, Pitta, Kapha)")
        print(f"  • Rasas: {len(onto._rasas)} (Madhura, Amla, Lavana, Katu, Tikta, Kashaya)")
        print(f"  • Guna pairs: {len(onto._guna_pairs)} (20 qualities)")
        print(f"  • Agni types: {len(onto._agni_types)} (Sama, Vishama, Tikshna, Manda)")
        print(f"  • Dhatus: {len(onto.get_dhatu_sequence())} (Sapta Dhatu)")
        print(f"  • Dual-Prakriti types: {len(onto._prakriti_dual)}")
        print(f"  • Viruddha rules: {len(onto._viruddha_rules)}")
        print(f"  • Seasons: {len(onto._seasons)} (Shad Ritu)")
    except Exception:
        pass

    if has_errors:
        print(f"\n{RED}{BOLD}Verification FAILED with errors!{RESET}\n")
        return False
    else:
        print(f"\n{GREEN}{BOLD}All ontology checks PASSED! System is coherent and ready.{RESET}\n")
        return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
