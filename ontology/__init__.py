"""
Ayurvedic Ontology Package
==========================

Contains the complete Ayurvedic knowledge base used by the AyurVarta RAG pipeline:

- **dosha_rules.json** — Tridosha (Vata/Pitta/Kapha), Shad-Rasa (6 tastes),
  Vimshati Guna (20 qualities), Agni types, Dhatus, and dual-Prakriti types.
  
- **viruddha_rules.json** — Viruddha Ahara (incompatible food combinations)
  from Charaka Samhita Sutra 26, including 18 categories of incompatibility
  and specific food-pair rules with severity levels.
  
- **ritu_charya.json** — Ritucharya (seasonal dietary regimens) for all 6
  Ayurvedic seasons with dosha state tracking, Agni strength, and food/lifestyle
  recommendations.

Usage::

    from ontology.loader import AyurvedicOntology

    ontology = AyurvedicOntology()

    # Get pacifying tastes for a dosha
    rasas = ontology.get_pacifying_rasas("pitta")  # ["madhura", "tikta", "kashaya"]

    # Check food incompatibilities
    violations = ontology.check_viruddha(["honey", "ghee", "milk", "fish"])

    # Build full context for the generation pipeline
    context = ontology.build_context(user_payload)

Editing & Verification::

    # After editing any JSON file, run:
    python -m ontology.verify
"""

from ontology.loader import AyurvedicOntology

__all__ = ["AyurvedicOntology"]
