"""
Ayurvedic Food Data
====================

Food data is stored in the Pinecone vector database (index: 'ayurvedic-foods-v2')
rather than in local JSON files.

This directory is reserved for optional local food property overrides or
supplementary data that enriches the Pinecone metadata.

The ontology's dosha_rules.json, viruddha_rules.json, and ritu_charya.json
provide the *rules* that apply to foods (which tastes pacify which dosha,
which combinations are incompatible, etc.), while Pinecone stores the
individual food entries with their embeddings and metadata.

Future use:
- Local food property overrides (food_overrides.json)
- Allergen synonym mappings (allergen_aliases.json)
- Regional cuisine → dosha suitability mappings
"""
