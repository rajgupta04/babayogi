import os
import json
import logging
from typing import Any, Dict, List
import numpy as np
import google.generativeai as genai
from pinecone import Pinecone

from ontology.loader import AyurvedicOntology

logger = logging.getLogger(__name__)

# --- Environment Variable and API Configuration --
try:
    from dotenv import load_dotenv
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY not found in environment variables.")
except (ImportError, ValueError) as e:
    logger.warning("Error during AI / vector DB setup: %s", e)


class DietGenerationLogic:
    """
    Handles core logic for generating an Ayurvedic diet plan.
    Integrates the Ayurvedic Ontology (Tridosha, Shad-Rasa, Vimshati Guna,
    Chaturvidha Agni, Ritucharya, and Viruddha Ahara rules) with Pinecone vector
    retrieval and Google Gemini generation.
    """

    def __init__(self):
        """
        Initializes AI models, Pinecone connection, and Ayurvedic ontology.
        """
        logger.info("Initializing DietGenerationLogic...")
        self.ontology = AyurvedicOntology()

        # Models from Google Gemini
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.generative_model_name = os.getenv("GENERATIVE_MODEL", "gemini-2.5-pro")
        self.generative_model = genai.GenerativeModel(self.generative_model_name)

        pinecone_index = os.getenv("PINECONE_INDEX_NAME", "ayurvedic-foods-v2")
        logger.info("Connecting to Pinecone index '%s'...", pinecone_index)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(pinecone_index)
        logger.info("Initialization complete. Logic instance is ready.")

    def _retrieve_suitable_foods_from_pinecone(self, user_payload: dict, top_k: int = 15) -> list[dict]:
        """
        Retrieves suitable foods from Pinecone using a hybrid approach:
        1. Metadata filtering for hard constraints (allergies).
        2. Semantic search expanded with ontology-derived pacifying tastes (Rasas) and qualities (Gunas).
        """
        metadata_filter = {}

        allergies = user_payload.get("dietPreferences", {}).get("allergies", [])
        if allergies:
            metadata_filter["Allergen Info"] = {"$nin": allergies}

        vikriti_map = user_payload.get("profile", {}).get("vikriti", {})
        primary_vikriti = max(vikriti_map, key=vikriti_map.get).lower() if vikriti_map else "vata"
        cuisines = user_payload.get("dietPreferences", {}).get("cuisine", [])

        # Enrich semantic search query with classical ontology rules
        pacifying_rasas = self.ontology.get_pacifying_rasas(primary_vikriti)
        pacifying_gunas = self.ontology.get_pacifying_qualities(primary_vikriti)
        rasa_text = f" with {', '.join(pacifying_rasas)} tastes" if pacifying_rasas else ""
        guna_text = f" and {', '.join(pacifying_gunas[:3])} qualities" if pacifying_gunas else ""

        season = user_payload.get("environment", {}).get("season", "general")
        agni = user_payload.get("health", {}).get("agni", "moderate")
        goal = user_payload.get("goals", {}).get("primaryGoal", "optimal wellness")

        search_query = (
            f"A healing Ayurvedic food{rasa_text}{guna_text} to pacify a {primary_vikriti.capitalize()} "
            f"dosha imbalance for a person whose goal is '{goal}'. "
            f"The food should be suitable for '{agni}' digestion during the {season} season."
        )

        if cuisines:
            cuisine_str = ", ".join(cuisines)
            search_query += f" The person is accustomed to and prefers {cuisine_str} cuisine."

        try:
            query_embedding_response = genai.embed_content(
                model=self.embedding_model_name,
                content=search_query,
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=384,  # Pinecone index dimensionality
            )
            query_embedding = query_embedding_response["embedding"]
        except Exception as e:
            logger.error("Error generating Gemini embedding: %s", e)
            return []

        try:
            logger.info("Querying Pinecone with filter: %s", metadata_filter)
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=metadata_filter,
                include_metadata=True,
            )
            suitable_foods = [match["metadata"] for match in query_response.get("matches", [])]
            logger.info("Found %d suitable foods from Pinecone.", len(suitable_foods))
            return suitable_foods
        except Exception as e:
            logger.error("Error querying Pinecone: %s", e)
            return []

    def _generate_guidelines_with_gemini(self, prompt: str) -> dict:
        """
        Calls the Gemini API to generate the diet guidelines based on the prompt.
        """
        try:
            response = self.generative_model.generate_content(prompt)
            text_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(text_response)
        except Exception as e:
            logger.error("Error calling Gemini API or parsing JSON: %s", e)
            return {"error": "Failed to generate guidelines from the AI model."}

    def _extract_all_recommended_foods(self, guidelines: dict) -> list[str]:
        """Extract flat list of all recommended food names from generated guidelines."""
        food_guidelines = guidelines.get("food_guidelines", {})
        names = []
        for category_data in food_guidelines.values():
            if isinstance(category_data, dict):
                can_eat = category_data.get("can_eat", [])
                for item in can_eat:
                    if isinstance(item, str):
                        names.append(item)
                    elif isinstance(item, dict) and "name" in item:
                        names.append(item["name"])
        return names

    def get_diet_plan(self, user_payload: dict) -> dict:
        """
        Main orchestrator: retrieves foods, injects ontology context,
        generates structured plan, and applies Viruddha Ahara safety guardrails.
        """
        suitable_foods = self._retrieve_suitable_foods_from_pinecone(user_payload)

        if not suitable_foods:
            return {
                "error": "Could not find suitable foods.",
                "message": "Based on your specific profile and allergies, we couldn't find recommended foods in our database.",
            }

        vikriti_scores = user_payload.get("profile", {}).get("vikriti", {})
        primary_dosha = max(vikriti_scores, key=vikriti_scores.get).capitalize() if vikriti_scores else "Vata"
        secondary_doshas = [
            d.capitalize()
            for d, s in sorted(vikriti_scores.items(), key=lambda item: item[1], reverse=True)
            if d.capitalize() != primary_dosha
        ]

        # Build grounded ontology context from verified Ayurvedic sources
        ontology_context = self.ontology.build_context(user_payload)
        pacifying_rasas = ontology_context.get("pacifying_rasas", [])
        aggravating_rasas = ontology_context.get("aggravating_rasas", [])
        pacifying_gunas = ontology_context.get("pacifying_qualities", [])

        food_inspiration_list = [
            {"name": f.get("Dish Name"), "category": f.get("Category")}
            for f in suitable_foods
        ]

        prompt = f"""
        You are an expert Ayurvedic dietetics consultant. Your task is to generate a comprehensive, personalized wellness guide based on classical Ayurvedic principles and the user's constitution.
        The output MUST be a single, valid JSON object conforming to the specified structure. Do not include any text outside the JSON object.

        **PATIENT AYURVEDIC PROFILE:**
        - Primary Imbalance (Vikriti): {primary_dosha}
        - Secondary Imbalances: {', '.join(secondary_doshas)}
        - Allergies: {', '.join(user_payload.get('dietPreferences', {}).get('allergies', [])) if user_payload.get('dietPreferences', {}).get('allergies') else "None"}
        - Dietary Preference: {user_payload.get('dietPreferences', {}).get('dietType', 'vegetarian')}
        - Cuisine Preference (Satmaya): {', '.join(user_payload.get('dietPreferences', {}).get('cuisine', []))}
        - Agni (Digestive Fire): {user_payload.get('health', {}).get('agni', 'moderate')}
        - Season (Ritu): {user_payload.get('environment', {}).get('season', 'general')}

        **DETERMINISTIC AYURVEDIC PRINCIPLES (Strict Rules from Charaka & Vagbhata):**
        - Tastes to FAVOR (Pacifying Rasas): {', '.join(pacifying_rasas)}
        - Tastes to MINIMIZE / REDUCE (Aggravating Rasas): {', '.join(aggravating_rasas)}
        - Qualities to FAVOR (Pacifying Gunas): {', '.join(pacifying_gunas[:4])}
        - Important Safety Rule: DO NOT combine honey and ghee in equal quantities, and DO NOT combine fish with milk (Viruddha Ahara).

        **RECOMMENDED FOODS FOR INSPIRATION (Pre-verified from Pinecone Database):**
        Base your "can_eat" suggestions on this curated list:
        - {json.dumps(food_inspiration_list, indent=2)}

        **INSTRUCTIONS:**
        1. Populate every field in the provided JSON structure with logical, expert Ayurvedic advice.
        2. In "food_guidelines", strictly recommend foods that align with the pacifying rasas ({', '.join(pacifying_rasas)}).
        3. Each category in "food_guidelines" must contain specific lists for "can_eat" (10-20 items) and "avoid" (10-20 items).
        4. Provide actionable, dosha-tailored meal timings and lifestyle tips.

        **JSON OUTPUT STRUCTURE (Strict):**
        {{
          "user_profile": {{
            "dosha": "{primary_dosha}-dominant",
            "secondary_doshas": {json.dumps(secondary_doshas)},
            "allergies": {json.dumps(user_payload.get('dietPreferences', {}).get('allergies', []))},
            "preferences": ["{user_payload.get('dietPreferences', {}).get('dietType', 'vegetarian')}"],
            "cuisine": {json.dumps(user_payload.get('dietPreferences', {}).get('cuisine', []))}
          }},
          "food_guidelines": {{
            "grains": {{ "can_eat": [], "avoid": [], "notes": "" }},
            "vegetables": {{ "can_eat": [], "avoid": [], "notes": "" }},
            "fruits": {{ "can_eat": [], "avoid": [], "notes": "" }},
            "proteins": {{ "can_eat": [], "avoid": [], "notes": "" }},
            "dairy": {{ "can_eat": [], "avoid": [], "notes": "" }},
            "spices": {{ "can_use": [], "avoid": [], "notes": "" }},
            "beverages": {{ "can_drink": [], "avoid": [] }}
          }},
          "nutrient_guidelines": {{
            "carbohydrates": {{ "suggested_range_percent": "40-50%", "notes": "" }},
            "proteins": {{ "suggested_range_percent": "20-25%", "notes": "" }},
            "fats": {{ "suggested_range_percent": "20-25%", "notes": "" }},
            "hydration": {{ "water_intake_liters": "2-3", "notes": "" }}
          }},
          "meal_timing": {{
            "breakfast": "7-9 AM",
            "lunch": "12-2 PM (main meal)",
            "snack": "3-4 PM",
            "dinner": "6-8 PM (light meal)",
            "notes": ""
          }},
          "portion_guidelines": {{
            "grains": "1-2 cups cooked per meal",
            "vegetables": "1-2 cups per meal",
            "fruits": "1 serving per snack",
            "proteins": "½-1 cup cooked legumes per meal",
            "fats": "1-2 tsp per meal"
          }},
          "lifestyle_recommendations": {{
            "exercise": "",
            "sleep": "",
            "mental_health": "",
            "detox": ""
          }},
          "dosha_alerts": [
            {{ "dosha": "Kapha", "alert": "" }},
            {{ "dosha": "Vata", "alert": "" }},
            {{ "dosha": "Pitta", "alert": "" }}
          ],
          "flexibility_options": {{
            "food_rotation": "",
            "seasonal_adjustments": "",
            "spice_variations": ""
          }}
        }}
        """

        guidelines = self._generate_guidelines_with_gemini(prompt)

        # Apply Viruddha Ahara Safety Guardrails
        if isinstance(guidelines, dict) and "error" not in guidelines:
            all_foods = self._extract_all_recommended_foods(guidelines)
            violations = self.ontology.check_viruddha(all_foods)

            # Attach verified safety warnings and ontology principles
            guidelines["viruddha_safety_checks"] = {
                "checked_foods_count": len(all_foods),
                "incompatible_combinations_found": len(violations),
                "warnings": violations,
            }
            guidelines["ayurvedic_principles"] = {
                "pacifying_rasas": pacifying_rasas,
                "aggravating_rasas": aggravating_rasas,
                "pacifying_gunas": pacifying_gunas[:4],
                "agni_type": user_payload.get("health", {}).get("agni"),
                "season": user_payload.get("environment", {}).get("season"),
            }

        return guidelines
