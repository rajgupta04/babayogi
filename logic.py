import os
import json
import numpy as np
import google.generativeai as genai
from pinecone import Pinecone

# --- Environment Variable and API Configuration --
try:
    from dotenv import load_dotenv
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY not found in environment variables.")
except (ImportError, ValueError) as e:
    print(f"Error during setup: {e}")

class DietGenerationLogic:
    """
    Handles all the core logic for generating an Ayurvedic diet plan.
    This version uses Gemini for both embeddings and generation, with Pinecone
    for vector storage and retrieval.
    """
    
    def __init__(self):
        """
        Initializes the logic class, sets up AI models, and connects to Pinecone.
        """
        print("Initializing DietGenerationLogic...")
        
        # --- Models are now all from Google Gemini ---
        self.embedding_model_name = 'models/gemini-embedding-001'
        self.generative_model = genai.GenerativeModel('gemini-2.5-pro')

        print("Connecting to Pinecone index 'ayurvedic-foods-v2'...")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("ayurvedic-foods-v2")
        print("Initialization complete. Logic instance is ready.")

    def _retrieve_suitable_foods_from_pinecone(self, user_payload: dict, top_k: int = 15) -> list[dict]:
        """
        Retrieves suitable foods from Pinecone using a hybrid approach:
        1. Metadata filtering for hard constraints (allergies).
        2. Semantic search using Gemini embeddings for contextual relevance.
        """
        metadata_filter = {}
        
        allergies = user_payload['dietPreferences']['allergies']
        if allergies:
            metadata_filter["Allergen Info"] = {"$nin": allergies}
        
        primary_vikriti = max(user_payload['profile']['vikriti'], key=user_payload['profile']['vikriti'].get)
        cuisines = user_payload['dietPreferences'].get('cuisine', [])
        
        search_query = (f"A healing food to pacify a {primary_vikriti} imbalance for a person whose goal is to "
                        f"'{user_payload['goals']['primaryGoal']}'. The food should be suitable for "
                        f"'{user_payload['health']['agni']}' digestion during the {user_payload['environment']['season']} season.")

        if cuisines:
            cuisine_str = ', '.join(cuisines)
            search_query += f" The person is accustomed to and prefers {cuisine_str} cuisine."
        
        try:
            query_embedding_response = genai.embed_content(
                model=self.embedding_model_name,
                content=search_query,
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=384  # This ensures compatibility with Pinecone
            )
            query_embedding = query_embedding_response['embedding']
        except Exception as e:
            print(f"Error generating Gemini embedding: {e}")
            return []

        try:
            print(f"Querying Pinecone with filter: {metadata_filter}")
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=metadata_filter,
                include_metadata=True
            )
            suitable_foods = [match['metadata'] for match in query_response['matches']]
            print(f"Found {len(suitable_foods)} suitable foods from Pinecone.")
            return suitable_foods
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
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
            print(f"Error calling Gemini API or parsing JSON: {e}")
            return {"error": "Failed to generate guidelines from the AI model."}

    def get_diet_plan(self, user_payload: dict) -> dict:
        """
        The main orchestrator function.
        """
        suitable_foods = self._retrieve_suitable_foods_from_pinecone(user_payload)
        
        if not suitable_foods:
            return {
                "error": "Could not find suitable foods.",
                "message": "Based on your specific profile and allergies, we couldn't find any recommended foods in our database."
            }
        
        vikriti_scores = user_payload['profile']['vikriti']
        primary_dosha = max(vikriti_scores, key=vikriti_scores.get).capitalize()
        secondary_doshas = [d.capitalize() for d, s in sorted(vikriti_scores.items(), key=lambda item: item[1], reverse=True) if d != primary_dosha]

        food_inspiration_list = [
            {'name': f.get('Dish Name'), 'category': f.get('Category')} for f in suitable_foods
        ]

        prompt = f"""
        You are an expert Ayurvedic consultant. Your task is to generate a comprehensive, personalized wellness guide based on the user's profile.
        The output MUST be a single, valid JSON object conforming to the specified structure. Do not include any text outside the JSON object.

        **USER PROFILE:**
        - Primary Imbalance (Vikriti): {primary_dosha}
        - Secondary Imbalances: {', '.join(secondary_doshas)}
        - Allergies: {', '.join(user_payload['dietPreferences']['allergies']) if user_payload['dietPreferences']['allergies'] else "None"}
        - Dietary Preference: {user_payload['dietPreferences']['dietType']}
        - Cuisine Preference (Satmaya): {', '.join(user_payload['dietPreferences']['cuisine'])}

        **RECOMMENDED FOODS FOR INSPIRATION:**
        Base your "can_eat" suggestions on this list of foods, which have been pre-selected as highly suitable for the user from our expert database. Distribute them into the correct categories.
        - {json.dumps(food_inspiration_list, indent=2)}

        **INSTRUCTIONS:**
        1.  Analyze the user profile to determine the dominant dosha and overall health picture.
        2.  Populate every field in the provided JSON structure with logical, expert Ayurvedic advice.
        3.  The "food_guidelines" must contain specific lists for "can_eat" and "avoid".
            - Each list must include at least 10 and at most 20 items.
            - Use the inspiration list for "can_eat" suggestions.
            - Ensure no duplicates across categories.
            - Notes must be highly relevant, actionable, and tailored to the user’s dosha and goals.
        4.  The "dosha_alerts" should provide specific warnings related to the user's imbalances.

        **JSON OUTPUT STRUCTURE (Strict):**
        {{
          "user_profile": {{ "dosha": "{primary_dosha}-dominant", "secondary_doshas": {json.dumps(secondary_doshas)}, "allergies": {json.dumps(user_payload['dietPreferences']['allergies'])}, "preferences": ["{user_payload['dietPreferences']['dietType']}"], "cuisine": {json.dumps(user_payload['dietPreferences']['cuisine'])} }},
          "food_guidelines": {{ "grains": {{ "can_eat": [], "avoid": [], "notes": "" }}, "vegetables": {{ "can_eat": [], "avoid": [], "notes": "" }}, "fruits": {{ "can_eat": [], "avoid": [], "notes": "" }}, "proteins": {{ "can_eat": [], "avoid": [], "notes": "" }}, "dairy": {{ "can_eat": [], "avoid": [], "notes": "" }}, "spices": {{ "can_use": [], "avoid": [], "notes": "" }}, "beverages": {{ "can_drink": [], "avoid": [] }} }},
          "nutrient_guidelines": {{ "carbohydrates": {{ "suggested_range_percent": "40-50%", "notes": "" }}, "proteins": {{ "suggested_range_percent": "20-25%", "notes": "" }}, "fats": {{ "suggested_range_percent": "20-25%", "notes": "" }}, "hydration": {{ "water_intake_liters": "2-3", "notes": "" }} }},
          "meal_timing": {{ "breakfast": "7-9 AM", "lunch": "12-2 PM (main meal)", "snack": "3-4 PM", "dinner": "6-8 PM (light meal)", "notes": "" }},
          "portion_guidelines": {{ "grains": "1-2 cups cooked per meal", "vegetables": "1-2 cups per meal", "fruits": "1 serving per snack", "proteins": "½-1 cup cooked legumes per meal", "fats": "1-2 tsp per meal" }},
          "lifestyle_recommendations": {{ "exercise": "", "sleep": "", "mental_health": "", "detox": "" }},
          "dosha_alerts": [ {{ "dosha": "Kapha", "alert": "" }}, {{ "dosha": "Vata", "alert": "" }}, {{ "dosha": "Pitta", "alert": "" }} ],
          "flexibility_options": {{ "food_rotation": "", "seasonal_adjustments": "", "spice_variations": "" }}
        }}
        """
        
        guidelines = self._generate_guidelines_with_gemini(prompt)
        return guidelines
