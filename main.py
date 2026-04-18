import hashlib
import hmac
import math
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel, Field
from pymongo import DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

from logic import DietGenerationLogic


load_dotenv()


def _parse_csv_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _get_mongo_client() -> MongoClient:
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI is not configured.")

    # Assumption: this API can run serverless (Vercel), so keep the pool small and idle cleanup fast.
    return MongoClient(
        mongo_uri,
        maxPoolSize=int(os.getenv("MONGO_MAX_POOL_SIZE", "5")),
        minPoolSize=int(os.getenv("MONGO_MIN_POOL_SIZE", "0")),
        maxIdleTimeMS=int(os.getenv("MONGO_MAX_IDLE_MS", "30000")),
        connectTimeoutMS=int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "10000")),
        serverSelectionTimeoutMS=int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000")),
        socketTimeoutMS=int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "30000")),
    )


MONGO_DB_NAME = os.getenv("MONGODB_DB_NAME", "ayurvarta")
MONGO_CLIENT: Optional[MongoClient] = None
DB = None
INDEXES_READY = False
ALLOWED_ASSESSMENT_TYPES = {"prakriti", "vikriti", "agni"}


def _get_collections():
    global MONGO_CLIENT, DB, INDEXES_READY

    if DB is None:
        MONGO_CLIENT = _get_mongo_client()
        DB = MONGO_CLIENT[MONGO_DB_NAME]

    users = DB["users"]
    diet_jobs = DB["diet_jobs"]
    diet_logs = DB["diet_logs"]
    assessment_results = DB["assessment_results"]

    if not INDEXES_READY:
        users.create_index("email", unique=True)
        diet_jobs.create_index([("user_id", DESCENDING), ("created_at", DESCENDING)])
        diet_jobs.create_index([("status", DESCENDING), ("profile_signature", DESCENDING), ("created_at", DESCENDING)])
        diet_logs.create_index([("user_id", DESCENDING), ("logged_at", DESCENDING)])
        assessment_results.create_index([("user_id", DESCENDING), ("type", DESCENDING), ("created_at", DESCENDING)])
        INDEXES_READY = True

    return users, diet_jobs, diet_logs, assessment_results


JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "168"))
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

security = HTTPBearer(auto_error=False)
_diet_logic: Optional[DietGenerationLogic] = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_password(password: str, salt_hex: Optional[str] = None) -> tuple[str, str]:
    salt_bytes = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 150000)
    return password_hash.hex(), salt_bytes.hex()


def _verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    candidate_hash, _ = _hash_password(password, stored_salt)
    return hmac.compare_digest(candidate_hash, stored_hash)


def _create_access_token(user_id: str, email: str) -> str:
    now = _utc_now()
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from exc


def _serialize_user(user_doc: dict) -> dict:
    return {
        "id": str(user_doc["_id"]),
        "email": user_doc.get("email"),
        "displayName": user_doc.get("display_name") or user_doc.get("email", "").split("@")[0],
        "role": user_doc.get("role", "Dietitian"),
        "provider": user_doc.get("provider", "local"),
    }


def _serialize_job(job_doc: dict) -> dict:
    return {
        "id": str(job_doc["_id"]),
        "status": job_doc.get("status", "queued"),
        "message": job_doc.get("message", ""),
        "result": job_doc.get("result"),
        "error": job_doc.get("error"),
        "source": job_doc.get("source", "generated"),
        "matchedFromJobId": str(job_doc["matched_from_job_id"]) if job_doc.get("matched_from_job_id") else None,
        "matchScore": job_doc.get("match_score"),
        "profileSignature": job_doc.get("profile_signature"),
        "createdAt": job_doc.get("created_at"),
        "updatedAt": job_doc.get("updated_at"),
    }


def _serialize_diet_log(log_doc: dict) -> dict:
    return {
        "id": str(log_doc["_id"]),
        "mealType": log_doc.get("meal_type"),
        "mealName": log_doc.get("meal_name"),
        "notes": log_doc.get("notes"),
        "adherence": log_doc.get("adherence"),
        "loggedAt": log_doc.get("logged_at"),
        "createdAt": log_doc.get("created_at"),
    }


def _serialize_assessment_result(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "type": doc.get("type"),
        "scores": doc.get("scores", {}),
        "schema": doc.get("schema"),
        "disease": doc.get("disease"),
        "capturedAt": doc.get("captured_at"),
        "createdAt": doc.get("created_at"),
    }


def _ratio_triplet(scores: Dict[str, Any], keys: List[str]) -> List[float]:
    values = [float(scores.get(key, 0) or 0) for key in keys]
    total = sum(v for v in values if v > 0)
    if total <= 0:
        return [0.0 for _ in keys]
    return [v / total for v in values]


def _agni_vector(agni: str) -> List[float]:
    value = (agni or "").strip().lower()
    if value in {"irregular", "vishama"}:
        return [1.0, 0.0, 0.0]
    if value in {"strong", "tikshna"}:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _dominant_key(scores: Dict[str, Any]) -> str:
    if not scores:
        return "unknown"
    normalized = [(str(k).lower(), float(v or 0)) for k, v in scores.items()]
    if not any(v > 0 for _, v in normalized):
        return "unknown"
    return max(normalized, key=lambda item: item[1])[0]


def _build_profile_signature(payload: Dict[str, Any]) -> str:
    profile = payload.get("profile") or {}
    health = payload.get("health") or {}
    diet_preferences = payload.get("dietPreferences") or {}
    environment = payload.get("environment") or {}

    primary_prakriti = _dominant_key(profile.get("prakriti") or {})
    primary_vikriti = _dominant_key(profile.get("vikriti") or {})
    agni = (health.get("agni") or "unknown").strip().lower()
    diet_type = (diet_preferences.get("dietType") or "unknown").strip().lower()
    season = (environment.get("season") or "unknown").strip().lower()

    return f"{primary_prakriti}:{primary_vikriti}:{agni}:{diet_type}:{season}"


def _build_profile_vector(payload: Dict[str, Any]) -> List[float]:
    profile = payload.get("profile") or {}
    health = payload.get("health") or {}

    prakriti_vector = _ratio_triplet(profile.get("prakriti") or {}, ["vata", "pitta", "kapha"])
    vikriti_vector = _ratio_triplet(profile.get("vikriti") or {}, ["vata", "pitta", "kapha"])
    agni_vector = _agni_vector(health.get("agni") or "")
    return [*prakriti_vector, *vikriti_vector, *agni_vector]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_matching_completed_job(request_payload: Dict[str, Any], min_similarity: float = 0.97) -> Optional[dict]:
    _, diet_jobs_collection, _, _ = _get_collections()
    signature = _build_profile_signature(request_payload)
    current_vector = _build_profile_vector(request_payload)

    candidates = list(
        diet_jobs_collection.find(
            {
                "status": "completed",
                "result": {"$ne": None},
                "profile_signature": signature,
            }
        )
        .sort("created_at", DESCENDING)
        .limit(50)
    )

    best_match = None
    best_score = 0.0
    for candidate in candidates:
        candidate_vector = candidate.get("profile_vector") or _build_profile_vector(candidate.get("request_payload") or {})
        score = _cosine_similarity(current_vector, candidate_vector)
        if score >= min_similarity and score > best_score:
            best_match = candidate
            best_score = score

    if not best_match:
        return None

    best_match["_match_score"] = round(best_score, 4)
    return best_match


def _get_diet_logic() -> DietGenerationLogic:
    global _diet_logic
    if _diet_logic is None:
        _diet_logic = DietGenerationLogic()
    return _diet_logic


def _verify_google_token(id_token: str) -> dict:
    request = google_requests.Request()
    audience = GOOGLE_CLIENT_ID or None
    try:
        return google_id_token.verify_oauth2_token(id_token, request, audience=audience)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google token") from exc


def _to_object_id(value: str, field_name: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid {field_name}") from exc


def _get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    token_payload = _decode_token(credentials.credentials)
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    users_collection, _, _, _ = _get_collections()
    user = users_collection.find_one({"_id": _to_object_id(user_id, "user id")})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user


class DoshaScores(BaseModel):
    vata: int = Field(..., example=7, description="Vata dosha score")
    pitta: int = Field(..., example=3, description="Pitta dosha score")
    kapha: int = Field(..., example=2, description="Kapha dosha score")


class Profile(BaseModel):
    prakriti: DoshaScores = Field(...)
    vikriti: DoshaScores = Field(...)


class Health(BaseModel):
    agni: str = Field(..., example="weak")
    ama: str = Field(..., example="moderate")


class DietPreferences(BaseModel):
    dietType: str = Field(..., example="vegetarian")
    allergies: List[str] = Field(default=[], example=["Dairy"])
    cuisine: List[str] = Field(..., example=["North Indian"])


class Environment(BaseModel):
    season: str = Field(..., example="winter")


class Goals(BaseModel):
    primaryGoal: str = Field(..., example="Improve digestion and reduce bloating")


class DietRequest(BaseModel):
    profile: Profile
    health: Health
    dietPreferences: DietPreferences
    environment: Environment
    goals: Goals


class SignupRequest(BaseModel):
    email: str
    password: str
    displayName: str = Field(..., min_length=1)
    role: str = "Dietitian"


class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleAuthRequest(BaseModel):
    idToken: str
    role: str = "Dietitian"


class ForgotPasswordRequest(BaseModel):
    email: str


class DietJobStartRequest(BaseModel):
    dietRequest: DietRequest


class DietLogCreateRequest(BaseModel):
    mealType: str
    mealName: str
    notes: Optional[str] = ""
    adherence: Optional[int] = Field(default=None, ge=1, le=5)
    loggedAt: Optional[datetime] = None


class AssessmentResultCreateRequest(BaseModel):
    type: str
    scores: Dict[str, float]
    schema: Optional[str] = None
    disease: Optional[Dict[str, Any]] = None
    capturedAt: Optional[datetime] = None


app = FastAPI(
    title="Aayur.AI - Personalized Wellness Guide",
    version="4.0.0",
)

cors_origins = _parse_csv_env(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,https://localhost:3000,https://localhost:5173",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_diet_job(job_id: ObjectId, user_id: ObjectId, request_payload: dict) -> None:
    _, diet_jobs_collection, _, _ = _get_collections()

    profile_signature = _build_profile_signature(request_payload)
    profile_vector = _build_profile_vector(request_payload)

    diet_jobs_collection.update_one(
        {"_id": job_id, "user_id": user_id},
        {
            "$set": {
                "status": "running",
                "message": "Analyzing your profile and generating plan...",
                "profile_signature": profile_signature,
                "profile_vector": profile_vector,
                "updated_at": _utc_now(),
            }
        },
    )

    try:
        # Reuse an already generated plan for highly similar users in the same cohort.
        matched = _find_matching_completed_job(request_payload)
        if matched and matched.get("result"):
            diet_jobs_collection.update_one(
                {"_id": job_id, "user_id": user_id},
                {
                    "$set": {
                        "status": "completed",
                        "result": matched.get("result"),
                        "source": "matched",
                        "matched_from_job_id": matched.get("_id"),
                        "match_score": matched.get("_match_score"),
                        "message": "Diet ready (matched with a highly similar profile).",
                        "updated_at": _utc_now(),
                    }
                },
            )
            return

        result = _get_diet_logic().get_diet_plan(request_payload)
        if isinstance(result, dict) and "error" in result:
            diet_jobs_collection.update_one(
                {"_id": job_id, "user_id": user_id},
                {
                    "$set": {
                        "status": "failed",
                        "error": result,
                        "source": "generated",
                        "message": "Diet generation failed.",
                        "updated_at": _utc_now(),
                    }
                },
            )
            return

        diet_jobs_collection.update_one(
            {"_id": job_id, "user_id": user_id},
            {
                "$set": {
                    "status": "completed",
                    "result": result,
                        "source": "generated",
                        "match_score": None,
                        "matched_from_job_id": None,
                    "message": "Congratulations! Your diet is ready.",
                    "updated_at": _utc_now(),
                }
            },
        )
    except Exception as exc:
        diet_jobs_collection.update_one(
            {"_id": job_id, "user_id": user_id},
            {
                "$set": {
                    "status": "failed",
                    "error": {"message": str(exc)},
                        "source": "generated",
                    "message": "Diet generation failed.",
                    "updated_at": _utc_now(),
                }
            },
        )


@app.get("/")
async def read_root():
    return {"message": "API is running.", "version": "4.0.0"}


@app.get("/health")
async def health_check():
    return {"ok": True, "time": _utc_now()}


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    return FileResponse("favicon.ico")


@app.post("/auth/signup")
async def signup(request: SignupRequest):
    users_collection, _, _, _ = _get_collections()
    email = request.email.strip().lower()
    if not email or not request.password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email and password are required")

    password_hash, password_salt = _hash_password(request.password)
    now = _utc_now()
    user_doc = {
        "email": email,
        "display_name": request.displayName.strip(),
        "role": request.role,
        "provider": "local",
        "password_hash": password_hash,
        "password_salt": password_salt,
        "created_at": now,
        "updated_at": now,
        "last_login_at": now,
    }

    try:
        insert_result = users_collection.insert_one(user_doc)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered") from exc

    created_user = users_collection.find_one({"_id": insert_result.inserted_id})
    token = _create_access_token(str(created_user["_id"]), created_user["email"])
    return {"token": token, "user": _serialize_user(created_user)}


@app.post("/auth/login")
async def login(request: LoginRequest):
    users_collection, _, _, _ = _get_collections()
    email = request.email.strip().lower()
    user_doc = users_collection.find_one({"email": email})

    if not user_doc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if user_doc.get("provider") == "google" and not user_doc.get("password_hash"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account uses Google Sign-In. Please continue with Google.",
        )

    if not _verify_password(request.password, user_doc.get("password_hash", ""), user_doc.get("password_salt", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    users_collection.update_one({"_id": user_doc["_id"]}, {"$set": {"last_login_at": _utc_now(), "updated_at": _utc_now()}})
    token = _create_access_token(str(user_doc["_id"]), user_doc["email"])
    return {"token": token, "user": _serialize_user(user_doc)}


@app.post("/auth/google")
async def google_auth(request: GoogleAuthRequest):
    users_collection, _, _, _ = _get_collections()
    payload = _verify_google_token(request.idToken)
    email = (payload.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google account email not available")

    now = _utc_now()
    user_doc = users_collection.find_one({"email": email})
    if user_doc:
        users_collection.update_one(
            {"_id": user_doc["_id"]},
            {
                "$set": {
                    "display_name": payload.get("name") or user_doc.get("display_name") or email.split("@")[0],
                    "provider": "google",
                    "updated_at": now,
                    "last_login_at": now,
                }
            },
        )
        user_doc = users_collection.find_one({"_id": user_doc["_id"]})
    else:
        created = {
            "email": email,
            "display_name": payload.get("name") or email.split("@")[0],
            "role": request.role,
            "provider": "google",
            "created_at": now,
            "updated_at": now,
            "last_login_at": now,
        }
        inserted = users_collection.insert_one(created)
        user_doc = users_collection.find_one({"_id": inserted.inserted_id})

    token = _create_access_token(str(user_doc["_id"]), user_doc["email"])
    return {"token": token, "user": _serialize_user(user_doc)}


@app.get("/auth/me")
async def auth_me(user_doc: dict = Depends(_get_current_user)):
    return {"user": _serialize_user(user_doc)}


@app.post("/auth/logout")
async def auth_logout(_user_doc: dict = Depends(_get_current_user)):
    return {"ok": True}


@app.post("/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    # Simple placeholder endpoint for now; keeps UX flow consistent while avoiding Firebase dependency.
    return {
        "ok": True,
        "message": f"Password reset is not configured yet for {request.email}. Please contact support.",
    }


@app.post("/generate-diet-plan", tags=["Diet Generation"])
async def generate_diet_plan(request: DietRequest):
    try:
        diet_plan = _get_diet_logic().get_diet_plan(request.model_dump())
        if "error" in diet_plan:
            raise HTTPException(status_code=422, detail=diet_plan)
        return diet_plan
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc


@app.post("/diet-jobs/start")
async def start_diet_job(
    request: DietJobStartRequest,
    background_tasks: BackgroundTasks,
    user_doc: dict = Depends(_get_current_user),
):
    now = _utc_now()
    profile_signature = _build_profile_signature(request.dietRequest.model_dump())
    profile_vector = _build_profile_vector(request.dietRequest.model_dump())
    job_doc = {
        "user_id": user_doc["_id"],
        "status": "queued",
        "message": "Diet generation queued...",
        "request_payload": request.dietRequest.model_dump(),
        "profile_signature": profile_signature,
        "profile_vector": profile_vector,
        "source": "generated",
        "matched_from_job_id": None,
        "match_score": None,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }
    _, diet_jobs_collection, _, _ = _get_collections()
    inserted = diet_jobs_collection.insert_one(job_doc)
    background_tasks.add_task(_run_diet_job, inserted.inserted_id, user_doc["_id"], request.dietRequest.model_dump())

    created = diet_jobs_collection.find_one({"_id": inserted.inserted_id, "user_id": user_doc["_id"]})
    return _serialize_job(created)


@app.get("/diet-jobs/{job_id}")
async def get_diet_job(job_id: str, user_doc: dict = Depends(_get_current_user)):
    _, diet_jobs_collection, _, _ = _get_collections()
    obj_id = _to_object_id(job_id, "job id")
    job_doc = diet_jobs_collection.find_one({"_id": obj_id, "user_id": user_doc["_id"]})
    if not job_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diet job not found")
    return _serialize_job(job_doc)


@app.get("/diet-jobs/latest")
async def get_latest_diet_job(user_doc: dict = Depends(_get_current_user)):
    _, diet_jobs_collection, _, _ = _get_collections()
    job_doc = diet_jobs_collection.find_one({"user_id": user_doc["_id"]}, sort=[("created_at", DESCENDING)])
    return _serialize_job(job_doc) if job_doc else {"id": None, "status": "none"}


@app.post("/diet-logs")
async def create_diet_log(request: DietLogCreateRequest, user_doc: dict = Depends(_get_current_user)):
    _, _, diet_logs_collection, _ = _get_collections()
    now = _utc_now()
    payload = {
        "user_id": user_doc["_id"],
        "meal_type": request.mealType,
        "meal_name": request.mealName,
        "notes": request.notes or "",
        "adherence": request.adherence,
        "logged_at": request.loggedAt or now,
        "created_at": now,
    }
    inserted = diet_logs_collection.insert_one(payload)
    created = diet_logs_collection.find_one({"_id": inserted.inserted_id, "user_id": user_doc["_id"]})
    return _serialize_diet_log(created)


@app.get("/diet-logs")
async def list_diet_logs(limit: int = 30, user_doc: dict = Depends(_get_current_user)):
    _, _, diet_logs_collection, _ = _get_collections()
    capped_limit = max(1, min(limit, 200))
    docs = list(
        diet_logs_collection.find({"user_id": user_doc["_id"]})
        .sort("logged_at", DESCENDING)
        .limit(capped_limit)
    )
    return {"items": [_serialize_diet_log(doc) for doc in docs]}


@app.delete("/diet-logs/{log_id}")
async def delete_diet_log(log_id: str, user_doc: dict = Depends(_get_current_user)):
    _, _, diet_logs_collection, _ = _get_collections()
    obj_id = _to_object_id(log_id, "log id")
    deleted = diet_logs_collection.delete_one({"_id": obj_id, "user_id": user_doc["_id"]})
    if deleted.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diet log not found")
    return {"ok": True}


@app.post("/assessments")
async def create_assessment_result(request: AssessmentResultCreateRequest, user_doc: dict = Depends(_get_current_user)):
    assessment_type = request.type.strip().lower()
    if assessment_type not in ALLOWED_ASSESSMENT_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid assessment type")

    _, _, _, assessment_collection = _get_collections()
    now = _utc_now()
    payload = {
        "user_id": user_doc["_id"],
        "type": assessment_type,
        "scores": request.scores,
        "schema": request.schema,
        "disease": request.disease,
        "captured_at": request.capturedAt or now,
        "created_at": now,
    }

    inserted = assessment_collection.insert_one(payload)
    created = assessment_collection.find_one({"_id": inserted.inserted_id, "user_id": user_doc["_id"]})
    return _serialize_assessment_result(created)


@app.get("/assessments/latest")
async def get_latest_assessments(user_doc: dict = Depends(_get_current_user)):
    _, _, _, assessment_collection = _get_collections()
    result = {}
    for assessment_type in ALLOWED_ASSESSMENT_TYPES:
        doc = assessment_collection.find_one(
            {"user_id": user_doc["_id"], "type": assessment_type},
            sort=[("created_at", DESCENDING)],
        )
        result[assessment_type] = _serialize_assessment_result(doc) if doc else None
    return result
