from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
from pathlib import Path
from app.urdu_mappings import URDU_TO_ENGLISH, ENGLISH_TO_URDU_DISPLAY


router = APIRouter()


# Pydantic models for request/response
class Animation(BaseModel):
    id: str
    name: str
    description: str
    file_path: str
    category: str
    tags: List[str]


class ResolveAnimationRequest(BaseModel):
    phrase: str
    language: str = "psl"


class ResolveAnimationResponse(BaseModel):
    animation: Animation
    matched_word: str
    confidence: float


# Load animations from config file
def load_animations() -> List[Animation]:
    """Load animation metadata from JSON config file."""
    config_path = Path(__file__).parent.parent / "animations_config.json"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Animation(**item) for item in data]
    except FileNotFoundError:
        raise RuntimeError(f"Animation config file not found at {config_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading animations config: {e}")


# In-memory cache of animations (loaded once at startup)
ANIMATIONS_CACHE: Optional[List[Animation]] = None


def get_animations() -> List[Animation]:
    """Get all animations with caching."""
    global ANIMATIONS_CACHE

    if ANIMATIONS_CACHE is None:
        ANIMATIONS_CACHE = load_animations()

    return ANIMATIONS_CACHE


@router.get("/animations", response_model=List[Animation])
async def list_animations():
    """
    Get list of all available animations with metadata.

    Returns:
        List of animation objects with id, name, description, file_path, category, and tags
    """
    return get_animations()


@router.get("/animations/{animation_id}", response_model=Animation)
async def get_animation_by_id(animation_id: str):
    """
    Get a single animation by its ID.

    Args:
        animation_id: The unique identifier for the animation (e.g., "hello", "thanks")

    Returns:
        Animation object with full metadata

    Raises:
        404: If animation not found
    """
    animations = get_animations()

    # Find animation by ID
    animation = next((a for a in animations if a.id == animation_id), None)

    if not animation:
        raise HTTPException(status_code=404, detail=f"Animation '{animation_id}' not found")

    return animation


@router.post("/resolve-animation", response_model=ResolveAnimationResponse)
async def resolve_animation(request: ResolveAnimationRequest):
    """
    Resolve a phrase to the best matching animation.

    This endpoint takes a text phrase and returns the best animation to represent it.
    Supports both English and Urdu input through word mapping.

    Args:
        request: Object containing phrase and language

    Returns:
        The best matching animation with confidence score

    Raises:
        404: If no animation found for the phrase
    """
    animations = get_animations()
    phrase_lower = request.phrase.lower().strip()
    original_phrase = phrase_lower

    # If Urdu language, try to translate to English first
    if request.language == "ur":
        # Try exact phrase match first
        if phrase_lower in URDU_TO_ENGLISH:
            translated = URDU_TO_ENGLISH[phrase_lower]
            animation = next((a for a in animations if a.id == translated), None)
            if animation:
                return ResolveAnimationResponse(
                    animation=animation,
                    matched_word=phrase_lower,
                    confidence=1.0  # Exact Urdu match
                )

        # Try word-by-word translation
        words = phrase_lower.split()
        translated_words = []
        for word in words:
            if word in URDU_TO_ENGLISH:
                translated_words.append(URDU_TO_ENGLISH[word])
            else:
                translated_words.append(word)

        # Use translated phrase for matching
        if translated_words:
            phrase_lower = " ".join(translated_words)

    # Simple strategy: try to match first word in phrase to animation ID
    # Split phrase into words
    words = phrase_lower.split()

    if not words:
        raise HTTPException(status_code=400, detail="Phrase cannot be empty")

    # Try to find exact match for first word
    first_word = words[0]
    animation = next((a for a in animations if a.id == first_word), None)

    if animation:
        return ResolveAnimationResponse(
            animation=animation,
            matched_word=first_word,
            confidence=1.0  # Exact match
        )

    # Try fuzzy matching - check if any animation ID is contained in phrase
    for word in words:
        animation = next((a for a in animations if a.id == word), None)
        if animation:
            return ResolveAnimationResponse(
                animation=animation,
                matched_word=word,
                confidence=0.8  # Word found but not first
            )

    # Try matching tags
    for animation in animations:
        for tag in animation.tags:
            if tag.lower() in phrase_lower:
                return ResolveAnimationResponse(
                    animation=animation,
                    matched_word=tag,
                    confidence=0.6  # Tag match
                )

    # No match found
    if request.language == "ur":
        raise HTTPException(
            status_code=404,
            detail=f"اس جملے کے لیے کوئی اینیمیشن نہیں ملا '{original_phrase}'"
        )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No animation found for phrase '{request.phrase}'. Available animations: {', '.join([a.id for a in animations[:10]])}..."
        )


@router.get("/urdu-words")
async def get_urdu_words():
    """
    Get Urdu display names for all available animations.

    Returns:
        Dictionary mapping English animation IDs to their Urdu names
    """
    return ENGLISH_TO_URDU_DISPLAY
