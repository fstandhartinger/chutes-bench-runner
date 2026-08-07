"""Model synchronization service."""
from typing import Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.model import Model
from app.core.config import get_settings
from app.services.chutes_client import get_chutes_client
from app.services.openrouter_client import get_openrouter_client

logger = get_logger(__name__)

GREMIUM_MODELS = (
    {
        "slug": "gremium-consensus",
        "name": "Gremium (consensus)",
        "provider": "gremium-openai",
    },
    {
        "slug": "gremium-consensus-anthropic",
        "name": "Gremium (consensus • Anthropic)",
        "provider": "gremium-anthropic",
    },
)

RLM_MODELS = (
    {
        "slug": "rlm-gpt-4o",
        "name": "RLM (GPT-4o)",
        "provider": "rlm",
    },
    {
        "slug": "rlm-claude-3-5-sonnet",
        "name": "RLM (Claude 3.5 Sonnet)",
        "provider": "rlm",
    },
)


def _is_valid_uuid(value: Optional[str]) -> bool:
    if not value:
        return False
    try:
        UUID(value)
    except ValueError:
        return False
    return True


async def sync_models(db: AsyncSession) -> int:
    """
    Sync models from Chutes API to local database using upsert.
    
    Returns:
        Number of models updated/created
    """
    client = get_chutes_client()
    models = await client.list_models()
    llm_identifiers: set[str] = set()
    try:
        llm_identifiers = await client.get_llm_identifiers()
    except Exception as exc:
        logger.warning("Failed to fetch LLM model list", error=str(exc))
    
    # Filter out models without slugs and deduplicate by slug
    seen_slugs: set[str] = set()
    unique_models: list[dict] = []
    for model_data in models:
        slug = model_data.get("slug")
        if slug and slug not in seen_slugs:
            seen_slugs.add(slug)
            unique_models.append(model_data)
    
    if not unique_models:
        logger.warning("No models to sync")

    # Use PostgreSQL upsert (INSERT ... ON CONFLICT UPDATE)
    for model_data in unique_models:
        slug = model_data.get("slug")
        chute_id = model_data.get("chute_id")
        is_llm_flag = bool(model_data.get("is_llm", True))
        if llm_identifiers:
            is_llm = slug in llm_identifiers or (chute_id in llm_identifiers if chute_id else False)
        else:
            is_llm = is_llm_flag
        stmt = pg_insert(Model).values(
            slug=slug,
            name=model_data.get("name", slug),
            tagline=model_data.get("tagline"),
            user=model_data.get("user"),
            logo=model_data.get("logo"),
            chute_id=chute_id,
            instance_count=model_data.get("instance_count", 0),
            is_active=is_llm,
            provider="chutes",
        ).on_conflict_do_update(
            index_elements=["slug"],
            set_={
                "name": model_data.get("name", slug),
                "tagline": model_data.get("tagline"),
                "user": model_data.get("user"),
                "logo": model_data.get("logo"),
                "chute_id": chute_id,
                "instance_count": model_data.get("instance_count", 0),
                "is_active": is_llm,
                "provider": "chutes",
            }
        )
        await db.execute(stmt)

    settings = get_settings()
    if settings.enable_gremium_provider:
        await ensure_gremium_models(db)
    if settings.enable_rlm_provider:
        await ensure_rlm_models(db)
    openrouter_count = 0
    if settings.openrouter_api_key:
        try:
            openrouter_count = await ensure_openrouter_models(db)
        except Exception as exc:
            # OpenRouter is an optional alternate provider. A catalog outage
            # must not take the existing Chutes model sync down with it.
            logger.warning("Failed to sync OpenRouter models", error=str(exc))

    await db.commit()
    count = len(unique_models) + openrouter_count
    logger.info(
        "Models synced",
        count=count,
        chutes_count=len(unique_models),
        openrouter_count=openrouter_count,
    )
    return count


async def get_models(
    db: AsyncSession,
    active_only: bool = True,
    search: Optional[str] = None,
    provider: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Model]:
    """Get models from database with optional filtering."""
    query = select(Model)

    if active_only:
        query = query.where(Model.is_active == True)  # noqa: E712

    if search:
        query = query.where(Model.name.ilike(f"%{search}%") | Model.slug.ilike(f"%{search}%"))

    if provider:
        query = query.where(Model.provider == provider)

    query = query.order_by(Model.instance_count.desc(), Model.name).offset(offset).limit(limit)

    result = await db.execute(query)
    return list(result.scalars().all())


async def get_model_by_id(db: AsyncSession, model_id: str) -> Optional[Model]:
    """Get a model by ID."""
    if not _is_valid_uuid(model_id):
        return None
    result = await db.execute(select(Model).where(Model.id == model_id))
    return result.scalar_one_or_none()


async def get_model_by_slug(db: AsyncSession, slug: str) -> Optional[Model]:
    """Get a model by slug."""
    result = await db.execute(select(Model).where(Model.slug == slug))
    return result.scalar_one_or_none()


async def get_model_by_name(db: AsyncSession, name: str) -> Optional[Model]:
    """Get a model by display name (case-insensitive)."""
    lowered = name.strip().lower()
    result = await db.execute(select(Model).where(func.lower(Model.name) == lowered))
    return result.scalar_one_or_none()


async def get_model_by_chute_id(db: AsyncSession, chute_id: str) -> Optional[Model]:
    """Get a model by its Chutes chute_id."""
    result = await db.execute(select(Model).where(Model.chute_id == chute_id))
    return result.scalar_one_or_none()


async def resolve_model_identifier(
    db: AsyncSession,
    identifier: str,
    provider: Optional[str] = None,
) -> Optional[Model]:
    """Resolve a model by internal UUID, slug/name, or Chutes chute_id."""
    if _is_valid_uuid(identifier):
        model = await get_model_by_id(db, identifier)
        if model and (provider is None or model.provider == provider):
            return model
        model = await get_model_by_chute_id(db, identifier)
        if model and (provider is None or model.provider == provider):
            return model
    model = await get_model_by_slug(db, identifier)
    if model and (provider is None or model.provider == provider):
        return model
    model = await get_model_by_name(db, identifier)
    if model and (provider is None or model.provider == provider):
        return model
    model = await get_model_by_chute_id(db, identifier)
    if model and (provider is None or model.provider == provider):
        return model
    return None


async def ensure_gremium_models(db: AsyncSession) -> None:
    """Ensure synthetic Gremium model entries exist in the database."""
    for entry in GREMIUM_MODELS:
        stmt = pg_insert(Model).values(
            slug=entry["slug"],
            name=entry["name"],
            tagline="Gremium consensus routing",
            instance_count=1,
            is_active=True,
            provider=entry["provider"],
        ).on_conflict_do_update(
            index_elements=["slug"],
            set_={
                "name": entry["name"],
                "instance_count": 1,
                "is_active": True,
                "provider": entry["provider"],
            },
        )
        await db.execute(stmt)


async def ensure_rlm_models(db: AsyncSession) -> None:
    """Ensure synthetic RLM model entries exist in the database."""
    for entry in RLM_MODELS:
        stmt = pg_insert(Model).values(
            slug=entry["slug"],
            name=entry["name"],
            tagline="RLM long-context handling via recursive slicing",
            instance_count=1,
            is_active=True,
            provider=entry["provider"],
        ).on_conflict_do_update(
            index_elements=["slug"],
            set_={
                "name": entry["name"],
                "instance_count": 1,
                "is_active": True,
                "provider": entry["provider"],
            },
        )
        await db.execute(stmt)


async def ensure_openrouter_models(db: AsyncSession) -> int:
    """Upsert the configured OpenRouter benchmark target from its live catalog."""
    models = await get_openrouter_client().list_models()
    for entry in models:
        stmt = pg_insert(Model).values(
            slug=entry["slug"],
            name=entry["name"],
            tagline=entry.get("tagline"),
            user=entry.get("user"),
            logo=entry.get("logo"),
            chute_id=None,
            instance_count=entry.get("instance_count", 1),
            is_active=True,
            provider="openrouter",
        ).on_conflict_do_update(
            index_elements=["slug"],
            set_={
                "name": entry["name"],
                "tagline": entry.get("tagline"),
                "user": entry.get("user"),
                "logo": entry.get("logo"),
                "chute_id": None,
                "instance_count": entry.get("instance_count", 1),
                "is_active": True,
                "provider": "openrouter",
            },
        )
        await db.execute(stmt)
    return len(models)
