"""Model synchronization service."""
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.model import Model
from app.services.chutes_client import get_chutes_client

logger = get_logger(__name__)


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
        return 0
    
    # Use PostgreSQL upsert (INSERT ... ON CONFLICT UPDATE)
    for model_data in unique_models:
        slug = model_data.get("slug")
        is_llm = bool(model_data.get("is_llm", True))
        stmt = pg_insert(Model).values(
            slug=slug,
            name=model_data.get("name", slug),
            tagline=model_data.get("tagline"),
            user=model_data.get("user"),
            logo=model_data.get("logo"),
            chute_id=model_data.get("chute_id"),
            instance_count=model_data.get("instance_count", 0),
            is_active=is_llm,
        ).on_conflict_do_update(
            index_elements=["slug"],
            set_={
                "name": model_data.get("name", slug),
                "tagline": model_data.get("tagline"),
                "user": model_data.get("user"),
                "logo": model_data.get("logo"),
                "chute_id": model_data.get("chute_id"),
                "instance_count": model_data.get("instance_count", 0),
                "is_active": is_llm,
            }
        )
        await db.execute(stmt)
    
    await db.commit()
    count = len(unique_models)
    logger.info("Models synced", count=count)
    return count


async def get_models(
    db: AsyncSession,
    active_only: bool = True,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Model]:
    """Get models from database with optional filtering."""
    query = select(Model)

    if active_only:
        query = query.where(Model.is_active == True)  # noqa: E712

    if search:
        query = query.where(Model.name.ilike(f"%{search}%") | Model.slug.ilike(f"%{search}%"))

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


async def get_model_by_chute_id(db: AsyncSession, chute_id: str) -> Optional[Model]:
    """Get a model by its Chutes chute_id."""
    result = await db.execute(select(Model).where(Model.chute_id == chute_id))
    return result.scalar_one_or_none()


async def resolve_model_identifier(db: AsyncSession, identifier: str) -> Optional[Model]:
    """Resolve a model by internal UUID, slug/name, or Chutes chute_id."""
    if _is_valid_uuid(identifier):
        model = await get_model_by_id(db, identifier)
        if model:
            return model
        model = await get_model_by_chute_id(db, identifier)
        if model:
            return model
    model = await get_model_by_slug(db, identifier)
    if model:
        return model
    return await get_model_by_chute_id(db, identifier)
