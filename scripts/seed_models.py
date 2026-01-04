#!/usr/bin/env python3
"""Seed models from Chutes API."""
import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv

load_dotenv()


async def main():
    from app.db.session import async_session_maker
    from app.services.model_service import sync_models

    print("🔄 Syncing models from Chutes API...")
    
    async with async_session_maker() as db:
        count = await sync_models(db)
        print(f"✅ Synced {count} models")


if __name__ == "__main__":
    asyncio.run(main())














