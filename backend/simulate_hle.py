import asyncio
import os
from app.services.chutes_client import get_chutes_client
from app.benchmarks.adapters.hle import HLEAdapter

async def test_hle_single():
    # Use a real model slug from the list
    model_slug = "Qwen/Qwen3-32B"
    client = get_chutes_client()
    
    adapter = HLEAdapter(client, model_slug)
    await adapter.preload()
    
    print(f"Total items: {await adapter.get_total_items()}")
    
    # Evaluate the first item
    item_id = "0"
    print(f"Evaluating item {item_id}...")
    result = await adapter.evaluate_item(item_id)
    
    print("\nResult:")
    print(f"ID: {result.item_id}")
    print(f"Correct: {result.is_correct}")
    print(f"Score: {result.score}")
    print(f"Error: {result.error}")
    print(f"Response: {result.response[:100]}..." if result.response else "Response: None")

if __name__ == "__main__":
    asyncio.run(test_hle_single())

