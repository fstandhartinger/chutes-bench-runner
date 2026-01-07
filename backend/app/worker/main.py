"""Worker entry point with health check server."""
import asyncio
import os
from aiohttp import web

from app.worker.runner import run_worker


async def health_handler(request):
    """Simple health check endpoint."""
    return web.Response(text="OK", status=200)


async def run_health_server():
    """Run a minimal HTTP server for Render health checks."""
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/health", health_handler)
    
    port = int(os.environ.get("PORT", 10000))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"Health check server running on port {port}")
    
    # Keep running forever
    while True:
        await asyncio.sleep(3600)


async def main():
    """Run both the worker and health check server."""
    disabled = os.environ.get("WORKER_DISABLED", "").strip().lower() in {"1", "true", "yes"}
    if disabled:
        print("Worker disabled via WORKER_DISABLED; running health server only")
        await run_health_server()
        return

    # Run both tasks concurrently
    await asyncio.gather(
        run_health_server(),
        run_worker(),
    )


if __name__ == "__main__":
    asyncio.run(main())
