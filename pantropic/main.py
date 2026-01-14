"""Pantropic - Main Entry Point.

Run the Pantropic LLM server.

Usage:
    python -m pantropic.main           # Normal startup (uses models.json)
    python -m pantropic.main --scan    # Rescan models before starting
    python -m pantropic.main --scan-only  # Only rescan, don't start server
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import uvicorn

from pantropic.core.config import Config
from pantropic.core.container import Container
from pantropic.observability.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pantropic - Intelligent Local LLM Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pantropic.main              # Start server (fast, uses models.json)
  python -m pantropic.main --scan       # Rescan models then start
  python -m pantropic.main --scan-only  # Only rescan models.json
        """,
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Rescan models directory and update models.json before starting",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only rescan models.json, don't start the server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port from config",
    )
    return parser.parse_args()


async def scan_models(config: Config, log) -> None:
    """Scan models directory and update models.json."""
    from pantropic.model_manager.scanner import PerfectGGUFScanner

    models_dir = config.models_dir
    output_file = str(models_dir.parent / "models.json")

    log.info(f"Scanning models in {models_dir}...")

    scanner = PerfectGGUFScanner()
    scanner.scan_directory(str(models_dir), output_file=output_file)

    log.info("Scan complete. models.json updated.")


async def main() -> None:
    """Main entry point for Pantropic."""
    args = parse_args()

    # Load configuration
    config = Config.load()

    # Override port if specified
    if args.port:
        config.port = args.port

    # Setup logging
    setup_logging(
        level=config.server.log_level,
        json_output=False,
    )
    log = get_logger("main")

    log.info("=" * 60)
    log.info("Pantropic v1.0.0 - Intelligent Local LLM Server")
    log.info("=" * 60)

    # Rescan models if requested
    if args.scan or args.scan_only:
        await scan_models(config, log)
        if args.scan_only:
            log.info("Scan complete. Exiting.")
            return

    # Validate config
    warnings = config.validate()
    for warning in warnings:
        log.warning(f"Config: {warning}")

    # Initialize container
    async with await Container.create(config) as container:
        # Start background tasks
        container.start_background_tasks()

        # Log discovered models
        models = container.model_registry.list_models()
        log.info(f"Ready with {len(models)} models:")
        for model in models:
            caps = ", ".join(model.capabilities.as_list())
            log.info(f"  - {model.id}: {model.specs.architecture} {model.specs.parameters_b}B [{caps}]")

        # Preload model if configured
        if config.models.preload_model:
            preload_id = config.models.preload_model
            log.info(f"Preloading model: {preload_id}")
            try:
                preload_model = container.model_registry.get_model(preload_id)
                model_loader = container.inference_engine.loader
                await model_loader.load(preload_model)
                log.info(f"Preloaded {preload_id}")
            except Exception as e:
                log.warning(f"Failed to preload {preload_id}: {e}")

        # Start auto-unload worker
        async def start_auto_unload() -> None:
            inf_loader = container.inference_engine.loader
            await inf_loader.auto_unload_worker(
                idle_timeout=float(config.models.auto_unload_timeout),
            )

        asyncio.create_task(start_auto_unload())

        # Import and run FastAPI app
        from pantropic.api.app import create_app

        app = create_app(container)

        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level=config.server.log_level.lower(),
            access_log=True,
        )

        server = uvicorn.Server(uvicorn_config)

        log.info(f"Starting server on http://{config.server.host}:{config.server.port}")

        await server.serve()


def run() -> None:
    """Console entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
