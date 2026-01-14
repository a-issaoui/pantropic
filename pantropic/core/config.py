"""Pantropic - Simple Configuration.

Minimal config for a local LLM server like Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from pantropic.observability.logging import get_logger

log = get_logger("config")


@dataclass
class Config:
    """Simple Pantropic configuration."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Models
    models_dir: Path = field(default_factory=lambda: Path("models"))
    default_context: int = 8192
    max_context: int = 131072
    preload_model: str | None = None
    auto_unload_timeout: int = 300  # 5 minutes

    # Hardware
    flash_attention: bool = True
    use_mmap: bool = True
    cpu_threads: int = 0  # 0 = auto

    # Intelligent Context Management (Agent Mode)
    agent_mode: bool = True  # Enable smart context sizing for agents
    gpu_priority: str = "max"  # "max" | "balanced" | "efficient"
    context_expansion_threshold: float = 0.8  # Expand when usage > 80%
    min_gpu_layers_percent: int = 70  # Minimum 70% layers on GPU before offload

    # Sessions
    max_sessions: int = 100
    session_timeout: int = 3600  # 1 hour

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        """Load config from YAML file."""
        # Search paths
        search = [
            Path("config.yaml"),
            Path("pantropic.yaml"),
            Path("config/default.yaml"),
        ]

        if path:
            search.insert(0, Path(path))

        # Find config file
        config_file = None
        for p in search:
            if p.exists():
                config_file = p
                break

        if config_file:
            log.info(f"Loading config from {config_file}")
            with open(config_file) as f:
                data = yaml.safe_load(f) or {}
            return cls._from_dict(data)

        log.info("No config file found, using defaults")
        return cls()

    @classmethod
    def _from_dict(cls, data: dict) -> Config:
        """Create Config from dictionary."""
        config = cls()

        # Simple flat mapping
        if "host" in data:
            config.host = str(data["host"])
        if "port" in data:
            config.port = int(data["port"])
        if "log_level" in data:
            config.log_level = str(data["log_level"])

        if "models_dir" in data:
            config.models_dir = Path(data["models_dir"])
        if "default_context" in data:
            config.default_context = int(data["default_context"])
        if "max_context" in data:
            config.max_context = int(data["max_context"])
        if "preload_model" in data:
            config.preload_model = str(data["preload_model"]) if data["preload_model"] else None
        if "auto_unload_timeout" in data:
            config.auto_unload_timeout = int(data["auto_unload_timeout"])

        if "flash_attention" in data:
            config.flash_attention = bool(data["flash_attention"])
        if "use_mmap" in data:
            config.use_mmap = bool(data["use_mmap"])
        if "cpu_threads" in data:
            config.cpu_threads = int(data["cpu_threads"])

        if "max_sessions" in data:
            config.max_sessions = int(data["max_sessions"])
        if "session_timeout" in data:
            config.session_timeout = int(data["session_timeout"])

        return config

    def validate(self) -> list[str]:
        """Validate config. Returns list of warnings."""
        warnings = []

        if self.port < 1 or self.port > 65535:
            warnings.append(f"Invalid port: {self.port}")

        if self.default_context > self.max_context:
            warnings.append("default_context > max_context")

        if not self.models_dir.exists():
            warnings.append(f"Models directory not found: {self.models_dir}")

        return warnings

    # Compatibility properties for old code
    @property
    def server(self):
        """Compatibility: server config."""
        return type("ServerConfig", (), {
            "host": self.host,
            "port": self.port,
            "log_level": self.log_level,
            "access_log": True,
        })()

    @property
    def models(self):
        """Compatibility: models config."""
        return type("ModelConfig", (), {
            "directory": self.models_dir,
            "default_context": self.default_context,
            "max_context": self.max_context,
            "preload_model": self.preload_model,
            "auto_unload_timeout": self.auto_unload_timeout,
        })()

    @property
    def inference(self):
        """Compatibility: inference config."""
        return type("InferenceConfig", (), {
            "flash_attention": self.flash_attention,
            "mmap": self.use_mmap,
            "cpu_threads": self.cpu_threads,
        })()

    @property
    def api(self):
        """Compatibility: API config."""
        return type("APIConfig", (), {
            "cors_origins": ["*"],
            "require_auth": False,
            "api_key": "",
        })()

    @property
    def sessions(self):
        """Compatibility: sessions config."""
        return type("SessionConfig", (), {
            "max_sessions": self.max_sessions,
            "timeout_seconds": self.session_timeout,
        })()
