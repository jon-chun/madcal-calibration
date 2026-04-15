"""
Model configuration for AgenticSimLaw debate experiments.

Manages model lists from Ollama YAML configs and commercial model definitions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass
class ModelSpec:
    """Specification for a single LLM model."""
    model_id: str
    display_name: str
    provider: str  # "ollama", "openai", "google", "xai", "openrouter"
    max_tokens: int = 1024
    temperature: float = 0.7
    reasoning_effort: str = ""  # For thinking models: "minimal", "low", "medium", "high"


# Commercial models (hard-coded fallback)
COMMERCIAL_MODELS: List[ModelSpec] = [
    ModelSpec("gpt-4o-mini", "GPT-4o Mini", "openai"),
    ModelSpec("gpt-5-mini", "GPT-5 Mini", "openai", reasoning_effort="minimal"),
    ModelSpec("gemini/gemini-2.5-flash", "Gemini 2.5 Flash", "google"),
    ModelSpec("xai/grok-4.1-fast", "Grok 4.1 Fast", "xai"),
]


def load_ollama_models_from_yaml(config_name: str) -> List[ModelSpec]:
    """Load model list from configs/config_ollama_models_{config_name}.yaml."""
    yaml_path = CONFIGS_DIR / f"config_ollama_models_{config_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    model_ids = data.get("ollama_pull_models", [])
    return [
        ModelSpec(
            model_id=mid,
            display_name=mid,
            provider="ollama",
        )
        for mid in model_ids
    ]


def load_commercial_models_from_yaml() -> List[ModelSpec]:
    """Load commercial model list from configs/config_commercial_models.yaml."""
    yaml_path = CONFIGS_DIR / "config_commercial_models.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    models = data.get("commercial_models", [])
    return [
        ModelSpec(
            model_id=m["model_id"],
            display_name=m.get("display_name", m["model_id"]),
            provider=m.get("provider", "unknown"),
            max_tokens=m.get("max_tokens", 1024),
            reasoning_effort=m.get("reasoning_effort", ""),
        )
        for m in models
    ]


def load_remote_oss_models_from_yaml() -> List[ModelSpec]:
    """Load remote open-source models from configs/config_commercial_models.yaml."""
    yaml_path = CONFIGS_DIR / "config_commercial_models.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    models = data.get("remote_oss_models", [])
    return [
        ModelSpec(
            model_id=m["model_id"],
            display_name=m.get("display_name", m["model_id"]),
            provider=m.get("provider", "openrouter"),
            max_tokens=m.get("max_tokens", 1024),
        )
        for m in models
    ]


def _load_yaml_section_from(
    filename: str, section: str, default_provider: str = "openrouter"
) -> List[ModelSpec]:
    """Load a named section from an arbitrary YAML config file."""
    yaml_path = CONFIGS_DIR / filename
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    models = data.get(section, [])
    return [
        ModelSpec(
            model_id=m["model_id"],
            display_name=m.get("display_name", m["model_id"]),
            provider=m.get("provider", default_provider),
            max_tokens=m.get("max_tokens", 1024),
        )
        for m in models
    ]


def _load_yaml_section(section: str, default_provider: str = "openrouter") -> List[ModelSpec]:
    """Load a named section from config_commercial_models.yaml."""
    yaml_path = CONFIGS_DIR / "config_commercial_models.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    models = data.get(section, [])
    return [
        ModelSpec(
            model_id=m["model_id"],
            display_name=m.get("display_name", m["model_id"]),
            provider=m.get("provider", default_provider),
            max_tokens=m.get("max_tokens", 1024),
        )
        for m in models
    ]


def get_model_list(ensemble: str) -> List[ModelSpec]:
    """Return models for a named ensemble or comma-separated model IDs.

    Named ensembles:
      "commercial"    — 4 commercial models (GPT, Gemini, Grok)
      "remote-oss"    — 10 cloud-hosted OSS models via OpenRouter
      "fireworks-oss" — 5 OSS models available on Fireworks.ai
      "openrouter-oss"— 5 OSS models only on OpenRouter
      "split-oss"     — all 10 split across Fireworks + OpenRouter
      "all-remote"    — commercial + remote-oss combined
      "signtest-oss"  — 4 OSS models for sign test expansion (NLSY97)
      "size", "oss", "reasoning", "all" — Ollama YAML ensembles
    Otherwise treats input as comma-separated model IDs.
    """
    if ensemble == "commercial":
        return load_commercial_models_from_yaml()

    if ensemble == "remote-oss":
        return load_remote_oss_models_from_yaml()

    if ensemble == "fireworks-oss":
        return _load_yaml_section("fireworks_oss_models", "fireworks")

    if ensemble == "openrouter-oss":
        return _load_yaml_section("openrouter_only_oss_models", "openrouter")

    if ensemble == "split-oss":
        return (_load_yaml_section("fireworks_oss_models", "fireworks")
                + _load_yaml_section("openrouter_only_oss_models", "openrouter"))

    if ensemble == "all-remote":
        return load_commercial_models_from_yaml() + load_remote_oss_models_from_yaml()

    if ensemble == "signtest-oss":
        return _load_yaml_section_from(
            "config_signtest_oss.yaml", "signtest_oss_models", "openrouter"
        )

    # Ollama YAML ensembles
    ollama_ensembles = {"size", "oss", "reasoning", "all"}
    if ensemble in ollama_ensembles:
        return load_ollama_models_from_yaml(ensemble)

    # Comma-separated model IDs
    model_ids = [m.strip() for m in ensemble.split(",") if m.strip()]
    return [
        ModelSpec(
            model_id=mid,
            display_name=mid,
            provider=_infer_provider(mid),
        )
        for mid in model_ids
    ]


def _infer_provider(model_id: str) -> str:
    """Infer provider from a model ID string.

    Ollama models contain ':' but never '/'.
    Cloud-routed models always contain '/' (e.g. openrouter/, fireworks_ai/, gemini/).
    """
    if "/" in model_id:
        if model_id.startswith("openrouter/"):
            return "openrouter"
        if model_id.startswith("fireworks_ai/"):
            return "fireworks"
        if model_id.startswith("gemini/"):
            return "google"
        if model_id.startswith("xai/"):
            return "xai"
        return "commercial"
    if ":" in model_id:
        return "ollama"
    # Bare names like gpt-4o-mini
    return "commercial"
