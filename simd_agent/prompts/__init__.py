# simd_agent/prompts/__init__.py
"""Prompt management for simd_agent."""

from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).parent / "packs"


def load_prompt(pack: str, name: str) -> str:
    """Load a prompt file from a pack.
    
    Args:
        pack: The prompt pack name (e.g., 'simd')
        name: The prompt file name without extension (e.g., 'system')
        
    Returns:
        The prompt content
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    path = PROMPTS_DIR / pack / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text()


def load_pack(pack: str) -> dict[str, str]:
    """Load all prompts from a pack.
    
    Args:
        pack: The prompt pack name
        
    Returns:
        Dictionary mapping prompt names to content
    """
    pack_dir = PROMPTS_DIR / pack
    if not pack_dir.exists():
        raise FileNotFoundError(f"Pack not found: {pack}")
    
    prompts = {}
    for path in pack_dir.glob("*.md"):
        name = path.stem
        prompts[name] = path.read_text()
    
    return prompts


def list_packs() -> list[str]:
    """List available prompt packs.
    
    Returns:
        List of pack names
    """
    return [d.name for d in PROMPTS_DIR.iterdir() if d.is_dir()]
