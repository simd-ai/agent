# simd_agent/run/apply_recommendation.py
"""Apply convergence recommendations to generated OpenFOAM files.

Modifies system/fvSolution (relaxation factors) and system/controlDict
(time step, maxCo) based on structured recommendation actions.
"""

import logging
import re

logger = logging.getLogger(__name__)


def apply_recommendation(
    files: dict[str, str],
    action: dict,
) -> dict[str, str]:
    """Apply a recommendation action to generated files.

    Args:
        files: Dict of path → content (generated OpenFOAM files).
        action: Action dict with 'type' and 'changes' keys.

    Returns:
        Modified files dict (only changed files differ from input).
    """
    action_type = action.get("type")
    changes = action.get("changes", {})

    if not changes:
        return files

    result = dict(files)  # shallow copy

    if action_type == "relaxation":
        result = _apply_relaxation(result, changes)
    elif action_type == "time_step":
        result = _apply_time_step(result, changes)
    elif action_type == "more_iterations":
        result = _apply_more_iterations(result, changes)

    return result


# ── Relaxation factor modification ────────────────────────────────────────

def _apply_relaxation(
    files: dict[str, str],
    changes: dict[str, float],
) -> dict[str, str]:
    """Modify relaxation factors in system/fvSolution."""
    fv_key = _find_file(files, "system/fvSolution")
    if not fv_key:
        logger.warning("[APPLY] system/fvSolution not found in generated files")
        return files

    content = files[fv_key]
    original = content

    # Separate changes into fields vs equations.
    # In OpenFOAM fvSolution, relaxationFactors has two sub-dicts:
    #   fields { p 0.3; rho 0.1; }
    #   equations { U 0.7; k 0.7; ... }
    field_vars = {"p", "p_rgh", "rho"}
    field_changes = {k: v for k, v in changes.items() if k in field_vars}
    eq_changes = {k: v for k, v in changes.items() if k not in field_vars}

    # Apply to fields block
    if field_changes:
        content = _patch_relaxation_block(content, "fields", field_changes)

    # Apply to equations block
    if eq_changes:
        content = _patch_relaxation_block(content, "equations", eq_changes)

    if content != original:
        files = dict(files)
        files[fv_key] = content
        changed = {**field_changes, **eq_changes}
        logger.info(f"[APPLY] Modified relaxation factors: {changed}")

    return files


def _patch_relaxation_block(
    content: str,
    block_name: str,  # "fields" or "equations"
    changes: dict[str, float],
) -> str:
    """Patch individual values inside a relaxationFactors sub-block.

    Handles the OpenFOAM dict format:
        relaxationFactors
        {
            fields
            {
                p       0.3;
            }
            equations
            {
                U       0.7;
            }
        }

    Only modifies values INSIDE the relaxationFactors block to avoid
    accidentally changing residualControl thresholds or solver tolerances.
    """
    # Find the relaxationFactors block
    relax_match = re.search(
        r'(relaxationFactors\s*\{)',
        content,
    )
    if not relax_match:
        logger.debug("[APPLY] relaxationFactors block not found")
        return content

    # Find the matching closing brace for relaxationFactors
    start = relax_match.start()
    brace_depth = 0
    end = start
    for i in range(relax_match.end() - 1, len(content)):
        if content[i] == '{':
            brace_depth += 1
        elif content[i] == '}':
            brace_depth -= 1
            if brace_depth == 0:
                end = i + 1
                break

    # Extract the relaxationFactors block, modify it, put it back
    block = content[start:end]

    for var_name, new_value in changes.items():
        pattern = re.compile(
            rf'(\b{re.escape(var_name)}\s+)'
            rf'[\d.eE+-]+'
            rf'(\s*;)',
        )
        new_block = pattern.sub(rf'\g<1>{new_value}\2', block)
        if new_block != block:
            block = new_block
            logger.debug(f"[APPLY] relaxation {var_name}: → {new_value}")
        else:
            logger.debug(f"[APPLY] {var_name} not found in relaxationFactors, skipping")

    return content[:start] + block + content[end:]


# ── Time step modification ────────────────────────────────────────────────

def _apply_time_step(
    files: dict[str, str],
    changes: dict[str, float],
) -> dict[str, str]:
    """Modify time step settings in system/controlDict."""
    cd_key = _find_file(files, "system/controlDict")
    if not cd_key:
        logger.warning("[APPLY] system/controlDict not found in generated files")
        return files

    content = files[cd_key]
    original = content

    # Apply maxCo
    if "maxCo" in changes:
        content = _patch_numeric_value(content, "maxCo", changes["maxCo"])

    # Apply deltaT factor (multiply current deltaT by the factor)
    if "deltaT_factor" in changes:
        factor = changes["deltaT_factor"]
        match = re.search(r'deltaT\s+([\d.eE+-]+)\s*;', content)
        if match:
            current_dt = float(match.group(1))
            new_dt = current_dt * factor
            content = _patch_numeric_value(content, "deltaT", new_dt)
            logger.debug(f"[APPLY] deltaT: {current_dt} × {factor} → {new_dt}")

    # Apply maxAlphaCo if present
    if "maxAlphaCo" in changes:
        content = _patch_numeric_value(content, "maxAlphaCo", changes["maxAlphaCo"])

    if content != original:
        files = dict(files)
        files[cd_key] = content
        logger.info(f"[APPLY] Modified controlDict: {changes}")

    return files


# ── More iterations ───────────────────────────────────────────────────────

def _apply_more_iterations(
    files: dict[str, str],
    changes: dict[str, float],
) -> dict[str, str]:
    """Extend simulation by multiplying endTime or maxIterations."""
    cd_key = _find_file(files, "system/controlDict")
    if not cd_key:
        return files

    content = files[cd_key]
    original = content
    multiplier = changes.get("iteration_multiplier", 2)

    # Try endTime first (transient)
    match = re.search(r'endTime\s+([\d.eE+-]+)\s*;', content)
    if match:
        current = float(match.group(1))
        new_val = current * multiplier
        content = _patch_numeric_value(content, "endTime", new_val)
        logger.debug(f"[APPLY] endTime: {current} × {multiplier} → {new_val}")

    # Also try writeInterval
    match = re.search(r'writeInterval\s+([\d.eE+-]+)\s*;', content)
    if match:
        # Keep writeInterval proportional
        current_wi = float(match.group(1))
        new_wi = current_wi * multiplier
        content = _patch_numeric_value(content, "writeInterval", new_wi)

    if content != original:
        files = dict(files)
        files[cd_key] = content
        logger.info(f"[APPLY] Extended simulation: multiplier={multiplier}")

    return files


# ── Utilities ─────────────────────────────────────────────────────────────

def _find_file(files: dict[str, str], suffix: str) -> str | None:
    """Find a file key that ends with the given suffix."""
    for key in files:
        if key.endswith(suffix) or key == suffix:
            return key
    return None


def _patch_numeric_value(content: str, key: str, value: float) -> str:
    """Replace a numeric value for a given key in OpenFOAM dict format."""
    # Format value: use scientific notation for very small/large, else plain
    if abs(value) < 1e-3 or abs(value) > 1e6:
        formatted = f"{value:.2e}"
    elif value == int(value):
        formatted = str(int(value))
    else:
        formatted = f"{value:g}"

    pattern = re.compile(
        rf'(\b{re.escape(key)}\s+)'
        rf'[\d.eE+-]+'
        rf'(\s*;)',
    )
    return pattern.sub(rf'\g<1>{formatted}\2', content)
