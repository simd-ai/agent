"""
OpenFOAM simulation runner with mesh conversion support.

COPY THIS TO YOUR SIMULATION SERVER's app/runner.py
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from app.config import RUNS_DIR, build_shell_command
from app.models import RunMode, RunStatus
from app.store import emit_event, get_run, runs
from app.openfoam import (
    collect_artifacts,
    detect_solver,
    parse_openfoam_log_line,
    patch_control_dict_for_test,
)


def detect_mesh_file(case_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    Detect if there's an external mesh file that needs conversion.
    
    Returns (mesh_path, mesh_format) or (None, None) if no external mesh.
    """
    mesh_extensions = [".msh", ".cas", ".cgns", ".unv", ".neu"]
    
    for ext in mesh_extensions:
        matches = list(case_dir.glob(f"*{ext}"))
        if matches:
            return matches[0], ext.lstrip(".")
    
    return None, None


def get_mesh_converter(mesh_format: str) -> str:
    """Get the OpenFOAM mesh converter command for a given format."""
    converters = {
        "msh": "fluentMeshToFoam",
        "cas": "fluentMeshToFoam", 
        "cgns": "cgnsToFoam",
        "unv": "ideasUnvToFoam",
        "neu": "gambitToFoam",
    }
    return converters.get(mesh_format, "fluentMeshToFoam")


async def run_openfoam_command(
    run_id: str,
    command: str,
    case_dir: Path,
    event_prefix: str = "cmd",
    emit_progress: bool = True,
) -> Tuple[int, str, str]:
    """
    Run an OpenFOAM command and stream its output.
    
    Returns (exit_code, stdout, stderr).
    """
    full_cmd = build_shell_command(f"cd {case_dir} && {command}")
    
    process = await asyncio.create_subprocess_shell(
        full_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    stdout_lines = []
    log_buffer = []
    buffer_flush_interval = 10  # Flush every N lines
    
    # Read stdout line by line
    async for raw_line in process.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        stdout_lines.append(line)
        
        # Batch log lines for efficiency (only emit if progress tracking enabled)
        if emit_progress and line.strip():
            log_buffer.append(line)
            
            # Flush buffer periodically or on important lines
            is_important = any(keyword in line for keyword in [
                "Error", "error", "Warning", "FATAL", "Solving for",
                "Time =", "End", "Finished", "cells", "faces", "patches"
            ])
            
            if len(log_buffer) >= buffer_flush_interval or is_important:
                # Send batched log
                batched_msg = log_buffer[-1] if is_important else f"[{len(log_buffer)} lines]"
                emit_event(
                    run_id, f"{event_prefix}_log", batched_msg,
                    payload={
                        "stream": "stdout",
                        "lines_processed": len(stdout_lines),
                        "important": is_important,
                    }
                )
                log_buffer = []
    
    # Flush remaining buffer
    if log_buffer:
        emit_event(
            run_id, f"{event_prefix}_log", f"Completed ({len(stdout_lines)} lines)",
            payload={"stream": "stdout", "lines_processed": len(stdout_lines)}
        )
    
    # Wait for completion
    await process.wait()
    
    # Capture stderr
    stderr_data = await process.stderr.read()
    stderr_text = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
    
    if stderr_text.strip():
        # Only emit stderr if it contains actual errors (not just warnings)
        has_real_error = any(word in stderr_text.lower() for word in ["error", "fatal", "failed"])
        if has_real_error:
            emit_event(
                run_id, f"{event_prefix}_log", stderr_text[:2000],
                level="error", payload={"stream": "stderr"}
            )
    
    return process.returncode, "\n".join(stdout_lines), stderr_text


async def run_simulation(run_id: str, case_dir: Path, mode: RunMode):
    """
    Execute OpenFOAM simulation in the background.

    Flow:
    1. Check for external mesh file and convert if needed (fluentMeshToFoam, etc.)
    2. Run blockMesh if blockMeshDict exists and no polyMesh
    3. Run checkMesh
    4. Run the solver

    For TEST mode: override controlDict to run only 1 iteration.
    For FULL mode: run as-is.
    """
    try:
        run = get_run(run_id)
        if run:
            run.status = RunStatus.RUNNING
            run.started_at = datetime.utcnow().isoformat() + "Z"
        
        start_time = time.time()
        poly_mesh_dir = case_dir / "constant" / "polyMesh"

        # ─── Step 1: Check for external mesh and convert if needed ───
        mesh_file, mesh_format = detect_mesh_file(case_dir)
        
        if mesh_file and not poly_mesh_dir.exists():
            if run:
                run.status = RunStatus.MESHING
            
            emit_event(
                run_id, "mesh_conversion_started",
                f"Converting mesh: {mesh_file.name} ({mesh_format} format)",
                payload={"mesh_file": mesh_file.name, "format": mesh_format}
            )
            
            converter = get_mesh_converter(mesh_format)
            exit_code, stdout, stderr = await run_openfoam_command(
                run_id, f"{converter} {mesh_file.name}", case_dir, event_prefix="mesh"
            )
            
            # Check if mesh was ACTUALLY created, not just exit code
            # OpenFOAM converters often return exit code 1 due to warnings
            # but still successfully create the mesh
            mesh_created = poly_mesh_dir.exists() and (poly_mesh_dir / "points").exists()
            
            if not mesh_created:
                # Mesh conversion truly failed
                error_msg = stderr[:500] if stderr else f"No mesh created (exit code {exit_code})"
                if run:
                    run.status = RunStatus.FAILED
                    run.error = f"Mesh conversion failed: {error_msg}"
                    run.exit_code = exit_code
                emit_event(
                    run_id, "mesh_conversion_failed",
                    f"Mesh conversion failed: {error_msg}",
                    level="error",
                    payload={"exit_code": exit_code, "stderr": stderr[:2000]}
                )
                return
            
            # Mesh was created (even if exit code was non-zero due to warnings)
            warning_note = " (with warnings)" if exit_code != 0 else ""
            emit_event(
                run_id, "mesh_conversion_complete",
                f"Mesh conversion successful{warning_note}",
                payload={
                    "converter": converter,
                    "exit_code": exit_code,
                    "had_warnings": exit_code != 0,
                }
            )
            
            # ─── Post-mesh-conversion fixes ───
            # Run fix_mesh_setup.sh if it exists (fixes boundary types, wallDist, etc.)
            fix_script = case_dir / "fix_mesh_setup.sh"
            if fix_script.exists():
                emit_event(
                    run_id, "post_mesh_fix_started",
                    "Running post-mesh-conversion fixes (boundary types, fvSchemes wallDist)..."
                )
                fix_exit, fix_stdout, fix_stderr = await run_openfoam_command(
                    run_id, "bash fix_mesh_setup.sh", case_dir,
                    event_prefix="post_mesh_fix"
                )
                if fix_exit != 0:
                    emit_event(
                        run_id, "post_mesh_fix_warning",
                        f"Post-mesh fix script had issues (exit code {fix_exit}), continuing...",
                        level="warn",
                        payload={"stderr": fix_stderr[:1000]}
                    )
                else:
                    emit_event(
                        run_id, "post_mesh_fix_complete",
                        "Post-mesh-conversion fixes applied successfully"
                    )

        # ─── Step 2: Run blockMesh if needed ───
        block_mesh_dict = case_dir / "system" / "blockMeshDict"
        if block_mesh_dict.exists() and not poly_mesh_dir.exists():
            if run:
                run.status = RunStatus.MESHING
            emit_event(run_id, "blockmesh_started", "Running blockMesh")
            
            exit_code, stdout, stderr = await run_openfoam_command(
                run_id, "blockMesh", case_dir, event_prefix="mesh"
            )
            
            if exit_code != 0:
                if run:
                    run.status = RunStatus.FAILED
                    run.error = f"blockMesh failed: {stderr[:500]}"
                    run.exit_code = exit_code
                emit_event(
                    run_id, "blockmesh_failed",
                    f"blockMesh failed with exit code {exit_code}",
                    level="error",
                    payload={"exit_code": exit_code, "stderr": stderr[:2000]}
                )
                return
            
            emit_event(run_id, "blockmesh_complete", "blockMesh completed successfully")

        # ─── Verify mesh exists ───
        if not poly_mesh_dir.exists():
            if run:
                run.status = RunStatus.FAILED
                run.error = "No mesh found: constant/polyMesh does not exist"
            emit_event(
                run_id, "run_failed",
                "No mesh found - need either external mesh file or blockMeshDict",
                level="error"
            )
            return

        # ─── Step 3: Run checkMesh (optional, don't fail on warnings) ───
        emit_event(run_id, "checkmesh_started", "Running checkMesh")
        exit_code, stdout, stderr = await run_openfoam_command(
            run_id, "checkMesh", case_dir, event_prefix="mesh"
        )
        emit_event(
            run_id, "checkmesh_complete",
            f"checkMesh completed (exit code: {exit_code})",
            level="warn" if exit_code != 0 else "info"
        )

        # ─── Step 4: Run the solver ───
        if run:
            run.status = RunStatus.RUNNING
        solver = detect_solver(case_dir)

        # For TEST mode, patch controlDict to run 1 iteration
        if mode == RunMode.TEST:
            patch_control_dict_for_test(case_dir)

        emit_event(
            run_id, "run_started",
            f"Starting {solver} ({mode.value} mode)",
            payload={"solver": solver, "mode": mode.value}
        )

        # Run the solver
        full_cmd = build_shell_command(f"cd {case_dir} && {solver}")
        
        process = await asyncio.create_subprocess_shell(
            full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        iteration = 0
        residuals: dict = {}

        # Stream stdout line by line
        async for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            
            # Emit raw log
            emit_event(run_id, "run_log", line, payload={"stream": "stdout"})
            
            # Parse for progress
            parsed = parse_openfoam_log_line(line)
            if parsed:
                if "field" in parsed and "residual" in parsed:
                    residuals[parsed["field"]] = parsed["residual"]
                if "time" in parsed:
                    iteration += 1
                    emit_event(
                        run_id, "run_progress",
                        f"Iteration {iteration}",
                        payload={
                            "iteration": iteration,
                            "time": parsed.get("time"),
                            "residuals": dict(residuals),
                        }
                    )
                    residuals.clear()

        # Wait for process to finish
        await process.wait()
        duration = time.time() - start_time

        # Capture any remaining stderr
        stderr_data = await process.stderr.read()
        if stderr_data:
            stderr_text = stderr_data.decode("utf-8", errors="replace")
            emit_event(
                run_id, "run_log", stderr_text[:2000],
                level="warn", payload={"stream": "stderr"}
            )

        # Update run status
        if run:
            run.exit_code = process.returncode
            run.duration_seconds = round(duration, 2)
            run.completed_at = datetime.utcnow().isoformat() + "Z"

        if process.returncode == 0:
            if run:
                run.status = RunStatus.SUCCEEDED
            emit_event(
                run_id, "run_succeeded",
                f"Simulation completed in {duration:.1f}s",
                payload={
                    "exit_code": 0,
                    "duration_seconds": round(duration, 2),
                    "mode": mode.value,
                }
            )

            # Post-process: collect artifacts
            artifacts = collect_artifacts(run_id, case_dir)
            emit_event(
                run_id, "artifacts_ready",
                f"Results ready ({len(artifacts)} files)",
                payload={"artifacts": artifacts}
            )
        else:
            if run:
                run.status = RunStatus.FAILED
                run.error = f"Solver exited with code {process.returncode}"
            
            stderr_text = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
            emit_event(
                run_id, "run_failed",
                f"Solver failed (exit code {process.returncode})",
                level="error",
                payload={
                    "exit_code": process.returncode,
                    "duration_seconds": round(duration, 2),
                    "stderr": stderr_text[:2000],
                }
            )

    except Exception as e:
        run = get_run(run_id)
        if run:
            run.status = RunStatus.FAILED
            run.error = str(e)
        emit_event(run_id, "run_failed", f"Error: {str(e)}", level="error")
