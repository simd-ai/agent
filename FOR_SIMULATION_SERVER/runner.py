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


_DECOMPOSE_PAR_DICT_TEMPLATE = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains  {n};

method  scotch;

// ************************************************************************* //
"""


def _ensure_decompose_par_dict(case_dir: Path, n_cores: int) -> None:
    """Write system/decomposeParDict if not already present or if n_cores changed."""
    dict_path = case_dir / "system" / "decomposeParDict"
    if dict_path.exists():
        content = dict_path.read_text()
        # If the existing file already has the correct n, leave it alone.
        if f"numberOfSubdomains  {n_cores};" in content or f"numberOfSubdomains {n_cores};" in content:
            return
    dict_path.write_text(_DECOMPOSE_PAR_DICT_TEMPLATE.format(n=n_cores))


def _patch_control_dict_for_test_steps(case_dir: Path, n_steps: int = 20) -> None:
    """Patch system/controlDict so the solver stops after exactly n_steps time-steps.

    Strategy:
    1. Read the current deltaT (falling back to 1.0 if not found).
    2. Set endTime = startTime + n_steps * deltaT.
    3. Force stopAt endTime so no residual control overrides it.
    4. Set writeInterval = endTime so we get exactly one write at the end.

    This is far cheaper than the old approach of setting endTime=1 when
    deltaT is very small (e.g. deltaT=0.001 would cause 1000 iterations).
    """
    import re

    ctrl = case_dir / "system" / "controlDict"
    if not ctrl.exists():
        return

    text = ctrl.read_text()

    def _float(key: str, default: float) -> float:
        m = re.search(rf"^\s*{key}\s+([\d.eE+\-]+)\s*;", text, re.MULTILINE)
        return float(m.group(1)) if m else default

    delta_t   = _float("deltaT",   1.0)
    start_t   = _float("startTime", 0.0)
    end_time  = start_t + n_steps * delta_t

    def _replace(key: str, value: str) -> str:
        nonlocal text
        pattern = rf"(^\s*{key}\s+)[^\n]+;"
        replacement = rf"\g<1>{value};"
        new_text, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
        if n == 0:
            # Key not present — append before the final closing line
            new_text = text.rstrip().rstrip("//").rstrip() + f"\n{key}  {value};\n"
        text = new_text
        return text

    _replace("stopAt",       "endTime")
    _replace("endTime",      f"{end_time:.6g}")
    _replace("writeControl", "timeStep")
    _replace("writeInterval", str(n_steps))   # write once at the last step

    ctrl.write_text(text)


async def run_simulation(run_id: str, case_dir: Path, mode: RunMode, n_cores: int = 1):
    # Guard: if routes.py passed a FastAPI Form descriptor instead of a parsed int, coerce it.
    try:
        n_cores = int(n_cores)
    except (TypeError, ValueError):
        n_cores = 1
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

        # ─── Step 4: Run the solver (serial or MPI parallel) ───
        if run:
            run.status = RunStatus.RUNNING
        solver = detect_solver(case_dir)
        parallel = n_cores > 1

        # For TEST mode, patch controlDict to run exactly TEST_STEPS time-steps.
        # We read the actual deltaT from the generated controlDict so we don't
        # waste time when deltaT is very small (e.g. 0.001 → endTime=1 would
        # mean 1000 iterations with the old approach).
        if mode == RunMode.TEST:
            _patch_control_dict_for_test_steps(case_dir, n_steps=20)

        # ─── Step 4a: decomposePar (only when running parallel) ───
        if parallel:
            _ensure_decompose_par_dict(case_dir, n_cores)

            emit_event(
                run_id, "decompose_started",
                f"Decomposing mesh into {n_cores} subdomains…",
                payload={"n_cores": n_cores},
            )
            dec_code, _, dec_err = await run_openfoam_command(
                run_id, "decomposePar", case_dir, event_prefix="decompose"
            )
            if dec_code != 0:
                if run:
                    run.status = RunStatus.FAILED
                    run.error = f"decomposePar failed (exit code {dec_code})"
                emit_event(
                    run_id, "decompose_failed",
                    f"decomposePar failed (exit code {dec_code})",
                    level="error",
                    payload={"exit_code": dec_code, "stderr": dec_err[:2000]},
                )
                return
            emit_event(
                run_id, "decompose_complete",
                f"Mesh decomposed into {n_cores} subdomains",
                payload={"n_cores": n_cores},
            )

        # ─── Step 4b: Solver command ───
        if parallel:
            solver_cmd = f"mpirun -np {n_cores} {solver} -parallel"
        else:
            solver_cmd = solver

        emit_event(
            run_id, "run_started",
            f"Starting {solver} ({mode.value} mode, {n_cores} core{'s' if n_cores > 1 else ''})",
            payload={"solver": solver, "mode": mode.value, "n_cores": n_cores, "parallel": parallel},
        )

        full_cmd = build_shell_command(f"cd {case_dir} && {solver_cmd}")

        process = await asyncio.create_subprocess_shell(
            full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        import re as _re

        # ── Per-time-step patterns ───────────────────────────────────────────
        # OpenFOAM log structure per time step:
        #   Time = 0.482
        #   Courant Number mean: 0.04 max: 0.27
        #   smoothSolver: Solving for Ux, Initial residual = 3.67e-3, Final residual = 1.03e-8, No Iterations 1
        #   continuity errors : sum local = 1.33e-7, global = 5.7e-10, cumulative = -4.16e-6
        #   ExecutionTime = 41.2 s  ClockTime = 42 s   ← end of step → emit
        _RE_TIME    = _re.compile(r'^Time\s*=\s*([\d.e+\-]+)')
        _RE_SOLVE   = _re.compile(
            r'Solving for (\w+),\s*Initial residual\s*=\s*([\d.e+\-]+),'
            r'\s*Final residual\s*=\s*([\d.e+\-]+),\s*No Iterations\s+(\d+)'
        )
        _RE_COURANT = _re.compile(r'Courant Number mean:\s*([\d.e+\-]+)\s+max:\s*([\d.e+\-]+)')
        _RE_CONT    = _re.compile(
            r'continuity errors.*?sum local\s*=\s*([\d.e+\-]+),'
            r'\s*global\s*=\s*([\d.e+\-]+),\s*cumulative\s*=\s*([-\d.e+\-]+)'
        )
        _RE_EXEC    = _re.compile(r'ExecutionTime\s*=\s*([\d.]+)\s*s.*?ClockTime\s*=\s*([\d.]+)\s*s')

        # ── Accumulator for the current time step ────────────────────────────
        def _fresh_step() -> dict:
            return {"time": None, "fields": [], "residuals": {}, "courant": None,
                    "continuity": None, "execution": None}

        step = _fresh_step()
        iteration = 0
        stdout_lines: list[str] = []

        # Stream stdout line by line
        async for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            stdout_lines.append(line)

            # Emit raw log
            emit_event(run_id, "run_log", line, payload={"stream": "stdout"})

            # ── Time marker: start of new step ───────────────────────────────
            m = _RE_TIME.match(line)
            if m:
                step["time"] = float(m.group(1))
                continue

            # ── Solver residuals ─────────────────────────────────────────────
            m = _RE_SOLVE.search(line)
            if m:
                field = m.group(1)
                if field not in step["fields"]:
                    step["fields"].append(field)
                step["residuals"][field] = {
                    "initial": float(m.group(2)),
                    "final":   float(m.group(3)),
                    "iters":   int(m.group(4)),
                }
                continue

            # ── Courant number ───────────────────────────────────────────────
            m = _RE_COURANT.search(line)
            if m:
                step["courant"] = {"mean": float(m.group(1)), "max": float(m.group(2))}
                continue

            # ── Continuity errors ────────────────────────────────────────────
            m = _RE_CONT.search(line)
            if m:
                step["continuity"] = {
                    "local":      float(m.group(1)),
                    "global":     float(m.group(2)),
                    "cumulative": float(m.group(3)),
                }
                continue

            # ── ExecutionTime: end of step → emit run_progress ───────────────
            m = _RE_EXEC.search(line)
            if m and step["time"] is not None and step["residuals"]:
                t, ct = float(m.group(1)), float(m.group(2))
                step["execution"] = {
                    "step_seconds":  t,
                    "clock_seconds": ct,
                    "label":         f"{t:.2f}s (clock {ct:.2f}s)",
                }
                iteration += 1
                emit_event(
                    run_id, "run_progress",
                    f"Time {step['time']:.4g} | Step {iteration}",
                    payload={
                        "iteration":   iteration,
                        "time":        step["time"],
                        "fields":      list(step["fields"]),
                        "residuals":   dict(step["residuals"]),
                        "courant":     step["courant"],
                        "continuity":  step["continuity"],
                        "execution":   step["execution"],
                    },
                )
                step = _fresh_step()

        # Wait for process to finish
        await process.wait()
        duration = time.time() - start_time

        # Capture any remaining stderr
        stderr_data = await process.stderr.read()
        stderr_text = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
        if stderr_text.strip():
            emit_event(
                run_id, "run_log", stderr_text[:2000],
                level="warn", payload={"stream": "stderr"}
            )

        # ─── Step 4c: reconstructPar (parallel runs only) ───
        if parallel and process.returncode == 0:
            emit_event(
                run_id, "reconstruct_started",
                "Reconstructing parallel results…",
                payload={"n_cores": n_cores},
            )
            rec_code, _, rec_err = await run_openfoam_command(
                run_id, "reconstructPar", case_dir, event_prefix="reconstruct"
            )
            if rec_code != 0:
                # Not fatal — results still exist in processor* dirs; warn but continue
                emit_event(
                    run_id, "reconstruct_failed",
                    f"reconstructPar failed (exit code {rec_code}) — results kept in processor* directories",
                    level="warn",
                    payload={"exit_code": rec_code, "stderr": rec_err[:2000]},
                )
            else:
                emit_event(
                    run_id, "reconstruct_complete",
                    "Parallel results reconstructed successfully",
                    payload={"n_cores": n_cores},
                )

        # ─── Update run status ───
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
                    "n_cores": n_cores,
                }
            )

            # Post-process: collect artifacts
            artifacts = collect_artifacts(run_id, case_dir)
            emit_event(
                run_id, "artifacts_ready",
                f"Results ready ({len(artifacts)} files)",
                payload={"artifacts": artifacts}
            )

            # Kick off VTK precomputation in the background so all timestep
            # VTPs are ready on disk before the agent (or frontend) asks.
            # Lazy import avoids a circular dependency with routes.py.
            async def _kick_vtk_precompute():
                try:
                    from app.routes import _precompute_all_timesteps
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, _precompute_all_timesteps, case_dir, run_id
                    )
                    emit_event(
                        run_id, "vtk_precompute_complete",
                        "VTK precomputation finished — all timestep VTPs ready",
                    )
                except Exception as exc:
                    print(f"[VTK] Background precompute failed for {run_id}: {exc}")

            asyncio.create_task(_kick_vtk_precompute())
        else:
            if run:
                run.status = RunStatus.FAILED
                run.error = f"Solver exited with code {process.returncode}"

            # OpenFOAM writes FATAL errors and FPE stack traces to stdout, not stderr.
            # Include the last 100 stdout lines so the agent can diagnose the crash
            # even when stderr is empty (e.g. exit code 136 = SIGFPE).
            stdout_tail = "\n".join(stdout_lines[-100:]) if stdout_lines else ""

            emit_event(
                run_id, "run_failed",
                f"Solver failed (exit code {process.returncode})",
                level="error",
                payload={
                    "exit_code": process.returncode,
                    "duration_seconds": round(duration, 2),
                    # Both streams — agent uses whichever is non-empty
                    "stderr": stderr_text[:2000],
                    "stdout": stdout_tail[-4000:],   # last ~4k chars of solver output
                }
            )

    except Exception as e:
        run = get_run(run_id)
        if run:
            run.status = RunStatus.FAILED
            run.error = str(e)
        emit_event(run_id, "run_failed", f"Error: {str(e)}", level="error")
