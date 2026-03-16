# simd_agent/error_summarizer.py
"""Error summarizer agent for analyzing sandbox failures."""

import asyncio
import logging
import re
from typing import Any

from simd_agent.run.event_bus import EventBus
from simd_agent.models import ErrorSummary
from simd_agent.settings import get_settings

logger = logging.getLogger(__name__)

# Common OpenFOAM error patterns and their root causes
ERROR_PATTERNS = [
    # blockMesh errors
    {
        "pattern": r"FOAM FATAL ERROR.*?blockMesh",
        "root_cause": "blockMesh failed - mesh generation error",
        "affected_files": ["system/blockMeshDict"],
    },
    {
        "pattern": r"Cannot find patchField entry",
        "root_cause": "Missing boundary condition definition",
        "affected_files": ["0/*"],
    },
    {
        "pattern": r"fvPatchField.*?type not found",
        "root_cause": "Invalid boundary condition type specified",
        "affected_files": ["0/*"],
    },
    # Solver errors
    {
        "pattern": r"Maximum number of iterations.*?exceeded",
        "root_cause": "Solution divergence - solver did not converge",
        "affected_files": ["system/fvSolution", "system/fvSchemes"],
    },
    {
        "pattern": r"Floating point exception",
        "root_cause": "Numerical instability causing floating point exception",
        "affected_files": ["system/fvSolution", "system/controlDict"],
    },
    # Mesh errors
    {
        "pattern": r"checkMesh.*?FAILED",
        "root_cause": "Mesh quality check failed",
        "affected_files": ["system/blockMeshDict", "constant/polyMesh"],
    },
    {
        "pattern": r"negative volume",
        "root_cause": "Mesh has negative volume cells",
        "affected_files": ["system/blockMeshDict"],
    },
    {
        "pattern": r"face area magnitudes? .*? less than",
        "root_cause": "Mesh has very small faces",
        "affected_files": ["system/blockMeshDict"],
    },
    # File errors
    {
        "pattern": r"cannot find file",
        "root_cause": "Required file not found",
        "affected_files": [],
    },
    {
        "pattern": r"keyword.*?not found",
        "root_cause": "Missing required keyword in configuration file",
        "affected_files": [],
    },
    # Boundary condition errors
    {
        "pattern": r"inlet.*?not defined",
        "root_cause": "Inlet boundary condition not defined",
        "affected_files": ["0/*", "system/blockMeshDict"],
    },
    {
        "pattern": r"outlet.*?not defined",
        "root_cause": "Outlet boundary condition not defined",
        "affected_files": ["0/*", "system/blockMeshDict"],
    },
]

# Actionable fixes for common issues
ACTIONABLE_FIXES = {
    "blockMesh failed": [
        {"action": "check_block_vertices", "description": "Verify block vertex coordinates are correct"},
        {"action": "check_grading", "description": "Reduce mesh grading if too aggressive"},
    ],
    "Solution divergence": [
        {"action": "reduce_relaxation", "description": "Reduce relaxation factors in fvSolution"},
        {"action": "improve_mesh", "description": "Improve mesh quality near walls"},
        {"action": "reduce_timestep", "description": "Reduce time step for transient cases"},
    ],
    "Numerical instability": [
        {"action": "use_upwind", "description": "Switch to upwind schemes for stability"},
        {"action": "reduce_courant", "description": "Reduce maximum Courant number"},
    ],
    "Missing boundary": [
        {"action": "add_boundary", "description": "Add missing boundary condition to 0/* files"},
        {"action": "check_patch_names", "description": "Ensure patch names match between blockMeshDict and field files"},
    ],
}


class ErrorSummarizer:
    """Analyzes sandbox execution failures and provides actionable fixes.
    
    Uses pattern matching for common errors and optionally LLM for complex cases.
    """
    
    def __init__(
        self,
        event_bus: EventBus | None = None,
        use_llm: bool = False,
        code_generator: Any = None,
    ):
        """Initialize the error summarizer.
        
        Args:
            event_bus: Optional event bus for progress updates
            use_llm: Whether to use LLM for complex analysis
            code_generator: Optional CodeGenerator from codegen
        """
        self.event_bus = event_bus
        self.use_llm = use_llm
        self.code_generator = code_generator
        self.settings = get_settings()
    
    async def summarize(
        self,
        logs: str,
        exit_code: int | None,
        current_files: dict[str, str] | None = None,
    ) -> ErrorSummary:
        """Analyze sandbox execution logs and produce error summary.

        Strategy:
        1. Always run LLM diagnosis with full context (error + all generated files).
           The LLM understands OpenFOAM errors far better than regex patterns.
        2. Fall back to pattern matching if LLM fails.
        """
        # Try LLM diagnosis first — it has full context and OpenFOAM knowledge
        try:
            summary = await self._llm_diagnose(logs, exit_code, current_files)
            if summary and summary.confidence >= 0.5:
                return summary
        except Exception as _e:
            logger.warning(f"[DIAGNOSE] LLM diagnosis failed (falling back to patterns): {_e}")

        # Fallback: pattern matching
        return self._pattern_match(logs, exit_code)

    async def _llm_diagnose(
        self,
        logs: str,
        exit_code: int | None,
        current_files: dict[str, str] | None = None,
    ) -> ErrorSummary | None:
        """Use Gemini to diagnose the OpenFOAM error with full case context."""
        try:
            from google import genai as google_genai
            from google.genai import types as genai_types
            settings = get_settings()
            api_key = getattr(settings, "gemini_api_key", None)
            model_name = getattr(settings, "gemini_model", "gemini-2.0-flash-001")
            if not api_key:
                return None

            client = google_genai.Client(api_key=api_key)

            # Build the diagnosis prompt
            file_listing = ""
            if current_files:
                # Include the most relevant files for diagnosis (limit total size)
                _priority = [
                    "system/fvSolution", "system/fvSchemes", "constant/thermophysicalProperties",
                    "system/fvOptions", "0/U", "0/p", "0/T",
                ]
                _shown: list[str] = []
                _total = 0
                for _p in _priority + [k for k in current_files if k not in _priority]:
                    _content = current_files.get(_p, "")
                    if not _content:
                        continue
                    _shown.append(f"\n### {_p}\n```\n{_content[:1500]}\n```")
                    _total += len(_content)
                    if _total > 8000:
                        break
                file_listing = "\n## Generated case files\n" + "".join(_shown)

            prompt = (
                "You are an expert OpenFOAM CFD engineer. "
                "Analyze this simulation failure and identify the exact root cause.\n\n"
                f"## Error output (exit code: {exit_code})\n"
                f"```\n{logs[-4000:]}\n```"
                f"{file_listing}\n\n"
                "Respond in this exact format:\n"
                "ROOT_CAUSE: <one sentence — the specific OpenFOAM error and why it happens>\n"
                "AFFECTED_FILES: <comma-separated list of files to fix>\n"
                "FIX_1: <specific change to make>\n"
                "FIX_2: <optional second change>\n"
                "CONFIDENCE: <0.0-1.0>"
            )

            resp = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.2),
            )
            text = resp.text or ""
            logger.info(f"[DIAGNOSE] LLM response:\n{text[:500]}")

            root_cause = "Unknown"
            affected: list[str] = []
            fixes: list[dict] = []
            confidence = 0.6

            for line in text.splitlines():
                line = line.strip()
                if line.startswith("ROOT_CAUSE:"):
                    root_cause = line.split(":", 1)[1].strip()
                elif line.startswith("AFFECTED_FILES:"):
                    affected = [f.strip() for f in line.split(":", 1)[1].split(",") if f.strip()]
                elif line.startswith("FIX_"):
                    fix_text = line.split(":", 1)[1].strip() if ":" in line else line
                    if fix_text:
                        fixes.append({"action": "llm_fix", "description": fix_text})
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            return ErrorSummary(
                root_cause=root_cause,
                actionable_changes=fixes,
                affected_files=affected,
                confidence=confidence,
            )
        except Exception as exc:
            logger.warning(f"[DIAGNOSE] _llm_diagnose error: {exc}")
            return None
    
    def _pattern_match(
        self,
        logs: str,
        exit_code: int | None,
    ) -> ErrorSummary:
        """Use pattern matching to identify errors."""
        logs_lower = logs.lower()
        
        matched_patterns = []
        for pattern_info in ERROR_PATTERNS:
            pattern = pattern_info["pattern"]
            if re.search(pattern, logs, re.IGNORECASE | re.DOTALL):
                matched_patterns.append(pattern_info)
        
        if not matched_patterns:
            # Generic error based on exit code
            if exit_code and exit_code != 0:
                return ErrorSummary(
                    root_cause=f"Process exited with code {exit_code}",
                    actionable_changes=[
                        {"action": "check_logs", "description": "Review full logs for error details"},
                    ],
                    affected_files=[],
                    confidence=0.3,
                )
            return ErrorSummary(
                root_cause="Unknown error",
                actionable_changes=[],
                affected_files=[],
                confidence=0.1,
            )
        
        # Use the first matched pattern (most specific)
        primary = matched_patterns[0]
        root_cause = primary["root_cause"]
        affected_files = primary.get("affected_files", [])
        
        # Find actionable fixes
        actionable_changes = []
        for key, fixes in ACTIONABLE_FIXES.items():
            if key.lower() in root_cause.lower():
                actionable_changes.extend(fixes)
                break
        
        # Calculate confidence based on number of patterns matched
        confidence = min(0.5 + (len(matched_patterns) * 0.1), 0.9)
        
        return ErrorSummary(
            root_cause=root_cause,
            actionable_changes=actionable_changes,
            affected_files=affected_files,
            confidence=confidence,
        )
    
    async def _llm_analyze(
        self,
        logs: str,
        exit_code: int | None,
        current_files: dict[str, str] | None = None,
    ) -> ErrorSummary:
        """Use LLM to analyze complex errors."""
        try:
            from codegen import GenerationContext
            
            # Build requirements string with all context
            requirements_parts = [
                "Analyze the following OpenFOAM execution logs and determine the root cause of failure.",
                f"\n\nExit code: {exit_code}",
                f"\n\nLogs (last 200 lines):\n{logs[-10000:]}",
            ]
            
            # Add current files if provided
            if current_files:
                file_summary = "\n".join(f"- {path}" for path in current_files.keys())
                requirements_parts.append(f"\n\nCase files present:\n{file_summary}")
            
            # Build context for error analysis (using codefix task for error analysis)
            context = GenerationContext(
                task="codefix",
                domain="openfoam",
                requirements="".join(requirements_parts),
                previous_code="",
                sandbox_error=f"Exit code: {exit_code}",
                sandbox_logs=logs[-10000:],
            )
            
            # Generate error analysis (codegen.generate is synchronous)
            result = await asyncio.to_thread(self.code_generator.generate, context)
            
            # Parse LLM response
            return self._parse_llm_response(result.final_text)
        except Exception as e:
            logger.error(f"LLM error analysis failed: {e}")
            # Fall back to pattern matching
            return self._pattern_match(logs, exit_code)
    
    def _parse_llm_response(self, response: str) -> ErrorSummary:
        """Parse LLM response into ErrorSummary."""
        # Extract root cause
        root_cause = "Unknown error"
        if "root cause:" in response.lower():
            match = re.search(r"root cause[:\s]+(.+?)(?:\n|$)", response, re.IGNORECASE)
            if match:
                root_cause = match.group(1).strip()
        elif response:
            # Use first line as root cause
            root_cause = response.split("\n")[0][:200]
        
        # Extract affected files
        affected_files = []
        file_matches = re.findall(r"(?:file|path)[:\s]+([^\n,]+)", response, re.IGNORECASE)
        affected_files.extend(f.strip() for f in file_matches)
        
        # Extract actionable changes
        actionable_changes = []
        action_matches = re.findall(r"(?:fix|change|modify|action)[:\s]+(.+?)(?:\n|$)", response, re.IGNORECASE)
        for action in action_matches:
            actionable_changes.append({
                "action": "llm_suggestion",
                "description": action.strip(),
            })
        
        return ErrorSummary(
            root_cause=root_cause,
            actionable_changes=actionable_changes,
            affected_files=affected_files,
            confidence=0.7,
        )
    
    def truncate_logs(self, logs: str, max_lines: int | None = None) -> str:
        """Truncate logs to last N lines.
        
        Args:
            logs: Full log text
            max_lines: Maximum lines (uses settings default if None)
            
        Returns:
            Truncated log text
        """
        max_lines = max_lines or self.settings.max_log_lines_in_event
        lines = logs.split("\n")
        if len(lines) > max_lines:
            return "\n".join(lines[-max_lines:])
        return logs
