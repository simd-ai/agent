# simd_agent/precheck.py
"""Precheck service — analyzes user prompts and extracts CFD simulation specs.

Uses Google GenAI with structured function-calling for reliable JSON output.
All Pydantic models, fluid presets, and the tool schema live in precheck_models.py.
"""

import logging
import math

from google import genai
from google.genai import types

from simd_agent.settings import get_settings
from simd_agent.precheck_models import (
    # request / response
    PrecheckRequest, PrecheckResponse, SuggestedConfig,
    SolverSettings, FluidProperties, TurbulenceSettings,
    FieldBC, PatchBoundaryCondition,
    BoundaryHint, VelocityBC, PressureBC, TemperatureBC,
    Interpretation, ConfidenceScores,
    # constants
    FLUID_PRESETS, CRYOGENIC_KEYWORDS,
    PRECHECK_MODEL, REVIEW_MODEL, PRECHECK_TOOL_SCHEMA,
)

# ---------------------------------------------------------------------------
# Auxiliary tool schemas (only used internally in precheck.py)
# ---------------------------------------------------------------------------

# Pass-2a: quick spec completeness check — decides whether to skip the review
_SELF_VERIFY_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="submit_verification",
            description=(
                "Report whether the generated CFD spec is complete and physically consistent. "
                "Call this once after reviewing the spec."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "all_correct": types.Schema(
                        type="BOOLEAN",
                        description=(
                            "True ONLY if every required field has a non-null, physically valid value "
                            "and all BCs are physically consistent. False if anything is missing or wrong."
                        ),
                    ),
                    "issues": types.Schema(
                        type="ARRAY",
                        description="Concise description of each issue found. Empty list when all_correct=True.",
                        items=types.Schema(type="STRING"),
                    ),
                },
                required=["all_correct", "issues"],
            ),
        )
    ]
)

# Pass-2c: reconcile reviewer thinking → structured JSON
_RECONCILE_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="submit_reconciled_bcs",
            description=(
                "Return the corrected boundary conditions after cross-checking your chain-of-thought. "
                "Every numeric value MUST match exactly what was computed in the reasoning text."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "corrected_boundary_conditions": types.Schema(
                        type="ARRAY",
                        description="All patches with corrected values. Every patch must be listed.",
                        items=types.Schema(
                            type="OBJECT",
                            properties={
                                "patch_name":    types.Schema(type="STRING"),
                                "patch_class":   types.Schema(type="STRING"),
                                "confidence":    types.Schema(type="NUMBER",  nullable=True),
                                "U_type":        types.Schema(type="STRING",  nullable=True),
                                "U_value":       types.Schema(type="ARRAY",   nullable=True, items=types.Schema(type="NUMBER")),
                                "p_type":        types.Schema(type="STRING",  nullable=True),
                                "p_value":       types.Schema(type="NUMBER",  nullable=True),
                                "T_type":        types.Schema(type="STRING",  nullable=True),
                                "T_value":       types.Schema(type="NUMBER",  nullable=True),
                                "k_type":        types.Schema(type="STRING",  nullable=True),
                                "k_value":       types.Schema(type="NUMBER",  nullable=True),
                                "omega_type":    types.Schema(type="STRING",  nullable=True),
                                "omega_value":   types.Schema(type="NUMBER",  nullable=True),
                                "epsilon_type":  types.Schema(type="STRING",  nullable=True),
                                "epsilon_value": types.Schema(type="NUMBER",  nullable=True),
                                "nut_type":      types.Schema(type="STRING",  nullable=True),
                                "nut_value":     types.Schema(type="NUMBER",  nullable=True),
                            },
                            required=["patch_name", "patch_class"],
                        ),
                    ),
                },
                required=["corrected_boundary_conditions"],
            ),
        )
    ]
)

logger = logging.getLogger(__name__)


def _norm_conf(value: float | int) -> float:
    """Normalise a confidence value to [0, 1].

    The LLM sometimes returns percentages (e.g. 80, 100) instead of fractions
    (0.8, 1.0).  Any value > 1 is assumed to be a percentage and divided by 100.
    """
    v = float(value)
    return v / 100.0 if v > 1.0 else v


# ---------------------------------------------------------------------------
# PrecheckService
# ---------------------------------------------------------------------------

class PrecheckService:
    """Analyze user prompts and return structured CFD simulation configuration.
    
    Uses Gemini function-calling for reliable structured output.
    The fallback path (LLM failure) returns a minimal default config — it does
    NOT attempt complex regex parsing since the LLM handles all interpretation.
    """

    def __init__(self):
        self.settings = get_settings()
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    # ── Public API ───────────────────────────────────────────────────────────

    async def analyze(self, request: PrecheckRequest) -> PrecheckResponse:
        """Analyze a user prompt and return a structured PrecheckResponse."""
        validation_error = request.validate_prompt()
        if validation_error:
            return self._create_friendly_error_response(validation_error)
        try:
            prompt = self._build_analysis_prompt(request)
            logger.info(f"[Precheck] Calling {PRECHECK_MODEL} with tool calling")
            response = self.client.models.generate_content(
                model=PRECHECK_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    tools=[PRECHECK_TOOL_SCHEMA],
                    tool_config=types.ToolConfig(
                        thinking_config=types.ThinkingConfig(
                            include_thoughts=True
                        ),
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY",
                            allowed_function_names=["submit_cfd_configuration"],
                        )
                    ),
                ),
            )
            logger.info("[Precheck] Got function call response")
            return self._parse_function_call_response(response, request)
        except Exception as e:
            logger.exception(f"Precheck analysis failed: {e}")
            return self._create_fallback_response(request, str(e))

    # ── Prompt builder ───────────────────────────────────────────────────────

    def _build_analysis_prompt(self, request: PrecheckRequest) -> str:
        mesh = request.get_mesh()
        parts = [
            "Analyze this CFD simulation request and return a complete configuration with CALCULATED values.",
            "",
            "=== OUTPUT FORMATTING (apply in ALL text fields: message, summaries, assumptions, key_physics) ===",
            "• Patch / field / BC identifiers  → backtick code: `inlet`, `kOmegaSST`, `fixedValue`, `p`, `U`, `T`",
            "  NEVER wrap patch or field names in quotes (\"inlet\") — always use backticks: `inlet`. You don't need to mention the task you are doing, just the names of the patches, fields, and boundary conditions.",
            "• Turbulence quantities, Greek symbols → inline LaTeX: $k$, $\\omega$, $\\varepsilon$, $\\nu_t$, $\\mu$, $\\rho$",
            "• Scientific values                → inline LaTeX: $1.81 \\times 10^{-5}$, $\\rho = 808\\,\\text{kg/m}^3$",
            "• Derived formulas                 → inline LaTeX: $k = 1.5\\,(U I)^2$, $\\omega = \\sqrt{k}/(C_\\mu^{0.25} L)$",
            "• Important solver/model names     → **bold**: **kOmegaSST**, **rhoPimpleFoam**, **icoPolynomial**",
            "• Never write raw unicode symbols (ω ε μ ρ) — always use LaTeX inside $...$",
            "",
            f"User description: {request.prompt}",
            "",
        ]

        if mesh:
            parts += [
                "Mesh info:",
                f"  File: {mesh.file_name} | Cells: {mesh.check_mesh.cells:,}",
                "  Patches:",
            ]
            for p in mesh.patches:
                parts.append(f"    - {p.name}  (mesh type: {p.type}, cells: {p.n_cells})")
            parts.append("")

        if request.previous_config:
            import json
            parts += ["Previous config (user is refining):", json.dumps(request.previous_config, indent=2), ""]

        parts += [
            "=== COMPRESSIBILITY ===",
            "compressible  → cryogenic liquid (LN2/LNG/LH2/LOX/Helium), gas at Mach>0.3, large ΔT in gas",
            "               → solvers: rhoSimpleFoam (steady) / rhoPimpleFoam or compressibleInterFoam (transient)",
            "incompressible → stable liquid (water/oil), low-speed gas (Mach<0.3)",
            "               → solvers: simpleFoam / pimpleFoam",
            "",
            "=== FLUID GUIDE ===",
            "Stable liquids  (water, oil): incompressible; rhoConst EOS safe if ΔT small.",
            "Cryogenic liquids (LN2 bp=77K, LNG bp=111K, LH2 bp=20K, LOX bp=90K, LHe bp=4K):",
            "  ALWAYS compressible + enable_heat_transfer=true.",
            "  Hot or cold wall → fluid T rises above boiling → NEVER rhoConst. Use icoPolynomial or perfectGas.",
            "",
            "=== FLUID PROPERTIES ===",
            "Air:    rho=1.225,  mu=1.81e-5, Cp=1006,  k=0.0257,  T=293K",
            "Water:  rho=998.2,  mu=1.002e-3,Cp=4182,  k=0.598,   T=293K",
            "LN2:    rho=808,    mu=1.58e-4, Cp=2042,  k=0.140,   T=77K",
            "LNG:    rho=450,    mu=1.2e-4,  Cp=3500,  k=0.185,   T=111K",
            "Helium: rho=0.164,  mu=1.96e-5, Cp=5193,  k=0.152,   T=293K",
            "",
            "=== CALCULATIONS ===",
            "Velocity:  U = m_dot / (rho * A),  A = π*(D/2)²",
            "Turbulence (Cmu=0.09, I=0.05, L=0.07*Dh):",
            "  k = 1.5*(U*I)²  |  omega = sqrt(k)/(Cmu^0.25*L)  |  epsilon = Cmu^0.75*k^1.5/L",
            "",
            "=== BOUNDARY CONDITIONS (read user description carefully for each value) ===",
            "",
            "TEMPERATURE RULE — read this first:",
            "  • T_inlet  = temperature of the fluid entering the domain (e.g. '77 K LN2 inlet')",
            "  • T_wall   = temperature imposed on the solid wall   (e.g. 'wall heated to 400 K')",
            "  • T_outlet = zeroGradient — NEVER fixedValue on an outlet",
            "  ⚠️  NEVER assign T_wall to the inlet patch. NEVER assign T_inlet to the wall patch.",
            "  ⚠️  If the user does not mention a wall temperature, use zeroGradient on wall T.",
            "",
            "MANDATORY T ASSIGNMENT when enable_heat_transfer=true (NEVER omit T from any patch):",
            "  • inlet:   T_type='fixedValue',   T_value=<fluid_temperature in K>  ← FLUID temp, NOT wall temp",
            "  • outlet:  T_type='zeroGradient', T_value=null",
            "  • wall:    T_type='fixedValue',   T_value=<wall temperature in K>   ← WALL temp, NOT fluid temp",
            "             (if user gave no wall temperature: T_type='zeroGradient', T_value=null)",
            "  • empty:   T_type='empty',         T_value=null",
            "",
            "PRESSURE RULE:",
            "  • Incompressible outlet: p=fixedValue, value=0 (gauge).",
            "  • Compressible outlet: p=fixedValue, value = operating pressure from user context",
            "    (e.g. '1 atm' → 101325 Pa, '2 bar' → 200000 Pa, '0.5 MPa' → 500000 Pa).",
            "    If no operating pressure is stated, use 101325 Pa as a safe default.",
            "  ⚠️  NEVER invent a pressure value — derive it from what the user said.",
            "",
            "PATCH PATTERNS (list every field you must set for each patch):",
            "  INLET:    U_type='fixedValue',     U_value=[Ux,Uy,Uz] (computed from mass flow or stated velocity)",
            "            p_type='zeroGradient',   p_value=null",
            "            T_type='fixedValue',     T_value=<fluid_temperature K>  (heat transfer only)",
            "            k_type='fixedValue',     k_value=<k computed above>",
            "            omega_type='fixedValue', omega_value=<omega computed above>  (kOmegaSST)",
            "            epsilon_type='fixedValue', epsilon_value=<epsilon computed above>  (kEpsilon)",
            "            nut_type='calculated',   nut_value=0",
            "  OUTLET:   U_type='zeroGradient',   p_type='fixedValue' (see pressure rule)",
            "            T_type='zeroGradient',   T_value=null  (heat transfer only)",
            "            k_type='zeroGradient',   omega_type='zeroGradient',  nut_type='calculated'",
            "  WALL:     U_type='noSlip',          p_type='zeroGradient'",
            "            T_type per MANDATORY T ASSIGNMENT above",
            "            k_type='kqRWallFunction', omega_type='omegaWallFunction',",
            "            epsilon_type='epsilonWallFunction', nut_type='nutkWallFunction'",
            "  EMPTY:    frontAndBack in 2D mesh — ALL fields type='empty', patch_class='empty'. NEVER symmetry.",
            "  SYMMETRY: only when mesh patch type is symmetry/symmetryPlane.",
            "",
            "VALUE CONSISTENCY RULE:",
            "  Compute U, k, omega/epsilon once from the CALCULATIONS section above.",
            "  Use those EXACT same numbers in BOTH the message text AND the boundary_conditions array.",
            "  Do NOT write one value in the message and a different value in the structured fields.",
            "",
        ]

        return "\n".join(parts)

    # ── Auxiliary LLM passes ─────────────────────────────────────────────────

    async def _llm_self_verify(
        self,
        request: PrecheckRequest,
        response: PrecheckResponse,
    ) -> tuple[bool, list[str]]:
        """Pass-2a: quick LLM completeness check on the first-pass spec.

        Returns (all_correct, issues).  When all_correct=True the full review
        pass is skipped so a correct spec cannot be accidentally degraded.
        """
        import json as _json

        spec_json = _json.dumps(response.model_dump(by_alias=True), indent=2, default=str)

        # Detect heat transfer from the request flags or keywords
        enable_heat = (
            getattr(request, "enable_heat_transfer", False)
            or any(kw in request.prompt.lower() for kw in ("heat", "temperature", "thermal", "°c", "°f", "kelvin"))
        )

        ht_rule = (
            "\n• Heat transfer is active: `inlet` MUST have T_type='fixedValue' "
            "with a real fluid temperature in K (not null, not zero)."
            if enable_heat else ""
        )

        prompt = f"""\
You are a CFD expert doing a rapid quality check on an OpenFOAM configuration.

=== USER REQUEST ===
{request.prompt}

=== GENERATED SPEC ===
{spec_json}

Silently check for these physical issues:
• Any inlet/outlet/wall BC field that should have a value is null or zero{ht_rule}
• Turbulence inlet values (k, omega/epsilon) are missing or physically implausible
• Inlet velocity is null when the user specified a velocity or flow rate
• Pressure BCs are inverted (fixedValue at inlet instead of outlet)
• Any setting that is physically impossible or self-contradictory

If everything is set correctly and consistently → all_correct=True, empty issues list.
If anything is wrong or incomplete → all_correct=False, brief issues list (one string per problem)."""

        try:
            print(f"[Precheck/self-verify] → calling generate_content (model={REVIEW_MODEL})...", flush=True)
            resp = await self.client.aio.models.generate_content(
                model=REVIEW_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    tools=[_SELF_VERIFY_TOOL],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY",
                            allowed_function_names=["submit_verification"],
                        )
                    ),
                ),
            )
            print("[Precheck/self-verify] ✓ response received", flush=True)
            for part in resp.candidates[0].content.parts:
                if getattr(part, "function_call", None) is not None:
                    args = dict(part.function_call.args)
                    all_correct = bool(args.get("all_correct", False))
                    issues = list(args.get("issues", []))
                    print(f"[Precheck] Self-verify → all_correct={all_correct}, issues={issues}", flush=True)
                    return all_correct, issues
        except Exception as e:
            print(f"[Precheck/self-verify] ✗ FAILED: {e}", flush=True)
            logger.warning(f"[Precheck] Self-verify failed: {e}")

        # Fail-safe: assume something might be wrong and run the review
        return False, ["self-verify call failed; running full review"]

    async def _llm_reconcile(
        self,
        thinking_text: str,
        proposed_bcs: list[dict],
    ) -> list[dict]:
        """Pass-2c: ground-truth check — reviewer thinking vs. proposed JSON.

        The reviewer sometimes computes correct values in its chain-of-thought
        but transcribes them wrong into the structured function call.  This pass
        treats the thinking text as the authoritative source and returns a
        BC list where every value matches what was actually computed.
        """
        import json as _json

        if not thinking_text.strip():
            # No thinking available — nothing to cross-check against
            return proposed_bcs

        proposed_json = _json.dumps(proposed_bcs, indent=2, default=str)

        prompt = f"""\
You previously reviewed a CFD configuration. Here is your complete reasoning:

=== YOUR REASONING ===
{thinking_text}

=== YOUR PROPOSED CORRECTIONS ===
{proposed_json}

Task: verify that every numeric value in the proposed corrections EXACTLY matches
what you computed in your reasoning above.

Rules:
- If a value in the proposed corrections does NOT match your reasoning → fix it to match your reasoning.
- If a value is null but your reasoning computed a specific number → fill it in.
- If a value is correct already → keep it unchanged.
- Include ALL patches, not just changed ones.
- Do NOT introduce any new value that is not supported by your reasoning.
- Temperature rule: inlet T_value = fluid temperature you computed; wall T_value = wall temperature you computed."""

        try:
            resp = await self.client.aio.models.generate_content(
                model=REVIEW_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    tools=[_RECONCILE_TOOL],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY",
                            allowed_function_names=["submit_reconciled_bcs"],
                        )
                    ),
                ),
            )
            import json as _j2
            for part in resp.candidates[0].content.parts:
                if getattr(part, "function_call", None) is not None:
                    args = dict(part.function_call.args)
                    reconciled = list(args.get("corrected_boundary_conditions", []))
                    print(f"[Precheck] Reconcile → {len(reconciled)} patches:")
                    print(_j2.dumps(reconciled, indent=2, default=str))
                    return reconciled
        except Exception as e:
            logger.warning(f"[Precheck] Reconcile failed: {e}")

        return proposed_bcs

    # ── Streaming API ────────────────────────────────────────────────────────

    async def analyze_stream(self, request: PrecheckRequest):
        """Async generator for streaming precheck over WebSocket.

        Yields a sequence of typed event dicts:
            {"type": "start"}
            {"type": "thought",        "text": "<incremental thinking text>"}  # 0-N times
            {"type": "spec_generating"}                                          # once
            {"type": "spec",           "data": {<PrecheckResponse camelCase>}}  # once
            {"type": "done"}
            # OR on error:
            {"type": "error",          "message": "<description>"}
            {"type": "done"}
        """
        print(f"[Precheck] analyze_stream called — prompt={request.prompt[:60]!r}", flush=True)
        validation_error = request.validate_prompt()
        if validation_error:
            yield {"type": "start"}
            yield {"type": "error", "message": validation_error}
            yield {"type": "done"}
            return

        yield {"type": "start"}
        print("[Precheck] ✓ start yielded — building prompt...", flush=True)
        try:
            prompt = self._build_analysis_prompt(request)
            print(f"[Precheck] ✓ prompt built ({len(prompt)} chars)", flush=True)
            func_call_args: dict | None = None
            spec_generating_sent = False
            _thought_buf: list[str] = []  # accumulate for terminal print

            print(f"[Precheck] → calling generate_content_stream (model={PRECHECK_MODEL})...", flush=True)
            stream = await self.client.aio.models.generate_content_stream(
                model=PRECHECK_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                    tools=[PRECHECK_TOOL_SCHEMA],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY",
                            allowed_function_names=["submit_cfd_configuration"],
                        )
                    ),
                ),
            )
            print("[Precheck] ✓ stream object obtained — iterating chunks...", flush=True)
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                if chunk_count == 1:
                    print("[Precheck] ✓ first chunk received from LLM", flush=True)
                if not chunk.candidates:
                    continue
                for part in chunk.candidates[0].content.parts:
                    # ── Thought chunk — stream to frontend + collect for terminal
                    if getattr(part, "thought", False) and part.text:
                        yield {"type": "thought", "text": part.text}
                        _thought_buf.append(part.text)

                    # ── Function call — signal + collect ─────────────────────
                    elif getattr(part, "function_call", None) is not None:
                        if not spec_generating_sent:
                            print("[Precheck] ✓ function call received — spec_generating", flush=True)
                            yield {"type": "spec_generating"}
                            spec_generating_sent = True
                        # Overwrite each time — final chunk carries the complete args
                        func_call_args = dict(part.function_call.args)
            print(f"[Precheck] ✓ stream done — {chunk_count} chunks, func_call_args={'present' if func_call_args else 'MISSING'}", flush=True)

            # ── Print full first-pass thinking + raw args ─────────────────────
            import json as _json_dbg
            if _thought_buf:
                print("=" * 70)
                print("[Precheck] FIRST-PASS THINKING:")
                print("".join(_thought_buf))
                print("=" * 70)

            if func_call_args is not None:
                print("=" * 70)
                print("[Precheck] RAW first-pass function call args:")
                print(_json_dbg.dumps(func_call_args, indent=2, default=str))
                print("=" * 70)
                result = self._build_response_from_args(func_call_args, request)
                yield {"type": "spec", "data": result.model_dump(by_alias=True)}

                # ── Single review pass ────────────────────────────────────────
                yield {"type": "review_start"}
                print("[Precheck] → starting review...", flush=True)
                async for event in self._llm_review(request, result):
                    yield event
            else:
                yield {"type": "error", "message": "LLM did not return a configuration"}

        except Exception as e:
            logger.exception(f"[Precheck] Streaming failed: {e}")
            yield {"type": "error", "message": str(e)}
        finally:
            yield {"type": "done"}

    # ── LLM review ───────────────────────────────────────────────────────────

    async def _llm_review(self, request: PrecheckRequest, response: PrecheckResponse):
        """Single physics review: streams thinking live, then emits a summary item.

        No function-calling — plain streaming with include_thoughts so thoughts
        actually arrive incrementally (forced function-call mode suppresses thinking
        on flash models).

        Streams:
            {"type": "review_phase_start", "phase": "review", ...}
            {"type": "thought",            "text": "..."}        ← live thinking
            {"type": "review_phase_done",  "phase": "review", ...}
            {"type": "review_item",        "field": "summary", ...}  ← full text response
            {"type": "review_done"}
        """
        import json as _json

        spec_json = _json.dumps(response.model_dump(by_alias=True), indent=2, default=str)

        review_prompt = f"""\
You are a CFD expert reviewing an OpenFOAM configuration for physical correctness.

Write your assessment as plain flowing prose — like a colleague explaining the physics.
Use backticks for identifiers (`inlet`, `fixedValue`, `kOmegaSST`),
inline LaTeX for symbols ($k$, $\\omega$, $\\rho$), and display math ($$...$$)
on its own line for any formula with a numeric substitution.

=== USER REQUEST ===
{request.prompt}

=== GENERATED SPEC ===
{spec_json}

=== WHAT TO COVER ===

1. **Fluid & velocity** — identify the fluid; if a mass flow rate was given, show
   the velocity derivation:
   $$A = \\pi\\left(\\frac{{D}}{{2}}\\right)^2, \\quad U = \\frac{{\\dot{{m}}}}{{\\rho A}}$$
   State the result to 4 sig. fig.

2. **Turbulence** — narrate the derivation ($I=0.05$, $C_\\mu=0.09$, $L=0.07 D_h$):
   $$k = 1.5(U I)^2, \\quad \\omega = \\frac{{\\sqrt{{k}}}}{{C_\\mu^{{0.25}} L}}, \\quad \\varepsilon = C_\\mu^{{0.75}}\\frac{{k^{{1.5}}}}{{L}}$$
   State each result to 4 sig. fig.

3. **Boundary conditions** — one paragraph per patch; describe what each patch does
   physically and confirm the BC types are correct.

4. **Summary** — one paragraph recap, then a markdown table of the final BC values:

| Patch | Class | `U` | `p` | `T` | `k` / $\\omega$ |
|-------|-------|-----|-----|-----|----------------|
"""

        yield {
            "type": "review_phase_start",
            "phase": "review",
            "title": "Physics review",
            "description": "Reviewing boundary conditions, turbulence parameters, and fluid properties...",
        }
        print(f"[Precheck] → review (model={PRECHECK_MODEL}, no function-call so thoughts stream)...", flush=True)

        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=PRECHECK_MODEL,
                contents=review_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                ),
            )

            text_parts: list[str] = []

            async for chunk in stream:
                if not chunk.candidates:
                    continue
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "thought", False) and part.text:
                        yield {"type": "thought", "text": part.text}
                    elif getattr(part, "text", None):
                        text_parts.append(part.text)

            yield {
                "type": "review_phase_done",
                "phase": "review",
                "message": "Review complete.",
            }

            summary = "".join(text_parts).strip()
            if summary:
                yield {
                    "type": "review_item",
                    "status": "ok",
                    "patch": None,
                    "field": "summary",
                    "label": "Configuration review",
                    "detail": summary,
                }

        except Exception as e:
            logger.warning(f"[Precheck] Review failed: {e}")
            yield {
                "type": "review_item",
                "status": "warning",
                "patch": None,
                "field": "summary",
                "label": "Review unavailable",
                "detail": f"Could not complete the review: `{e}`",
            }

        yield {"type": "review_done"}

    # ── Response parser ──────────────────────────────────────────────────────

    def _parse_function_call_response(
        self, response: types.GenerateContentResponse, request: PrecheckRequest
    ) -> PrecheckResponse:
        try:
            part = response.candidates[0].content.parts[0]
            if not hasattr(part, "function_call") or part.function_call is None:
                raise ValueError("No function call in response")
            func_call = part.function_call
            if func_call.name != "submit_cfd_configuration":
                raise ValueError(f"Unexpected function: {func_call.name}")
            args = dict(func_call.args)
            logger.debug(f"[Precheck] Function args keys: {list(args.keys())}")
            return self._build_response_from_args(args, request)
        except Exception as e:
            logger.warning(f"Failed to parse function call response: {e}")
            return self._create_fallback_response(request, f"Parse error: {e}")

    def _build_response_from_args(
        self, args: dict, request: PrecheckRequest
    ) -> PrecheckResponse:
        """Build a PrecheckResponse from the LLM function-call args dict."""
        solver = SolverSettings(
            algorithm=args.get("solver_algorithm", "SIMPLE"),
            max_iterations=args.get("solver_max_iterations", 2000),
            convergence_criteria=args.get("solver_convergence_criteria", 1e-6),
            end_time=args.get("solver_end_time"),
            delta_t=args.get("solver_delta_t"),
            write_interval=args.get("solver_write_interval"),
        )

        preset_id = args.get("fluid_preset_id", "air")
        fluid = FluidProperties(
            preset_id=preset_id,
            name=args.get("fluid_name", preset_id.upper()),
            rho=args.get("fluid_rho", 1.225),
            mu=args.get("fluid_mu", 1.81e-5),
            Cp=args.get("fluid_Cp", 1006.0),
            k=args.get("fluid_k", 0.0257),
            temperature=args.get("fluid_temperature", 293.15),
        )

        flow_regime = args.get("flow_regime", "turbulent")
        turbulence = TurbulenceSettings(
            model=args.get("turbulence_model", "kOmegaSST"),
            turbulence_intensity=args.get("turbulence_intensity", 5.0),
            turbulence_length_scale=args.get("turbulence_length_scale", 0.01),
            hydraulic_diameter=args.get("hydraulic_diameter", 0.1),
            wall_functions=args.get("wall_functions", True),
        )

        boundary_conditions: dict[str, PatchBoundaryCondition] = {}
        for bc in args.get("boundary_conditions", []):
            patch_name = bc.get("patch_name", "unknown")
            boundary_conditions[patch_name] = PatchBoundaryCondition(
                patch_class=bc.get("patch_class", "wall"),
            confidence=_norm_conf(bc.get("confidence", 0.8)),
                U=FieldBC(type=bc.get("U_type", "fixedValue"), value=bc.get("U_value")),
                p=FieldBC(type=bc.get("p_type", "zeroGradient"), value=bc.get("p_value")),
                T=FieldBC(type=bc["T_type"], value=bc.get("T_value")) if bc.get("T_type") else None,
                k=FieldBC(type=bc["k_type"], value=bc.get("k_value")) if bc.get("k_type") else None,
                epsilon=FieldBC(type=bc["epsilon_type"], value=bc.get("epsilon_value")) if bc.get("epsilon_type") else None,
                omega=FieldBC(type=bc["omega_type"], value=bc.get("omega_value")) if bc.get("omega_type") else None,
                nut=FieldBC(type=bc["nut_type"], value=bc.get("nut_value")) if bc.get("nut_type") else None,
            )

        suggested_config = SuggestedConfig(
            case_type=args.get("case_type", "internal_pipe_flow"),
            flow_regime=flow_regime,
            time_scheme=args.get("time_scheme", "steady"),
            compressibility=args.get("compressibility", "incompressible"),
            enable_heat_transfer=args.get("enable_heat_transfer", False),
            gravity=args.get("gravity", False),
            solver=solver,
            fluid=fluid,
            turbulence=turbulence,
            boundary_conditions=boundary_conditions,
        )

        interpretation = Interpretation(
            summary=args.get("interpretation_summary", "CFD simulation analysis"),
            simulation_type=args.get("interpretation_simulation_type", "General CFD"),
            key_physics=args.get("interpretation_key_physics", []),
            assumptions=args.get("interpretation_assumptions", []),
            clarifications=args.get("interpretation_clarifications"),
        )

        confidence_scores = ConfidenceScores(
        overall=_norm_conf(args.get("confidence_overall", 0.8)),
        flow_regime=_norm_conf(args.get("confidence_flow_regime", 0.8)),
        boundary_conditions=_norm_conf(args.get("confidence_boundary_conditions", 0.7)),
        physics_settings=_norm_conf(args.get("confidence_physics_settings", 0.8)),
        )

        mesh = request.get_mesh()
        response = PrecheckResponse(
            success=True,
            confidence=confidence_scores.overall,
        message=args.get("message", f"Detected {flow_regime} {suggested_config.case_type}"),
            suggested_config=suggested_config,
        boundary_hints=self._build_boundary_hints(boundary_conditions),
        kpi_targets=None,
            interpretation=interpretation,
            confidence_scores=confidence_scores,
        next_step=2 if mesh else 1,
        should_show_mesh_viewer=mesh is not None,
        )

        import json
        print("=" * 70)
        print("[Precheck] Final response sent to frontend:")
        print(json.dumps(response.model_dump(by_alias=True), indent=2, default=str))
        print("=" * 70)

        return response

    # ── Helpers ──────────────────────────────────────────────────────────────
    
    def _build_boundary_hints(
        self, boundary_conditions: dict[str, PatchBoundaryCondition]
    ) -> dict[str, BoundaryHint]:
        hints: dict[str, BoundaryHint] = {}
        for patch_name, bc in boundary_conditions.items():
            hints[patch_name] = BoundaryHint(
                suggested_type=bc.patch_class if bc.patch_class != "empty" else "wall",
                velocity=VelocityBC(
                    type=bc.U.type,
                    value=bc.U.value if isinstance(bc.U.value, list) else None,
                    magnitude=bc.U.value if isinstance(bc.U.value, (int, float)) else None,
                ) if bc.U else None,
                pressure=PressureBC(
                    type=bc.p.type,
                    value=bc.p.value if isinstance(bc.p.value, (int, float)) else None,
                ) if bc.p else None,
                temperature=TemperatureBC(
                    type=bc.T.type,
                    value=bc.T.value if isinstance(bc.T.value, (int, float)) else None,
                ) if bc.T else None,
                confidence=bc.confidence,
                reasoning=f"Classified as {bc.patch_class}",
            )
        return hints
    
    # ── Fallback (LLM unavailable) ───────────────────────────────────────────
    
    def _create_friendly_error_response(self, message: str) -> PrecheckResponse:
        """Return a clean, non-technical error response shown directly to the user."""
        return PrecheckResponse(
            success=False,
            confidence=0.0,
            message=message,
            suggested_config=SuggestedConfig(
                case_type="general",
                flow_regime="turbulent",
                time_scheme="steady",
                compressibility="incompressible",
                enable_heat_transfer=False,
                gravity=False,
                solver=SolverSettings(),
                fluid=FLUID_PRESETS["air"],
                turbulence=TurbulenceSettings(model="kOmegaSST"),
                boundary_conditions={},
            ),
            interpretation=Interpretation(
                summary=message,
                simulation_type="Unknown",
                key_physics=[],
                assumptions=[],
            ),
            confidence_scores=ConfidenceScores(
                overall=0.0,
                flow_regime=0.0,
                boundary_conditions=0.0,
                physics_settings=0.0,
            ),
            errors=[message],
        )
    
    def _create_fallback_response(self, request: PrecheckRequest, error: str) -> PrecheckResponse:
        """Minimal fallback for when the LLM call completely fails.

        NOTE: The LLM handles all parsing and calculation during normal operation.
        This fallback deliberately avoids complex regex — it detects fluid type
        and cryogenic status from simple keyword matching, then returns safe defaults.
        The user is expected to review/edit the config manually.
        """
        prompt_lower = request.prompt.lower()
        mesh = request.get_mesh()
        
        # ── Fluid detection (keyword only) ───
        is_cryogenic = any(kw in prompt_lower for kw in CRYOGENIC_KEYWORDS)
        uses_lng = any(kw in prompt_lower for kw in ("lng", "liquefied natural gas", "liquid natural gas"))
        uses_helium = any(kw in prompt_lower for kw in ("helium", "lhe", "liquid helium"))
        uses_ln2 = any(kw in prompt_lower for kw in ("ln2", "liquid nitrogen", "nitrogen"))
        uses_water = any(kw in prompt_lower for kw in ("water", "hydraulic"))
        uses_oil = any(kw in prompt_lower for kw in ("oil", "lubricant"))

        if uses_lng:
            fluid = FLUID_PRESETS["lng"]
        elif uses_helium:
            fluid = FLUID_PRESETS["helium"]
        elif uses_ln2:
            fluid = FLUID_PRESETS["ln2"]
        elif uses_oil:
            fluid = FLUID_PRESETS["oil"]
        elif uses_water:
            fluid = FLUID_PRESETS["water"]
        else:
            fluid = FLUID_PRESETS["air"]

        # ── Physics flags ───
        is_high_speed = any(w in prompt_lower for w in ("mach", "supersonic", "transonic", "compressible"))
        compressibility = "compressible" if (is_cryogenic or is_high_speed) else "incompressible"
        is_transient = any(w in prompt_lower for w in ("transient", "unsteady", "pulsating", "oscillating"))
        time_scheme = "transient" if is_transient else "steady"
        flow_regime = "laminar" if "laminar" in prompt_lower else "turbulent"
        has_heat = is_cryogenic or any(w in prompt_lower for w in ("heat", "thermal", "temperature", "cooling", "heating"))

        # ── Geometry guess (for case_type label only) ───
        case_type = (
            "internal_pipe_flow" if any(w in prompt_lower for w in ("pipe", "duct", "channel", "tube"))
            else "external_aero" if any(w in prompt_lower for w in ("external", "wind", "airfoil", "wing"))
            else "general"
        )

        # ── Minimal turbulence defaults (Dh=25mm, U=1 m/s) ───
        Dh, U_ref, turb_intensity, Cmu = 0.025, 1.0, 0.05, 0.09
        L = 0.07 * Dh
        k0 = 1.5 * (U_ref * turb_intensity) ** 2
        omega0 = math.sqrt(k0) / (Cmu ** 0.25 * L)
        epsilon0 = Cmu ** 0.75 * k0 ** 1.5 / L

        outlet_p = 101325.0 if compressibility == "compressible" else 0.0

        # ── Boundary conditions from mesh patches ───
        boundary_conditions: dict[str, PatchBoundaryCondition] = {}
        if mesh:
            for patch in mesh.patches:
                pl = patch.name.lower().replace("_", "")
                turb = flow_regime == "turbulent"
                empty_bc = PatchBoundaryCondition(
                    patch_class="empty", confidence=1.0,
                    U=FieldBC(type="empty"), p=FieldBC(type="empty"),
                    T=FieldBC(type="empty") if has_heat else None,
                    k=FieldBC(type="empty") if turb else None,
                    omega=FieldBC(type="empty") if turb else None,
                    epsilon=FieldBC(type="empty") if turb else None,
                    nut=FieldBC(type="empty") if turb else None,
                )
                if pl in ("frontandback", "frontback", "defaultfaces") or patch.type == "empty":
                    boundary_conditions[patch.name] = empty_bc
                elif any(x in pl for x in ("inlet", "inflow")):
                    boundary_conditions[patch.name] = PatchBoundaryCondition(
                        patch_class="inlet", confidence=0.7,
                        U=FieldBC(type="fixedValue", value=[U_ref, 0.0, 0.0]),
                        p=FieldBC(type="zeroGradient"),
                        T=FieldBC(type="fixedValue", value=fluid.temperature) if has_heat else None,
                        k=FieldBC(type="fixedValue", value=k0) if turb else None,
                        omega=FieldBC(type="fixedValue", value=omega0) if turb else None,
                        epsilon=FieldBC(type="fixedValue", value=epsilon0) if turb else None,
                        nut=FieldBC(type="calculated", value=0) if turb else None,
                    )
                elif any(x in pl for x in ("outlet", "outflow", "exit")):
                    boundary_conditions[patch.name] = PatchBoundaryCondition(
                        patch_class="outlet", confidence=0.7,
                        U=FieldBC(type="zeroGradient"),
                        p=FieldBC(type="fixedValue", value=outlet_p),
                        T=FieldBC(type="zeroGradient") if has_heat else None,
                        k=FieldBC(type="zeroGradient") if turb else None,
                        omega=FieldBC(type="zeroGradient") if turb else None,
                        epsilon=FieldBC(type="zeroGradient") if turb else None,
                        nut=FieldBC(type="calculated", value=0) if turb else None,
                    )
                elif any(x in pl for x in ("sym", "symmetry")) and patch.type in ("symmetry", "symmetryPlane"):
                    boundary_conditions[patch.name] = PatchBoundaryCondition(
                        patch_class="symmetry", confidence=0.9,
                        U=FieldBC(type="symmetry"), p=FieldBC(type="symmetry"),
                        T=FieldBC(type="symmetry") if has_heat else None,
                        k=FieldBC(type="symmetry") if turb else None,
                        omega=FieldBC(type="symmetry") if turb else None,
                        epsilon=FieldBC(type="symmetry") if turb else None,
                        nut=FieldBC(type="symmetry") if turb else None,
                    )
                else:
                    boundary_conditions[patch.name] = PatchBoundaryCondition(
                        patch_class="wall", confidence=0.6,
                        U=FieldBC(type="noSlip"), p=FieldBC(type="zeroGradient"),
                        T=FieldBC(type="fixedValue", value=300.0) if has_heat else None,
                        k=FieldBC(type="kqRWallFunction", value=0) if turb else None,
                        omega=FieldBC(type="omegaWallFunction", value=0) if turb else None,
                        epsilon=FieldBC(type="epsilonWallFunction", value=0) if turb else None,
                        nut=FieldBC(type="nutkWallFunction", value=0) if turb else None,
        )

        suggested_config = SuggestedConfig(
            case_type=case_type,
            flow_regime=flow_regime,
            time_scheme=time_scheme,
            compressibility=compressibility,
            enable_heat_transfer=has_heat,
            gravity=False,
            solver=SolverSettings(
                algorithm="SIMPLE" if not is_transient else "PIMPLE",
                max_iterations=2000 if not is_transient else 100,
                end_time=1.0 if is_transient else None,
                delta_t=0.001 if is_transient else None,
                write_interval=0.1 if is_transient else None,
            ),
            fluid=fluid,
            turbulence=TurbulenceSettings(
                model="kOmegaSST" if flow_regime == "turbulent" else "laminar",
                turbulence_intensity=5.0,
                turbulence_length_scale=L,
                hydraulic_diameter=Dh,
                wall_functions=True,
            ),
            boundary_conditions=boundary_conditions,
        )

        return PrecheckResponse(
            success=False,
            confidence=0.3,
            message="Fallback defaults — LLM unavailable. Please review carefully.",
            suggested_config=suggested_config,
            boundary_hints=self._build_boundary_hints(boundary_conditions),
            interpretation=Interpretation(
                summary=f"Fallback (LLM error: {error})",
                simulation_type=case_type.replace("_", " ").title(),
                key_physics=["turbulence"] if flow_regime == "turbulent" else [],
                assumptions=["All values are conservative defaults — review before running"],
                clarifications=["LLM unavailable; numerical values (velocity, k, ω) are placeholders"],
            ),
            confidence_scores=ConfidenceScores(overall=0.3, flow_regime=0.4,
                                               boundary_conditions=0.3, physics_settings=0.4),
            next_step=1,
            should_show_mesh_viewer=mesh is not None,
            warnings=[f"LLM failed, using minimal fallback: {error}"],
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_precheck_service: PrecheckService | None = None


def get_precheck_service() -> PrecheckService:
    global _precheck_service
    if _precheck_service is None:
        _precheck_service = PrecheckService()
    return _precheck_service
