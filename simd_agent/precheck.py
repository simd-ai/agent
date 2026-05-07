# simd_agent/precheck.py — backward-compat shim; real code lives in simd_agent/precheck/
from simd_agent.precheck.service import PrecheckService, get_precheck_service  # noqa: F401
from simd_agent.precheck.models import *  # noqa: F401, F403
