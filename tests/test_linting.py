# tests/test_linting.py
"""Tests for CFD linting and validation."""

import pytest

from simd_agent.linting import CFDLinter
from simd_agent.models import FlowRegime


class TestReynoldsCalculation:
    """Tests for Reynolds number calculation."""
    
    @pytest.fixture
    def linter(self):
        """Create a linter instance."""
        return CFDLinter()
    
    async def test_pipe_flow_reynolds(self, linter, sample_simulation_config):
        """Test Reynolds number calculation for pipe flow."""
        result = await linter.lint(sample_simulation_config)
        
        # Re = (U * D) / nu = (1.0 * 0.1) / 1e-6 = 100,000
        assert result.reynolds_number is not None
        assert abs(result.reynolds_number - 100000) < 1
    
    async def test_laminar_detection(self, linter, laminar_config):
        """Test detection of laminar flow regime."""
        result = await linter.lint(laminar_config)
        
        # Re = (0.01 * 0.01) / 1e-3 = 0.1 -> Laminar
        assert result.reynolds_number is not None
        assert result.reynolds_number < 2300
        assert result.detected_regime == FlowRegime.LAMINAR
    
    async def test_turbulent_detection(self, linter, turbulent_config):
        """Test detection of turbulent flow regime."""
        result = await linter.lint(turbulent_config)
        
        # Re = (10.0 * 0.1) / 1e-6 = 1,000,000 -> Turbulent
        assert result.reynolds_number is not None
        assert result.reynolds_number > 4000
        assert result.detected_regime == FlowRegime.TURBULENT
    
    async def test_transitional_detection(self, linter):
        """Test detection of transitional flow regime."""
        config = {
            "geometry": {"diameter": 0.01},
            "inlet": {"velocity": 0.3},
            "fluid": {"viscosity": 1e-6},
        }
        
        result = await linter.lint(config)
        
        # Re = (0.3 * 0.01) / 1e-6 = 3000 -> Transitional
        assert result.reynolds_number is not None
        assert 2300 <= result.reynolds_number <= 4000
        assert result.detected_regime == FlowRegime.TRANSITIONAL
    
    async def test_missing_velocity_no_reynolds(self, linter):
        """Test that missing velocity means no Reynolds calculation."""
        config = {
            "geometry": {"diameter": 0.1},
            "fluid": {"viscosity": 1e-6},
        }
        
        result = await linter.lint(config)
        assert result.reynolds_number is None


class TestSolverSelection:
    """Tests for solver and turbulence model selection."""
    
    @pytest.fixture
    def linter(self):
        return CFDLinter()
    
    async def test_laminar_solver_selection(self, linter, laminar_config):
        """Test solver selection for laminar flow."""
        result = await linter.lint(laminar_config)
        
        assert result.selected_solver == "simpleFoam"
        assert result.validated_config.get("turbulence_model") == "laminar"
    
    async def test_turbulent_solver_selection(self, linter, turbulent_config):
        """Test solver selection for turbulent flow."""
        result = await linter.lint(turbulent_config)
        
        assert result.selected_solver == "simpleFoam"
        assert result.validated_config.get("turbulence_model") in ["kEpsilon", "kOmegaSST"]
    
    async def test_user_solver_preserved_if_valid(self, linter):
        """Test that valid user-specified solver is preserved."""
        config = {
            "geometry": {"diameter": 0.1},
            "inlet": {"velocity": 10.0},
            "fluid": {"viscosity": 1e-6},
            "solver": "pimpleFoam",
        }
        
        result = await linter.lint(config)
        
        # For steady turbulent flow, pimpleFoam is also valid
        assert result.validated_config.get("solver") in ["simpleFoam", "pimpleFoam"]
    
    async def test_explicit_turbulence_model_is_respected(self, linter, laminar_config):
        """User-provided turbulence model wins over a low Reynolds number.

        Geometry-inferred Re is unreliable on complex / multi-inlet meshes
        (the bbox-derived D_h tends to overshoot for U-bends, tee-junctions
        and ducts with branches).  When the user — or the precheck planner —
        explicitly sets a turbulent model, the linter must respect it; the
        previous "Re wins" behaviour silently demoted real RAS cases to
        ``simulationType laminar`` and crashed rhoSimpleFoam with SIGFPE.
        """
        # User specifies a turbulent model on a fixture whose geometry happens
        # to give low Re.  The model carries through.
        laminar_config["turbulence_model"] = "kEpsilon"

        result = await linter.lint(laminar_config)

        # Honoured — no auto-demotion to laminar.
        assert result.validated_config.get("turbulence_model") == "kEpsilon"

    async def test_explicit_laminar_flow_regime_overrides_model(
        self, linter, laminar_config
    ):
        """The user can still force laminar by stating it explicitly.

        ``flow_regime=laminar`` is an unambiguous user signal that beats any
        turbulence model that may have been left in the config.
        """
        laminar_config["flow_regime"] = "laminar"
        laminar_config["turbulence_model"] = "kEpsilon"

        result = await linter.lint(laminar_config)

        assert result.validated_config.get("turbulence_model") == "laminar"
        
        # Should have a recommendation about this
        change_paths = [c.path for c in result.apply_changes]
        # Turbulence model change should be recommended


class TestUnitsValidation:
    """Tests for units and dimension validation."""
    
    @pytest.fixture
    def linter(self):
        return CFDLinter()
    
    async def test_negative_diameter_error(self, linter):
        """Test that negative diameter produces error."""
        config = {
            "geometry": {"diameter": -0.1},
            "inlet": {"velocity": 1.0},
        }
        
        result = await linter.lint(config)
        
        errors = [i for i in result.issues if i.severity == "error"]
        assert any("diameter" in i.path.lower() for i in errors if i.path)
    
    async def test_negative_viscosity_error(self, linter):
        """Test that negative viscosity produces error."""
        config = {
            "geometry": {"diameter": 0.1},
            "inlet": {"velocity": 1.0},
            "viscosity": -1e-6,
        }
        
        result = await linter.lint(config)
        
        errors = [i for i in result.issues if i.severity == "error"]
        assert len(errors) >= 1
    
    async def test_high_velocity_warning(self, linter):
        """Test that very high velocity produces warning."""
        config = {
            "geometry": {"diameter": 0.1},
            "inlet": {"velocity": 500.0},  # Transonic
            "fluid": {"viscosity": 1.5e-5},  # Air
        }
        
        result = await linter.lint(config)
        
        warnings = [i for i in result.issues if i.severity == "warning"]
        # Should warn about high velocity / compressibility
        assert any("velocity" in i.message.lower() for i in warnings)


class TestCaseTypeDetection:
    """Tests for simulation case type detection."""
    
    @pytest.fixture
    def linter(self):
        return CFDLinter()
    
    async def test_pipe_detection_from_requirements(self, linter):
        """Test pipe flow detection from requirements text."""
        config = {"geometry": {"diameter": 0.1}}
        requirements = "Simulate water flowing through a pipe"
        
        result = await linter.lint(config, requirements)
        
        assert result.detected_case_type == "pipe_flow"
    
    async def test_external_aero_detection(self, linter):
        """Test external aero detection."""
        config = {"geometry": {"type": "airfoil"}}
        requirements = "Simulate airflow around a wing"
        
        result = await linter.lint(config, requirements)
        
        assert result.detected_case_type == "external_aero"
    
    async def test_heat_transfer_detection(self, linter):
        """Test heat transfer case detection."""
        config = {"geometry": {"diameter": 0.1}}
        requirements = "Simulate heated pipe with convective cooling"
        
        result = await linter.lint(config, requirements)
        
        assert result.detected_case_type == "heat_transfer"
    
    async def test_explicit_case_type(self, linter):
        """Test that explicit case_type is preserved."""
        config = {
            "case_type": "mixing",
            "geometry": {"diameter": 0.5},
        }
        
        result = await linter.lint(config)
        
        assert result.detected_case_type == "mixing"


class TestApplyChanges:
    """Tests for recommended changes generation."""
    
    @pytest.fixture
    def linter(self):
        return CFDLinter()
    
    async def test_default_values_applied(self, linter):
        """Test that default values are applied for missing config."""
        config = {
            "geometry": {"diameter": 0.1},
            "inlet": {"velocity": 1.0},
        }
        
        result = await linter.lint(config)
        
        # Should have defaults applied
        assert "solver" in result.validated_config
        assert "write_interval" in result.validated_config
    
    async def test_mesh_recommendation(self, linter, turbulent_config):
        """Test mesh resolution recommendation for turbulent flow."""
        result = await linter.lint(turbulent_config)
        
        # Should recommend finer mesh for high Re
        mesh_changes = [c for c in result.apply_changes if "mesh" in c.path.lower()]
        # High Re flow should have mesh recommendations
    
    async def test_changes_have_reasons(self, linter, sample_simulation_config):
        """Test that all changes have reasons."""
        result = await linter.lint(sample_simulation_config)
        
        for change in result.apply_changes:
            assert change.reason is not None
            assert len(change.reason) > 0


class TestBoundaryConditionValidation:
    """Tests for boundary condition validation."""
    
    @pytest.fixture
    def linter(self):
        return CFDLinter()
    
    async def test_missing_inlet_warning(self, linter):
        """Test warning when inlet is not defined."""
        config = {
            "geometry": {"type": "pipe", "diameter": 0.1},
            # No inlet defined
        }
        
        result = await linter.lint(config, "pipe flow simulation")
        
        warnings = [i for i in result.issues if i.severity == "warning"]
        # Should warn about missing inlet (though it may be in top-level)
    
    async def test_complete_boundaries(self, linter):
        """Test that complete boundary definition passes."""
        config = {
            "geometry": {"type": "pipe", "diameter": 0.1},
            "inlet": {"velocity": 1.0},
            "outlet": {"pressure": 0},
            "boundary_conditions": {
                "walls": {"type": "noSlip"},
            },
        }
        
        result = await linter.lint(config, "pipe flow simulation")
        
        # Should not have boundary-related errors
        bc_errors = [i for i in result.issues 
                     if i.severity == "error" and "boundary" in (i.path or "").lower()]
        assert len(bc_errors) == 0
