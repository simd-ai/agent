# tests/test_config_normalizer.py
"""Tests for configuration normalizer and BC validation."""

import pytest
from simd_agent.config_normalizer import (
    normalize_config,
    validate_config_completeness,
    detect_format,
    camel_to_snake,
    normalize_mesh,
    normalize_boundary_conditions,
    get_config_summary,
)
from simd_agent.models import (
    BoundaryType,
    FlowRegime,
    SimulationConfigV1,
)


class TestCamelToSnake:
    """Test camelCase to snake_case conversion."""
    
    def test_simple_camel(self):
        assert camel_to_snake("flowRegime") == "flow_regime"
    
    def test_multiple_words(self):
        assert camel_to_snake("maxIterations") == "max_iterations"
    
    def test_consecutive_caps(self):
        assert camel_to_snake("XMLParser") == "xml_parser"
    
    def test_already_snake(self):
        assert camel_to_snake("flow_regime") == "flow_regime"
    
    def test_single_word(self):
        assert camel_to_snake("mesh") == "mesh"


class TestDetectFormat:
    """Test format detection."""
    
    def test_empty_config(self):
        assert detect_format({}) == "empty"
    
    def test_v1_full_format(self):
        config = {
            "mesh": {"mesh_id": "abc123", "patches": []},
            "physics": {"flow_regime": "turbulent"},
            "solver": {"type": "simpleFoam"},
            "fluid": {"name": "air"},
            "boundary_conditions": {},
        }
        assert detect_format(config) == "v1_full"
    
    def test_legacy_camel_format(self):
        config = {
            "meshId": "abc123",
            "flowRegime": "turbulent",
            "boundaryConditions": {},
        }
        assert detect_format(config) == "legacy_camel"
    
    def test_legacy_minimal_mesh_string(self):
        config = {
            "mesh": "mesh-abc123",
        }
        assert detect_format(config) == "legacy_minimal"
    
    def test_legacy_minimal_mesh_id(self):
        config = {
            "mesh_id": "abc123",
        }
        assert detect_format(config) == "legacy_minimal"


class TestNormalizeMesh:
    """Test mesh normalization."""
    
    def test_string_mesh_id(self):
        mesh = normalize_mesh("mesh-abc123")
        assert mesh is not None
        assert mesh.mesh_id == "mesh-abc123"
        assert mesh.patches == []
    
    def test_dict_mesh_id(self):
        mesh = normalize_mesh({"mesh_id": "abc123"})
        assert mesh is not None
        assert mesh.mesh_id == "abc123"
    
    def test_camel_case_mesh_id(self):
        mesh = normalize_mesh({"meshId": "abc123"})
        assert mesh is not None
        assert mesh.mesh_id == "abc123"
    
    def test_full_mesh_info(self):
        mesh = normalize_mesh({
            "mesh_id": "abc123",
            "file_name": "pipe.stl",
            "patches": [
                {"name": "inlet", "type": "patch", "n_faces": 100},
                {"name": "outlet", "type": "patch", "n_faces": 100},
                {"name": "wall", "type": "wall", "n_faces": 5000},
            ],
            "check_mesh": {
                "cells": 50000,
                "faces": 150000,
                "points": 52000,
            },
        })
        assert mesh is not None
        assert mesh.mesh_id == "abc123"
        assert len(mesh.patches) == 3
        assert mesh.check_mesh is not None
        assert mesh.check_mesh.cells == 50000
    
    def test_none_mesh(self):
        assert normalize_mesh(None) is None


class TestNormalizeBoundaryConditions:
    """Test boundary condition normalization."""
    
    def test_empty_bcs(self):
        bcs = normalize_boundary_conditions(None, {})
        assert bcs == {}
    
    def test_simple_inlet_outlet(self):
        bcs = normalize_boundary_conditions({
            "inlet": {
                "patch_type": "inlet",
                "velocity": {"type": "fixedValue", "value": [5, 0, 0]},
            },
            "outlet": {
                "patch_type": "outlet",
                "pressure": {"type": "fixedValue", "value": 0},
            },
        }, {})
        
        assert "inlet" in bcs
        assert "outlet" in bcs
        assert bcs["inlet"].is_inlet()
        assert bcs["outlet"].is_outlet()
        assert bcs["inlet"].velocity is not None
        assert bcs["inlet"].velocity.get_velocity_vector() == [5, 0, 0]
    
    def test_legacy_top_level_inlet(self):
        """Test that legacy inlet/outlet at top level are converted."""
        bcs = normalize_boundary_conditions(None, {
            "inlet": {"velocity": [3, 0, 0]},
            "outlet": {"pressure": 0},
        })
        
        assert "inlet" in bcs
        assert "outlet" in bcs
    
    def test_camel_case_keys(self):
        bcs = normalize_boundary_conditions({
            "inlet": {
                "patchType": "inlet",
                "velocity": {"type": "fixedValue", "value": [1, 0, 0]},
            },
        }, {})
        
        assert bcs["inlet"].is_inlet()
    
    def test_velocity_magnitude(self):
        bcs = normalize_boundary_conditions({
            "inlet": {
                "patch_type": "inlet",
                "velocity": {"type": "fixedValue", "magnitude": 5, "direction": [1, 0, 0]},
            },
        }, {})
        
        assert bcs["inlet"].velocity is not None
        assert bcs["inlet"].velocity.get_magnitude() == 5


class TestNormalizeConfig:
    """Test full config normalization."""
    
    def test_minimal_legacy_payload(self):
        """Test that old minimal payload still parses."""
        raw = {
            "mesh": "mesh-abc123",
        }
        config, fmt, transforms = normalize_config(raw)
        
        assert fmt == "legacy_minimal"
        assert config.mesh is not None
        assert config.mesh.mesh_id == "mesh-abc123"
    
    def test_full_v1_payload(self):
        """Test full V1 payload parses correctly."""
        raw = {
            "mesh": {
                "mesh_id": "abc123",
                "patches": [
                    {"name": "inlet", "type": "patch", "n_faces": 100},
                    {"name": "outlet", "type": "patch", "n_faces": 100},
                    {"name": "wall", "type": "wall", "n_faces": 5000},
                ],
                "check_mesh": {"cells": 50000, "faces": 150000, "points": 52000},
            },
            "physics": {
                "flow_regime": "turbulent",
                "time_scheme": "steady",
                "turbulence_model": "kEpsilon",
            },
            "solver": {
                "type": "simpleFoam",
                "max_iterations": 2000,
            },
            "fluid": {
                "name": "water",
                "density": 1000,
                "kinematic_viscosity": 1e-6,
            },
            "geometry": {
                "type": "pipe",
                "diameter": 0.1,
                "length": 1.0,
            },
            "boundary_conditions": {
                "inlet": {
                    "patch_type": "inlet",
                    "velocity": {"type": "fixedValue", "value": [5, 0, 0]},
                },
                "outlet": {
                    "patch_type": "outlet",
                    "pressure": {"type": "fixedValue", "value": 0},
                },
                "wall": {
                    "patch_type": "wall",
                },
            },
        }
        
        config, fmt, transforms = normalize_config(raw)
        
        assert fmt == "v1_full"
        assert config.mesh is not None
        assert config.mesh.mesh_id == "abc123"
        assert len(config.mesh.patches) == 3
        assert config.physics.flow_regime == FlowRegime.TURBULENT
        assert config.solver.type == "simpleFoam"
        assert config.solver.max_iterations == 2000
        assert config.fluid.name == "water"
        assert config.fluid.density == 1000
        assert config.geometry is not None
        assert config.geometry.diameter == 0.1
        assert len(config.boundary_conditions) == 3
        assert config.boundary_conditions["inlet"].is_inlet()
    
    def test_camel_case_payload(self):
        """Test camelCase payload converts correctly."""
        raw = {
            "meshId": "abc123",
            "flowRegime": "turbulent",
            "timeScheme": "steady",
            "maxIterations": 1500,
            "boundaryConditions": {
                "inlet": {
                    "patchType": "inlet",
                    "velocity": {"type": "fixedValue", "value": [2, 0, 0]},
                },
            },
        }
        
        config, fmt, transforms = normalize_config(raw)
        
        assert "converted_camel_to_snake" in transforms
        assert config.physics.flow_regime == FlowRegime.TURBULENT
        assert config.solver.max_iterations == 1500


class TestValidateCompleteness:
    """Test config completeness validation."""
    
    def test_incomplete_missing_mesh(self):
        """Config without mesh is incomplete."""
        config = SimulationConfigV1()
        result = validate_config_completeness(config, "CFD_CODEGEN_RUN")
        
        assert not result.is_complete
        assert any(m.field == "mesh" for m in result.missing_fields)
    
    def test_incomplete_missing_inlet(self):
        """Config without inlet BC is incomplete."""
        raw = {
            "mesh": {"mesh_id": "abc123"},
            "boundary_conditions": {
                "outlet": {"patch_type": "outlet", "pressure": {"value": 0}},
            },
        }
        config, _, _ = normalize_config(raw)
        result = validate_config_completeness(config, "CFD_CODEGEN_RUN")
        
        assert not result.is_complete
        assert any("inlet" in m.field for m in result.missing_fields)
    
    def test_incomplete_missing_outlet(self):
        """Config without outlet BC is incomplete."""
        raw = {
            "mesh": {"mesh_id": "abc123"},
            "boundary_conditions": {
                "inlet": {
                    "patch_type": "inlet",
                    "velocity": {"value": [1, 0, 0]},
                },
            },
        }
        config, _, _ = normalize_config(raw)
        result = validate_config_completeness(config, "CFD_CODEGEN_RUN")
        
        assert not result.is_complete
        assert any("outlet" in m.field for m in result.missing_fields)
    
    def test_incomplete_inlet_without_velocity(self):
        """Inlet without velocity is incomplete."""
        raw = {
            "mesh": {"mesh_id": "abc123"},
            "boundary_conditions": {
                "inlet": {"patch_type": "inlet"},  # No velocity!
                "outlet": {"patch_type": "outlet", "pressure": {"value": 0}},
            },
        }
        config, _, _ = normalize_config(raw)
        result = validate_config_completeness(config, "CFD_CODEGEN_RUN")
        
        assert not result.is_complete
        assert any("velocity" in m.field for m in result.missing_fields)
    
    def test_complete_config(self):
        """Complete config passes validation."""
        raw = {
            "mesh": {"mesh_id": "abc123"},
            "boundary_conditions": {
                "inlet": {
                    "patch_type": "inlet",
                    "velocity": {"type": "fixedValue", "value": [5, 0, 0]},
                },
                "outlet": {
                    "patch_type": "outlet",
                    "pressure": {"type": "fixedValue", "value": 0},
                },
                "wall": {"patch_type": "wall"},
            },
        }
        config, _, _ = normalize_config(raw)
        result = validate_config_completeness(config, "CFD_CODEGEN_RUN")
        
        assert result.is_complete
        assert len(result.missing_fields) == 0
    
    def test_lint_operation_more_lenient(self):
        """CFD_LINT operation should be more lenient."""
        raw = {"mesh": "abc123"}  # Minimal config
        config, _, _ = normalize_config(raw)
        result = validate_config_completeness(config, "CFD_LINT")
        
        # Lint should still report missing fields but is_valid should be True
        assert result.is_valid  # Lint is lenient


class TestReynoldsCalculation:
    """Test Reynolds number calculation from normalized config."""
    
    def test_reynolds_from_full_config(self):
        """Test Reynolds calculation with complete data."""
        raw = {
            "mesh": {"mesh_id": "abc123"},
            "geometry": {"diameter": 0.1},  # 10cm pipe
            "fluid": {"kinematic_viscosity": 1e-6},  # water
            "boundary_conditions": {
                "inlet": {
                    "patch_type": "inlet",
                    "velocity": {"value": [1, 0, 0]},  # 1 m/s
                },
                "outlet": {"patch_type": "outlet"},
            },
        }
        config, _, _ = normalize_config(raw)
        
        # Re = U * D / nu = 1 * 0.1 / 1e-6 = 100,000
        velocity = config.get_inlet_velocity_magnitude()
        length = config.get_characteristic_length()
        nu = config.get_kinematic_viscosity()
        
        assert velocity == pytest.approx(1.0)
        assert length == pytest.approx(0.1)
        assert nu == pytest.approx(1e-6)
        
        re = (velocity * length) / nu
        assert re == pytest.approx(100000)


class TestConfigSummary:
    """Test config summary generation."""
    
    def test_summary_contains_key_info(self):
        raw = {
            "mesh": {"mesh_id": "abc123", "patches": [{"name": "inlet", "type": "patch"}]},
            "boundary_conditions": {
                "inlet": {"patch_type": "inlet", "velocity": {"value": [1, 0, 0]}},
            },
        }
        config, _, _ = normalize_config(raw)
        summary = get_config_summary(config)
        
        assert summary["has_mesh"] is True
        assert summary["mesh_id"] == "abc123"
        assert "inlet" in summary["mesh_patches"]
        assert "inlet" in summary["boundary_conditions"]
        assert summary["boundary_conditions"]["inlet"]["type"] == "inlet"


class TestBoundaryTypeInference:
    """Test that boundary types are correctly inferred from patch names."""
    
    def test_infer_inlet_from_name(self):
        raw = {
            "mesh": {
                "mesh_id": "abc",
                "patches": [{"name": "inlet_pipe", "type": "patch"}]
            },
            "boundary_conditions": {
                "inlet_pipe": {"velocity": {"value": [1, 0, 0]}},
            },
        }
        config, _, _ = normalize_config(raw)
        
        assert config.boundary_conditions["inlet_pipe"].is_inlet()
    
    def test_infer_outlet_from_name(self):
        raw = {
            "mesh": {"mesh_id": "abc"},
            "boundary_conditions": {
                "outlet_main": {"pressure": {"value": 0}},
            },
        }
        config, _, _ = normalize_config(raw)
        
        assert config.boundary_conditions["outlet_main"].is_outlet()
    
    def test_infer_wall_from_name(self):
        raw = {
            "mesh": {"mesh_id": "abc"},
            "boundary_conditions": {
                "walls": {},
            },
        }
        config, _, _ = normalize_config(raw)
        
        assert config.boundary_conditions["walls"].is_wall()
