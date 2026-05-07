# tests/test_packaging.py
"""Tests for OpenFOAM case packaging."""

import io
import zipfile

import pytest

from simd_agent.packaging import (
    extract_file_blocks,
    generate_run_script,
    package_case,
    package_from_llm_output,
    validate_openfoam_structure,
    merge_files,
    PackagingError,
)


class TestExtractFileBlocks:
    """Tests for file block extraction from LLM output."""
    
    def test_extract_single_file(self):
        """Test extracting a single file block."""
        llm_output = '''Here is the file:

```file:system/controlDict
FoamFile
{
    version 2.0;
}
application simpleFoam;
```
'''
        
        files = extract_file_blocks(llm_output)
        
        assert len(files) == 1
        assert "system/controlDict" in files
        assert "simpleFoam" in files["system/controlDict"]
    
    def test_extract_multiple_files(self):
        """Test extracting multiple file blocks."""
        llm_output = '''
```file:system/controlDict
content1
```

Some explanation text here.

```file:system/fvSchemes
content2
```

```file:0/U
content3
```
'''
        
        files = extract_file_blocks(llm_output)
        
        assert len(files) == 3
        assert "system/controlDict" in files
        assert "system/fvSchemes" in files
        assert "0/U" in files
    
    def test_extract_with_nested_content(self):
        """Test that file content with braces is preserved."""
        llm_output = '''```file:system/fvSolution
FoamFile { version 2.0; }

solvers
{
    p
    {
        solver GAMG;
    }
}
```
'''
        
        files = extract_file_blocks(llm_output)
        
        assert "system/fvSolution" in files
        content = files["system/fvSolution"]
        assert "solvers" in content
        assert "GAMG" in content
    
    def test_empty_output_returns_empty(self):
        """Test that empty output returns empty dict."""
        files = extract_file_blocks("")
        assert files == {}
    
    def test_no_file_blocks_returns_empty(self):
        """Test that text without file blocks returns empty."""
        llm_output = '''This is just some text.
No file blocks here.
'''
        
        files = extract_file_blocks(llm_output)
        assert files == {}


class TestValidateOpenFoamStructure:
    """Tests for OpenFOAM case structure validation."""
    
    def test_complete_structure_no_warnings(self):
        """Test that complete structure has no warnings."""
        files = {
            "system/controlDict": "content",
            "system/fvSchemes": "content",
            "system/fvSolution": "content",
            "system/blockMeshDict": "content",
            "0/U": "content",
            "0/p": "content",
        }
        
        warnings = validate_openfoam_structure(files)
        
        assert len(warnings) == 0
    
    def test_missing_controldict_warning(self):
        """Test warning for missing controlDict."""
        files = {
            "system/fvSchemes": "content",
            "system/fvSolution": "content",
            "0/U": "content",
        }
        
        warnings = validate_openfoam_structure(files)
        
        assert any("controlDict" in w for w in warnings)
    
    def test_missing_initial_conditions_warning(self):
        """Test warning for missing initial conditions."""
        files = {
            "system/controlDict": "content",
            "system/fvSchemes": "content",
            "system/fvSolution": "content",
            # No 0/ directory
        }
        
        warnings = validate_openfoam_structure(files)
        
        assert any("initial" in w.lower() for w in warnings)
    


class TestGenerateRunScript:
    """Tests for run script generation."""
    
    def test_default_script(self):
        """Test default run script generation."""
        script = generate_run_script()

        assert "#!/bin/bash" in script
        assert "simpleFoam" in script
    
    def test_custom_solver(self):
        """Test run script with custom solver."""
        script = generate_run_script(solver="pimpleFoam")
        
        assert "pimpleFoam" in script
    
    def test_script_is_executable_format(self):
        """Test that script has bash shebang."""
        script = generate_run_script()
        
        assert script.startswith("#!/bin/bash")


class TestPackageCase:
    """Tests for case packaging to zip."""
    
    def test_package_creates_zip(self):
        """Test that packaging creates a valid zip."""
        files = {
            "system/controlDict": "test content",
            "0/U": "velocity field",
        }
        
        zip_bytes, file_list = package_case(files)
        
        assert len(zip_bytes) > 0
        
        # Verify it's a valid zip
        zip_buffer = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            names = zf.namelist()
            assert any("controlDict" in n for n in names)
            assert any("run.sh" in n for n in names)
    
    def test_package_includes_run_script(self):
        """Test that run.sh is included by default."""
        files = {"system/controlDict": "content"}
        
        zip_bytes, file_list = package_case(files, include_run_script=True)
        
        assert any("run.sh" in f for f in file_list)
    
    def test_package_custom_case_name(self):
        """Test custom case name in zip."""
        files = {"system/controlDict": "content"}
        
        zip_bytes, file_list = package_case(files, case_name="my_case")
        
        assert all(f.startswith("my_case/") for f in file_list)
    
    def test_package_file_content_preserved(self):
        """Test that file content is preserved in zip."""
        content = "application simpleFoam;\nendTime 1000;"
        files = {"system/controlDict": content}
        
        zip_bytes, _ = package_case(files)
        
        zip_buffer = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            for name in zf.namelist():
                if "controlDict" in name:
                    extracted = zf.read(name).decode("utf-8")
                    assert "simpleFoam" in extracted


class TestPackageFromLLMOutput:
    """Tests for packaging directly from LLM output."""
    
    def test_package_from_llm_output(self):
        """Test end-to-end packaging from LLM output."""
        llm_output = '''```file:system/controlDict
application simpleFoam;
```

```file:system/fvSchemes
ddtSchemes { default steadyState; }
```

```file:system/fvSolution
solvers { }
```

```file:system/blockMeshDict
vertices ( );
```

```file:0/U
internalField uniform (0 0 0);
```
'''
        
        zip_bytes, file_list, warnings = package_from_llm_output(llm_output)
        
        assert len(zip_bytes) > 0
        assert len(file_list) > 0
    
    def test_package_from_empty_output_raises(self):
        """Test that empty LLM output raises error."""
        with pytest.raises(PackagingError):
            package_from_llm_output("")
    
    def test_package_from_no_files_raises(self):
        """Test that output with no files raises error."""
        llm_output = "Here is some explanation without any file blocks."
        
        with pytest.raises(PackagingError):
            package_from_llm_output(llm_output)


class TestMergeFiles:
    """Tests for file merging."""
    
    def test_merge_adds_new_files(self):
        """Test that merge adds new files."""
        base = {"file1": "content1"}
        patch = {"file2": "content2"}
        
        result = merge_files(base, patch)
        
        assert "file1" in result
        assert "file2" in result
    
    def test_merge_overwrites_existing(self):
        """Test that merge overwrites existing files."""
        base = {"file1": "old content"}
        patch = {"file1": "new content"}
        
        result = merge_files(base, patch)
        
        assert result["file1"] == "new content"
    
    def test_merge_preserves_unpatched(self):
        """Test that merge preserves files not in patch."""
        base = {"file1": "content1", "file2": "content2"}
        patch = {"file1": "updated"}
        
        result = merge_files(base, patch)
        
        assert result["file2"] == "content2"
