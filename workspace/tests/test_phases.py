"""Tests for phase modules (requirement, develop, fix, test, deliver)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestPhaseImports:
    """Test that all phase modules can be imported."""

    def test_import_requirement(self):
        """Test importing requirement phase."""
        from phases.requirement import requirement_phase

        assert callable(requirement_phase)

    def test_import_develop(self):
        """Test importing develop phase."""
        from phases.develop import develop_phase

        assert callable(develop_phase)

    def test_import_fix(self):
        """Test importing fix phase."""
        from phases.fix import fix_phase

        assert callable(fix_phase)

    def test_import_test_phases(self):
        """Test importing test phases."""
        from phases.test import light_test_phase, full_test_phase

        assert callable(light_test_phase)
        assert callable(full_test_phase)

    def test_import_deliver(self):
        """Test importing deliver phase."""
        from phases.deliver import deliver_phase

        assert callable(deliver_phase)

    def test_phases_init_exports(self):
        """Test that phases/__init__.py exports all required functions."""
        from phases import (
            requirement_phase,
            develop_phase,
            fix_phase,
            full_test_phase,
            light_test_phase,
            deliver_phase,
        )

        assert callable(requirement_phase)
        assert callable(develop_phase)
        assert callable(fix_phase)
        assert callable(full_test_phase)
        assert callable(light_test_phase)
        assert callable(deliver_phase)


class TestPhaseModuleDocstrings:
    """Test that all phase modules have module-level docstrings."""

    def test_requirement_docstring(self):
        """Test requirement.py has module docstring."""
        import phases.requirement

        assert phases.requirement.__doc__ is not None
        assert "Input" in phases.requirement.__doc__
        assert "Output" in phases.requirement.__doc__

    def test_develop_docstring(self):
        """Test develop.py has module docstring."""
        import phases.develop

        assert phases.develop.__doc__ is not None
        assert "Input" in phases.develop.__doc__
        assert "Output" in phases.develop.__doc__

    def test_fix_docstring(self):
        """Test fix.py has module docstring."""
        import phases.fix

        assert phases.fix.__doc__ is not None
        assert "Input" in phases.fix.__doc__

    def test_test_docstring(self):
        """Test test.py has module docstring."""
        import phases.test

        assert phases.test.__doc__ is not None
        assert "Input" in phases.test.__doc__

    def test_deliver_docstring(self):
        """Test deliver.py has module docstring."""
        import phases.deliver

        assert phases.deliver.__doc__ is not None
        assert "Input" in phases.deliver.__doc__


class TestProjectStructure:
    """Test that all required files exist."""

    REQUIRED_FILES = [
        "config.py",
        "logger.py",
        "state.py",
        "prompts.py",
        "tool.py",
        "agent.py",
        "main.py",
        "phases/__init__.py",
        "phases/requirement.py",
        "phases/develop.py",
        "phases/fix.py",
        "phases/test.py",
        "phases/deliver.py",
    ]

    def test_required_files_exist(self):
        """Test that all required source files exist."""
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")
        for file_path in self.REQUIRED_FILES:
            full_path = os.path.join(project_root, file_path)
            assert os.path.exists(full_path), f"Missing required file: {file_path}"
            assert os.path.getsize(full_path) > 0, f"Empty file: {file_path}"
