"""Tests for state.py - AgentState model and checkpointing."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestAgentState:
    """Tests for the AgentState Pydantic model."""

    def test_default_state(self):
        """Test AgentState default values."""
        from state import AgentState

        state = AgentState()
        assert state.requirement == ""
        assert state.current_phase == "requirement"
        assert state.fix_rounds == 0
        assert state.max_fix_rounds == 5
        assert state.should_continue is True
        assert state.human_intervention is False
        assert state.task_template == {}
        assert state.project_skeleton == {}
        assert state.test_results == []
        assert state.failure_analysis == []
        assert state.performance_metrics == {}
        assert state.phase_history == []
        assert state.logs == []
        assert state.failed_at is None

    def test_state_with_requirement(self):
        """Test AgentState creation with a requirement."""
        from state import AgentState

        state = AgentState(requirement="Build a REST API")
        assert state.requirement == "Build a REST API"
        assert state.current_phase == "requirement"

    def test_state_custom_max_fix_rounds(self):
        """Test custom max_fix_rounds setting."""
        from state import AgentState

        state = AgentState(max_fix_rounds=10)
        assert state.max_fix_rounds == 10

    def test_state_dict_fields(self):
        """Test that dict/list fields start empty."""
        from state import AgentState

        state = AgentState()
        assert state.task_template == {}
        assert state.test_results == []
        assert state.failure_analysis == []

    def test_state_serialization(self):
        """Test that AgentState can be serialized to JSON."""
        from state import AgentState

        state = AgentState(
            requirement="Test",
            fix_rounds=2,
            phase_history=["requirement", "develop"],
            logs=["log1", "log2"],
        )
        data = state.model_dump()
        assert data["requirement"] == "Test"
        assert data["fix_rounds"] == 2
        assert len(data["phase_history"]) == 2

    def test_state_deserialization(self):
        """Test that AgentState can be deserialized from JSON."""
        from state import AgentState

        data = {
            "requirement": "Test",
            "current_phase": "develop",
            "fix_rounds": 1,
            "max_fix_rounds": 5,
            "should_continue": True,
            "human_intervention": False,
        }
        state = AgentState(**data)
        assert state.requirement == "Test"
        assert state.current_phase == "develop"


class TestCheckpointing:
    """Tests for state checkpoint save/load functionality."""

    def test_save_and_load_checkpoint(self):
        """Test saving and loading a state checkpoint."""
        from state import AgentState, save_checkpoint, load_checkpoint

        state = AgentState(
            requirement="Test requirement",
            current_phase="develop",
            fix_rounds=3,
            phase_history=["requirement", "develop"],
            logs=["entry1", "entry2"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.json")
            save_checkpoint(state, path)

            assert os.path.exists(path)

            loaded = load_checkpoint(path)
            assert loaded is not None
            assert loaded.requirement == "Test requirement"
            assert loaded.current_phase == "develop"
            assert loaded.fix_rounds == 3
            assert loaded.phase_history == ["requirement", "develop"]

    def test_load_missing_checkpoint(self):
        """Test loading a checkpoint that doesn't exist."""
        from state import load_checkpoint

        result = load_checkpoint("/nonexistent/path/checkpoint.json")
        assert result is None

    def test_load_invalid_checkpoint(self):
        """Test loading a corrupt checkpoint file."""
        from state import load_checkpoint

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            path = f.name

        try:
            result = load_checkpoint(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_checkpoint_preserves_test_results(self):
        """Test that checkpoint preserves test results."""
        from state import AgentState, save_checkpoint, load_checkpoint

        state = AgentState(
            test_results=[
                {"phase": "light_test", "passed": True},
                {"phase": "full_test", "passed": False},
            ],
            failure_analysis=[{"failure_type": "incomplete_fix"}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.json")
            save_checkpoint(state, path)
            loaded = load_checkpoint(path)

            assert loaded is not None
            assert len(loaded.test_results) == 2
            assert len(loaded.failure_analysis) == 1
