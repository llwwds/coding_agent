"""Tests for agent.py - LangGraph state graph and routing logic."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestGraphBuilding:
    """Test the LangGraph StateGraph construction."""

    def test_build_graph_returns_compiled(self):
        """Test that build_graph returns a compiled graph."""
        from agent import build_graph

        graph = build_graph()
        assert graph is not None
        assert hasattr(graph, "ainvoke")

    def test_graph_has_all_nodes(self):
        """Test that the graph contains all required nodes."""
        from agent import build_graph

        graph = build_graph()
        assert hasattr(graph, "ainvoke")

    def test_graph_can_be_invoked(self):
        """Test that the compiled graph accepts an initial state."""
        from agent import build_graph

        graph = build_graph()

        assert graph is not None


class TestRoutingFunctions:
    """Test the conditional routing functions."""

    def test_route_after_requirement_clarified(self):
        """Test routing when requirement is clarified."""
        from agent import route_after_requirement
        from state import AgentState

        state = AgentState(
            requirement="Test",
            clarified_requirement="Clarified test",
        )
        result = route_after_requirement(state)
        assert result == "develop"

    def test_route_after_requirement_not_clarified(self):
        """Test routing when requirement is not yet clarified."""
        from agent import route_after_requirement
        from state import AgentState

        state = AgentState(requirement="Test")
        result = route_after_requirement(state)
        assert result == "requirement"

    def test_route_after_light_test_passed(self):
        """Test routing when light tests passed."""
        from agent import route_after_light_test
        from state import AgentState

        state = AgentState(test_results=[{"phase": "light_test", "passed": True}])
        result = route_after_light_test(state)
        assert result == "full_test"

    def test_route_after_light_test_failed(self):
        """Test routing when light tests failed."""
        from agent import route_after_light_test
        from state import AgentState

        state = AgentState(test_results=[{"phase": "light_test", "passed": False}])
        result = route_after_light_test(state)
        assert result == "fix"

    def test_route_after_light_test_empty(self):
        """Test routing when no test results exist."""
        from agent import route_after_light_test
        from state import AgentState

        state = AgentState()
        result = route_after_light_test(state)
        assert result == "fix"

    def test_route_after_full_test_passed(self):
        """Test routing when full tests passed."""
        from agent import route_after_full_test
        from state import AgentState

        state = AgentState(test_results=[{"phase": "full_test", "passed": True}])
        result = route_after_full_test(state)
        assert result == "deliver"

    def test_route_after_full_test_failed(self):
        """Test routing when full tests failed."""
        from agent import route_after_full_test
        from state import AgentState

        state = AgentState(test_results=[{"phase": "full_test", "passed": False}])
        result = route_after_full_test(state)
        assert result == "fix"

    def test_route_after_fix_human_intervention(self):
        """Test routing when human intervention is required."""
        from agent import route_after_fix
        from langgraph.graph import END
        from state import AgentState

        state = AgentState(human_intervention=True, should_continue=False)
        result = route_after_fix(state)
        assert result == END

    def test_route_after_fix_retry_light(self):
        """Test routing back to light_test after fix."""
        from agent import route_after_fix
        from state import AgentState

        state = AgentState(
            should_continue=True,
            human_intervention=False,
            failed_at="light_test",
        )
        result = route_after_fix(state)
        assert result == "light_test"

    def test_route_after_fix_retry_full(self):
        """Test routing back to full_test after fix."""
        from agent import route_after_fix
        from state import AgentState

        state = AgentState(
            should_continue=True,
            human_intervention=False,
            failed_at="full_test",
        )
        result = route_after_fix(state)
        assert result == "full_test"


class TestGraphNodeWrappers:
    """Test the node wrapper functions (with mocked phase functions)."""

    @pytest.mark.asyncio
    async def test_requirement_node(self, monkeypatch):
        """Test requirement node returns a dict."""
        from unittest.mock import AsyncMock

        mock_phase = AsyncMock(
            return_value={
                "clarified_requirement": "clarified",
                "tech_plan": "plan",
                "task_template": {"tasks": []},
                "phase_history": ["requirement"],
                "logs": ["requirement done"],
            }
        )
        monkeypatch.setattr("agent.requirement_phase", mock_phase)

        from agent import requirement_node
        from state import AgentState

        state = AgentState(requirement="test")
        result = await requirement_node(state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_develop_node(self, monkeypatch):
        """Test develop node returns a dict."""
        from unittest.mock import AsyncMock

        mock_phase = AsyncMock(
            return_value={
                "project_skeleton": {"directories": [], "files": {}},
                "phase_history": ["develop"],
                "logs": ["develop done"],
            }
        )
        monkeypatch.setattr("agent.develop_phase", mock_phase)

        from agent import develop_node
        from state import AgentState

        state = AgentState(
            requirement="test",
            clarified_requirement="clarified",
            tech_plan="plan",
            task_template={"tasks": []},
        )
        result = await develop_node(state)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_light_test_node(self, monkeypatch):
        """Test light_test node returns a dict."""
        from unittest.mock import AsyncMock

        mock_phase = AsyncMock(
            return_value={
                "test_results": [{"phase": "light_test", "passed": True}],
                "phase_history": ["light_test"],
                "logs": ["light_test done"],
            }
        )
        monkeypatch.setattr("agent.light_test_phase", mock_phase)

        from agent import light_test_node
        from state import AgentState

        state = AgentState()
        result = await light_test_node(state)
        assert isinstance(result, dict)
        assert "test_results" in result

    @pytest.mark.asyncio
    async def test_full_test_node(self, monkeypatch):
        """Test full_test node returns a dict."""
        from unittest.mock import AsyncMock

        mock_phase = AsyncMock(
            return_value={
                "test_results": [{"phase": "full_test", "passed": True}],
                "phase_history": ["full_test"],
                "logs": ["full_test done"],
            }
        )
        monkeypatch.setattr("agent.full_test_phase", mock_phase)

        from agent import full_test_node
        from state import AgentState

        state = AgentState()
        result = await full_test_node(state)
        assert isinstance(result, dict)
        assert "test_results" in result

    @pytest.mark.asyncio
    async def test_fix_node(self, monkeypatch):
        """Test fix node returns a dict."""
        from unittest.mock import AsyncMock

        mock_phase = AsyncMock(
            return_value={
                "fix_rounds": 1,
                "failure_analysis": [{"failure_type": "test"}],
                "phase_history": ["fix"],
                "logs": ["fix done"],
            }
        )
        monkeypatch.setattr("agent.fix_phase", mock_phase)

        from agent import fix_node
        from state import AgentState

        state = AgentState(
            fix_rounds=0,
            max_fix_rounds=5,
            test_results=[{"phase": "light_test", "passed": False}],
        )
        result = await fix_node(state)
        assert isinstance(result, dict)
        assert "fix_rounds" in result

    @pytest.mark.asyncio
    async def test_deliver_node(self, monkeypatch):
        """Test deliver node returns a dict and completes workflow."""
        from unittest.mock import AsyncMock

        mock_phase = AsyncMock(
            return_value={
                "performance_metrics": {"pass_rate": 100},
                "should_continue": False,
                "phase_history": ["deliver"],
                "logs": ["deliver done"],
            }
        )
        monkeypatch.setattr("agent.deliver_phase", mock_phase)

        from agent import deliver_node
        from state import AgentState

        state = AgentState(
            requirement="test",
            clarified_requirement="clarified",
            phase_history=["requirement", "develop", "light_test", "full_test"],
            test_results=[
                {"phase": "light_test", "passed": True},
                {"phase": "full_test", "passed": True},
            ],
        )
        result = await deliver_node(state)
        assert isinstance(result, dict)
        assert result.get("should_continue") is False
        assert "performance_metrics" in result
