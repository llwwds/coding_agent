"""
Main Agent logic using LangGraph StateGraph.

Builds a compiled state graph with nodes corresponding to each workflow phase:
    requirement -> develop -> light_test -> full_test -> fix -> deliver

Conditional edges enforce the exact flowchart:
    requirement: loop until clarified -> develop
    light_test: passed -> full_test | failed -> fix
    full_test:  passed -> deliver  | failed -> fix
    fix:        within limit -> retry test | exceeded -> END
    deliver:    -> END

Phase progress is automatically checkpointed to workspace/checkpoints/.
"""

import os

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from state import AgentState, save_checkpoint
from phases import (
    requirement_phase,
    develop_phase,
    fix_phase,
    full_test_phase,
    light_test_phase,
    deliver_phase,
)
from config import settings
from logger import get_logger

logger = get_logger(__name__)

CHECKPOINT_DIR = os.path.join(settings.WORKSPACE_DIR, "checkpoints")


def _checkpoint_state(state: AgentState, phase: str) -> None:
    """Save current state as a checkpoint for progress tracking.

    Args:
        state: The current AgentState to save.
        phase: Name of the phase for the checkpoint filename.
    """
    try:
        latest_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.json")
        phase_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{phase}.json")
        save_checkpoint(state, latest_path)
        save_checkpoint(state, phase_path)
        logger.debug(f"Checkpoint saved for phase: {phase}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


async def requirement_node(state: AgentState) -> dict:
    """Node wrapper for the requirement analysis phase.

    Args:
        state: The current AgentState.

    Returns:
        Partial state updates from the requirement phase.
    """
    logger.info("Executing requirement node")
    result = await requirement_phase(state)
    merged = state.model_dump()
    merged.update(result)
    _checkpoint_state(AgentState(**merged), "requirement")
    return result


async def develop_node(state: AgentState) -> dict:
    """Node wrapper for the development phase.

    Args:
        state: The current AgentState.

    Returns:
        Partial state updates from the development phase.
    """
    logger.info("Executing develop node")
    result = await develop_phase(state)
    merged = state.model_dump()
    merged.update(result)
    _checkpoint_state(AgentState(**merged), "develop")
    return result


async def light_test_node(state: AgentState) -> dict:
    """Node wrapper for the light test phase.

    Args:
        state: The current AgentState.

    Returns:
        Partial state updates from the light test phase.
    """
    logger.info("Executing light_test node")
    result = await light_test_phase(state)
    if not result["test_results"][-1].get("passed", False):
        result["failed_at"] = "light_test"
    merged = state.model_dump()
    merged.update(result)
    _checkpoint_state(AgentState(**merged), "light_test")
    return result


async def full_test_node(state: AgentState) -> dict:
    """Node wrapper for the full test phase.

    Args:
        state: The current AgentState.

    Returns:
        Partial state updates from the full test phase.
    """
    logger.info("Executing full_test node")
    result = await full_test_phase(state)
    if not result["test_results"][-1].get("passed", False):
        result["failed_at"] = "full_test"
    merged = state.model_dump()
    merged.update(result)
    _checkpoint_state(AgentState(**merged), "full_test")
    return result


async def fix_node(state: AgentState) -> dict:
    """Node wrapper for the fix phase.

    Args:
        state: The current AgentState.

    Returns:
        Partial state updates from the fix phase.
    """
    logger.info("Executing fix node")
    result = await fix_phase(state)
    merged = state.model_dump()
    merged.update(result)
    _checkpoint_state(AgentState(**merged), "fix")
    return result


async def deliver_node(state: AgentState) -> dict:
    """Node wrapper for the delivery phase.

    Args:
        state: The current AgentState.

    Returns:
        Partial state updates from the delivery phase.
    """
    logger.info("Executing deliver node")
    result = await deliver_phase(state)
    result["failed_at"] = None
    merged = state.model_dump()
    merged.update(result)
    _checkpoint_state(AgentState(**merged), "deliver")
    return result


def route_after_requirement(state: AgentState) -> str:
    """Determine next node after requirement phase.

    Routes to develop if requirement is clarified, otherwise loops back
    to requirement for further clarification.

    Args:
        state: The current AgentState.

    Returns:
        'develop' if clarified, 'requirement' otherwise.
    """
    if state.clarified_requirement and not state.human_intervention:
        logger.info("Requirement clarified, routing to develop")
        return "develop"
    logger.info("Requirement not yet clarified, looping back")
    return "requirement"


def route_after_light_test(state: AgentState) -> str:
    """Determine next node after light test phase.

    Routes to full_test if light tests passed, otherwise to fix.

    Args:
        state: The current AgentState.

    Returns:
        'full_test' if passed, 'fix' otherwise.
    """
    last_result = state.test_results[-1] if state.test_results else {}
    if last_result.get("passed", False):
        logger.info("Light tests passed, routing to full_test")
        return "full_test"
    logger.info("Light tests failed, routing to fix")
    return "fix"


def route_after_full_test(state: AgentState) -> str:
    """Determine next node after full test phase.

    Routes to deliver if all tests passed, otherwise to fix.

    Args:
        state: The current AgentState.

    Returns:
        'deliver' if passed, 'fix' otherwise.
    """
    full_results = [tr for tr in state.test_results if tr.get("phase") == "full_test"]
    last_result = full_results[-1] if full_results else {}
    if last_result.get("passed", False):
        logger.info("Full tests passed, routing to deliver")
        return "deliver"
    logger.info("Full tests failed, routing to fix")
    return "fix"


def route_after_fix(state: AgentState) -> str:
    """Determine next node after fix phase.

    If max rounds exceeded or human intervention required, route to END.
    Otherwise, route back to the test that failed (light_test or full_test).

    Args:
        state: The current AgentState.

    Returns:
        'light_test', 'full_test', or END.
    """
    if state.human_intervention or not state.should_continue:
        logger.info("Fix limit exceeded or intervention required, ending workflow")
        return END

    target = state.failed_at
    if target == "light_test":
        logger.info("Retrying light_test after fix")
        return "light_test"
    elif target == "full_test":
        logger.info("Retrying full_test after fix")
        return "full_test"

    logger.warning(f"Unknown failed_at: {target}, defaulting to END")
    return END


def build_graph() -> CompiledStateGraph:
    """Build and compile the LangGraph StateGraph for the Coding Agent.

    Constructs all nodes, conditional edges, and the entry point according
    to the defined workflow flowchart.

    Returns:
        A compiled StateGraph instance ready for invocation.
    """
    graph = StateGraph(AgentState)

    graph.add_node("requirement", requirement_node)
    graph.add_node("develop", develop_node)
    graph.add_node("light_test", light_test_node)
    graph.add_node("full_test", full_test_node)
    graph.add_node("fix", fix_node)
    graph.add_node("deliver", deliver_node)

    graph.set_entry_point("requirement")

    graph.add_conditional_edges(
        "requirement",
        route_after_requirement,
        {
            "develop": "develop",
            "requirement": "requirement",
        },
    )

    graph.add_edge("develop", "light_test")

    graph.add_conditional_edges(
        "light_test",
        route_after_light_test,
        {
            "full_test": "full_test",
            "fix": "fix",
        },
    )

    graph.add_conditional_edges(
        "full_test",
        route_after_full_test,
        {
            "deliver": "deliver",
            "fix": "fix",
        },
    )

    graph.add_conditional_edges(
        "fix",
        route_after_fix,
        {
            "light_test": "light_test",
            "full_test": "full_test",
            END: END,
        },
    )

    graph.add_edge("deliver", END)

    logger.info("StateGraph compiled successfully")
    return graph.compile()
