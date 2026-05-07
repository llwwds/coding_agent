"""
State management module.

Defines AgentState, the central Pydantic model that tracks all information
throughout the agent's lifecycle across all phases.
"""

from typing import Optional
from pydantic import BaseModel, Field
from config import settings


class AgentState(BaseModel):
    """Central state object for the Coding Agent.

    Tracks requirements, technical plans, task templates, project skeleton,
    current phase, test results, fix rounds, performance metrics, and logs.

    Attributes:
        requirement: The original user requirement string.
        clarified_requirement: The clarified/refined requirement after analysis.
        tech_plan: The technical approach and architecture plan.
        task_template: Structured task template with input, target files, test commands,
            and acceptance criteria.
        project_skeleton: Definition of the project directory structure and configs.
        current_phase: Identifier of the current execution phase.
        fix_rounds: Counter for the current number of fix-retry rounds.
        max_fix_rounds: Maximum allowed fix rounds before human intervention.
        test_results: Accumulated list of test result dictionaries.
        failure_analysis: Accumulated list of failure analysis records.
        performance_metrics: Dictionary collecting performance indicators.
        phase_history: Ordered list of phases the agent has passed through.
        logs: Accumulated operation log entries.
        should_continue: Flag indicating whether the workflow should continue.
        human_intervention: Flag indicating whether human intervention is required.
        failed_at: Records which test phase failed (light_test or full_test).
    """

    requirement: str = ""
    clarified_requirement: str = ""
    tech_plan: str = ""
    task_template: dict = Field(default_factory=dict)
    project_skeleton: dict = Field(default_factory=dict)
    current_phase: str = "requirement"
    fix_rounds: int = 0
    max_fix_rounds: int = settings.MAX_FIX_ROUNDS
    test_results: list = Field(default_factory=list)
    failure_analysis: list = Field(default_factory=list)
    performance_metrics: dict = Field(default_factory=dict)
    phase_history: list = Field(default_factory=list)
    logs: list = Field(default_factory=list)
    should_continue: bool = True
    human_intervention: bool = False
    failed_at: Optional[str] = None


def save_checkpoint(state: AgentState, path: str) -> None:
    """Save the current agent state to a JSON checkpoint file.

    Args:
        state: The AgentState to persist.
        path: File path for the checkpoint JSON file.

    Raises:
        OSError: If the file cannot be written.
    """
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, ensure_ascii=False, indent=2, default=str)


def load_checkpoint(path: str) -> Optional[AgentState]:
    """Load agent state from a JSON checkpoint file.

    Args:
        path: File path of the checkpoint JSON file.

    Returns:
        An AgentState instance if the file exists and is valid, None otherwise.
    """
    import json
    import os

    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AgentState(**data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
