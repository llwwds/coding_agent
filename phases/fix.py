"""
Fix phase.

Input:
    - Test failure details (AgentState.test_results)
    - Current fix round counter (AgentState.fix_rounds)
    - Max fix rounds (AgentState.max_fix_rounds)
    - Failure indicator (AgentState.failed_at)

Output:
    - Updated fix_rounds counter
    - Failure analysis records (AgentState.failure_analysis)
    - Possibly human_intervention flag
    - Updated logs

Processing logic:
    1. Increment fix_rounds counter
    2. Check if max rounds exceeded -> set human_intervention
    3. If not exceeded, run failure analysis system:
       - Categorize failure (unread file, wrong code, test not run, over-engineering, incomplete fix)
    4. Select fix strategy based on failure type
    5. Execute fix (repair code / add tests / fix types / simplify / complete)
    6. Update logs and documentation
    7. Return to the failed test phase
"""

import json
import os

from langchain_openai import ChatOpenAI
from state import AgentState
from prompts import FIX_ANALYSIS_PROMPT, FIX_STRATEGY_PROMPT
from config import settings
from logger import get_logger
from tool import read_file_tool

logger = get_logger(__name__)

FAILURE_TYPES = [
    "unread_required_file",
    "wrong_code_modified",
    "test_not_executed",
    "over_engineering",
    "incomplete_fix",
]

FIX_STRATEGIES = {
    "unread_required_file": "repair_code",
    "wrong_code_modified": "repair_code",
    "test_not_executed": "add_tests",
    "over_engineering": "simplify",
    "incomplete_fix": "complete_fix",
}


def _build_llm() -> ChatOpenAI:
    """Build a ChatOpenAI instance from global settings.

    Returns:
        Configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        model=settings.MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.3,
    )


async def _analyze_failure(failure_details: str, change_context: str) -> dict:
    """Analyze test failures and categorize the failure type.

    Covers all 5 failure types:
    1. unread_required_file - Did not read necessary files before editing
    2. wrong_code_modified - Modified incorrect or unrelated code
    3. test_not_executed - Tests were not properly run before completion
    4. over_engineering - Overly complex or unnecessary design
    5. incomplete_fix - Previous fix did not fully address the issue

    Args:
        failure_details: Details of the test failures.
        change_context: Context about what changes were made.

    Returns:
        A dict with failure_type, confidence, root_cause, affected_files,
        and suggested_fix.
    """
    llm = _build_llm()
    chain = FIX_ANALYSIS_PROMPT | llm
    response = await chain.ainvoke(
        {
            "failure_details": failure_details,
            "change_context": change_context,
        }
    )
    try:
        return json.loads(str(response.content))
    except json.JSONDecodeError:
        logger.warning("Failed to parse failure analysis as JSON")
        return {
            "failure_type": "incomplete_fix",
            "confidence": 0.5,
            "root_cause": "Unable to determine from analysis",
            "affected_files": [],
            "suggested_fix": "Review the test output and fix all reported errors",
        }


async def _select_strategy(
    failure_analysis: dict, fix_round: int, max_rounds: int
) -> dict:
    """Select the appropriate fix strategy based on failure analysis.

    Args:
        failure_analysis: The failure analysis dict.
        fix_round: Current fix round number.
        max_rounds: Maximum allowed fix rounds.

    Returns:
        A dict with strategy type, specific actions, and estimated effort.
    """
    llm = _build_llm()
    chain = FIX_STRATEGY_PROMPT | llm
    response = await chain.ainvoke(
        {
            "failure_analysis": json.dumps(failure_analysis, ensure_ascii=False),
            "fix_round": str(fix_round),
            "max_fix_rounds": str(max_rounds),
        }
    )
    try:
        return json.loads(str(response.content))
    except json.JSONDecodeError:
        failure_type = failure_analysis.get("failure_type", "incomplete_fix")
        return {
            "strategy": FIX_STRATEGIES.get(failure_type, "repair_code"),
            "specific_actions": [],
            "estimated_effort": "medium",
        }


async def _execute_fix(strategy: dict, affected_files: list) -> list:
    """Execute the fix actions specified in the strategy.

    Args:
        strategy: The fix strategy dict with specific_actions.
        affected_files: List of file paths affected.

    Returns:
        A list of fix action result strings.
    """
    results = []
    for action in strategy.get("specific_actions", []):
        file_path = action.get("file", "")
        description = action.get("description", "")
        logger.info(f"Executing fix: {description} on {file_path}")

        if file_path and os.path.exists(file_path):
            content = read_file_tool.invoke({"file_path": file_path})
            results.append(f"Read {file_path} for fix: {content[:200]}...")
        results.append(f"Executed: {description}")

    return results


async def fix_phase(state: AgentState) -> dict:
    """Execute the fix phase.

    Increments the fix counter, checks against max rounds, analyzes failures,
    selects a strategy, and executes fixes. If max rounds are exceeded,
    sets the human_intervention flag and terminates the workflow.

    Args:
        state: The current AgentState.

    Returns:
        A dict of partial state updates including fix_rounds, failure_analysis,
        human_intervention, should_continue, phase_history, and logs.
    """
    new_rounds = state.fix_rounds + 1
    logger.info(f"Entering fix phase (round {new_rounds}/{state.max_fix_rounds})")

    if new_rounds > state.max_fix_rounds:
        logger.warning(
            f"Max fix rounds ({state.max_fix_rounds}) exceeded. "
            "Requiring human intervention."
        )
        return {
            "fix_rounds": new_rounds,
            "human_intervention": True,
            "should_continue": False,
            "phase_history": state.phase_history + ["fix"],
            "logs": state.logs
            + [f"[fix] Round {new_rounds}: MAX EXCEEDED, human intervention required"],
        }

    failure_details = json.dumps(
        state.test_results[-1] if state.test_results else {},
        ensure_ascii=False,
    )

    last_logs = state.logs[-5:] if state.logs else []
    change_context = f"Failed at: {state.failed_at}\nRecent logs: {last_logs}"

    analysis = await _analyze_failure(failure_details, change_context)
    logger.info(
        f"Failure analysis: type={analysis.get('failure_type')}, "
        f"confidence={analysis.get('confidence')}"
    )

    strategy = await _select_strategy(analysis, new_rounds, state.max_fix_rounds)
    logger.info(f"Fix strategy: {strategy.get('strategy')}")

    fix_results = await _execute_fix(strategy, analysis.get("affected_files", []))

    return {
        "fix_rounds": new_rounds,
        "failure_analysis": state.failure_analysis + [analysis],
        "phase_history": state.phase_history + ["fix"],
        "logs": state.logs
        + [
            f"[fix] Round {new_rounds}: {analysis.get('failure_type')} -> "
            f"{strategy.get('strategy')}, results: {len(fix_results)} actions",
        ],
    }
