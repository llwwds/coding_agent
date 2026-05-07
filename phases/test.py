"""
Test phases (light and full).

Light Test Phase:
    Input: Generated code from develop phase
    Output: Test results (ruff/black/mypy/lint results)
    Processing: Runs quality gate checks and determines pass/fail

Full Test Phase:
    Input: Code that passed light tests
    Output: Complete test suite results (unit/integration/e2e), coverage data
    Processing:
        1. Create isolated test environment (virtual env / mock data)
        2. Execute unit tests
        3. Execute integration tests
        4. Execute end-to-end tests
        5. Trigger CI/CD flow (GitHub Actions)
        6. Determine overall pass/fail status
"""

import json
import os

from langchain_openai import ChatOpenAI
from state import AgentState
from prompts import LIGHT_TEST_PROMPT, FULL_TEST_PROMPT
from config import settings
from logger import get_logger
from tool import execute_command_tool, run_test_tool

logger = get_logger(__name__)


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


async def light_test_phase(state: AgentState) -> dict:
    """Execute lightweight quality checks.

    Runs ruff, black, mypy, and lint checks. Uses LLM to analyze the output
    and determine if the code passes quality gates.

    Args:
        state: The current AgentState.

    Returns:
        A dict with test_results, phase_history, and logs. The test result
        dict includes a 'passed' key indicating overall status.
    """
    logger.info("Entering light test phase")
    workdir = settings.WORKSPACE_DIR

    ruff_out = run_test_tool.invoke({"test_type": "ruff", "workdir": workdir})
    black_out = run_test_tool.invoke({"test_type": "black", "workdir": workdir})
    mypy_out = run_test_tool.invoke({"test_type": "mypy", "workdir": workdir})

    llm = _build_llm()
    chain = LIGHT_TEST_PROMPT | llm
    response = await chain.ainvoke(
        {
            "ruff_output": ruff_out,
            "black_output": black_out,
            "mypy_output": mypy_out,
            "lint_output": ruff_out,
        }
    )

    try:
        result = json.loads(str(response.content))
    except json.JSONDecodeError:
        passed = (
            "exit code: 0" in ruff_out
            and "exit code: 0" in black_out
            and "exit code: 0" in mypy_out
        )
        result = {
            "passed": passed,
            "issues": [],
            "summary": f"Ruff: {ruff_out[:200]}\nBlack: {black_out[:200]}\nMypy: {mypy_out[:200]}",
        }

    logger.info(f"Light test result: {'PASSED' if result.get('passed') else 'FAILED'}")

    return {
        "test_results": state.test_results + [{"phase": "light_test", **result}],
        "phase_history": state.phase_history + ["light_test"],
        "logs": state.logs
        + [f"[light_test] {'PASSED' if result.get('passed') else 'FAILED'}"],
    }


async def _setup_test_environment() -> str:
    """Set up an isolated test environment.

    Creates a virtual environment or configures the test directory with
    necessary dependencies and mock data.

    Returns:
        Status message about the test environment setup.
    """
    workdir = settings.WORKSPACE_DIR
    tests_dir = os.path.join(workdir, "tests")
    os.makedirs(tests_dir, exist_ok=True)

    test_config = """
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
    test_config_path = os.path.join(tests_dir, "conftest.py")
    if not os.path.exists(test_config_path):
        with open(test_config_path, "w", encoding="utf-8") as f:
            f.write(test_config.strip())
        logger.info(f"Created test conftest.py at {test_config_path}")

    logger.info(f"Test environment setup complete at {tests_dir}")
    return f"Test environment ready at {tests_dir}"


async def _run_unit_tests() -> str:
    """Run unit tests using pytest.

    Returns:
        Pytest output string.
    """
    return run_test_tool.invoke(
        {"test_type": "pytest", "workdir": settings.WORKSPACE_DIR}
    )


async def _run_integration_tests() -> str:
    """Run integration tests using pytest with integration marker.

    Returns:
        Pytest output string.
    """
    cmd = "pytest -m integration --tb=short 2>&1 || echo 'No integration tests found'"
    return execute_command_tool.invoke(
        {"command": cmd, "workdir": settings.WORKSPACE_DIR}
    )


async def _run_e2e_tests() -> str:
    """Run end-to-end tests using pytest with e2e marker.

    Returns:
        Pytest output string.
    """
    cmd = "pytest -m e2e --tb=short 2>&1 || echo 'No e2e tests found'"
    return execute_command_tool.invoke(
        {"command": cmd, "workdir": settings.WORKSPACE_DIR}
    )


async def _get_coverage() -> str:
    """Get test coverage report.

    Returns:
        Coverage output string.
    """
    cmd = "pytest --cov=. --cov-report=term 2>&1 || echo 'Coverage report unavailable'"
    return execute_command_tool.invoke(
        {"command": cmd, "workdir": settings.WORKSPACE_DIR}
    )


async def _trigger_ci_cd() -> str:
    """Trigger CI/CD pipeline (simulate GitHub Actions check).

    In a real environment, this would push to a branch and wait for
    GitHub Actions to complete. Here we simulate by running equivalent
    checks locally.

    Returns:
        CI/CD status output string.
    """
    ci_commands = [
        "ruff check .",
        "black --check .",
        "mypy .",
        "pytest --tb=short",
    ]
    results = []
    for cmd in ci_commands:
        output = execute_command_tool.invoke(
            {
                "command": cmd,
                "workdir": settings.WORKSPACE_DIR,
            }
        )
        results.append(f"CI [{cmd}]:\n{output[:500]}")
    return "\n\n".join(results)


async def full_test_phase(state: AgentState) -> dict:
    """Execute the full test phase.

    Creates an isolated test environment, runs unit, integration, and
    end-to-end tests, collects coverage data, and simulates CI/CD.

    Args:
        state: The current AgentState.

    Returns:
        A dict with test_results, phase_history, and logs.
    """
    logger.info("Entering full test phase")

    env_status = await _setup_test_environment()
    logger.info(env_status)

    logger.info("Running unit tests...")
    unit_output = await _run_unit_tests()

    logger.info("Running integration tests...")
    integration_output = await _run_integration_tests()

    logger.info("Running end-to-end tests...")
    e2e_output = await _run_e2e_tests()

    logger.info("Collecting coverage...")
    coverage_output = await _get_coverage()

    logger.info("Triggering CI/CD simulation...")
    await _trigger_ci_cd()

    llm = _build_llm()
    chain = FULL_TEST_PROMPT | llm
    response = await chain.ainvoke(
        {
            "unit_output": unit_output,
            "integration_output": integration_output,
            "e2e_output": e2e_output,
            "coverage_output": coverage_output,
        }
    )

    try:
        result = json.loads(str(response.content))
    except json.JSONDecodeError:
        unit_passed = "exit code: 0" in unit_output
        integration_passed = "exit code: 0" in integration_output
        e2e_passed = "exit code: 0" in e2e_output
        passed = unit_passed and integration_passed and e2e_passed
        result = {
            "passed": passed,
            "unit_test_results": {"passed": unit_passed, "output": unit_output},
            "integration_test_results": {
                "passed": integration_passed,
                "output": integration_output,
            },
            "e2e_test_results": {"passed": e2e_passed, "output": e2e_output},
            "coverage_percentage": 0,
            "failure_categories": [],
            "summary": f"Unit: {'OK' if unit_passed else 'FAIL'}, "
            f"Integration: {'OK' if integration_passed else 'FAIL'}, "
            f"E2E: {'OK' if e2e_passed else 'FAIL'}",
        }

    passed = result.get("passed", False)
    logger.info(f"Full test result: {'PASSED' if passed else 'FAILED'}")

    return {
        "test_results": state.test_results + [{"phase": "full_test", **result}],
        "phase_history": state.phase_history + ["full_test"],
        "logs": state.logs + [f"[full_test] {'PASSED' if passed else 'FAILED'}"],
    }
