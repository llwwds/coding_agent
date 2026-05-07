"""
Delivery phase.

Input:
    - Performance metrics (AgentState.performance_metrics)
    - Test results (AgentState.test_results)
    - Phase history (AgentState.phase_history)
    - Failure analysis records (AgentState.failure_analysis)

Output:
    - Final delivery report
    - Updated performance_metrics
    - Workflow termination (should_continue = False)

Processing logic:
    1. Collect performance metrics:
       - Pass rate
       - First-pass rate
       - Average fix rounds
       - Mis-change rate
    2. Generate comprehensive project report and documentation
    3. Deliver project, end workflow
"""

import json
import os

from langchain_openai import ChatOpenAI
from state import AgentState
from prompts import DELIVER_REPORT_PROMPT
from config import settings
from logger import get_logger

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


def _calculate_metrics(state: AgentState) -> dict:
    """Calculate performance metrics from collected data.

    Args:
        state: The current AgentState with all historical data.

    Returns:
        A dict with pass_rate, first_pass_rate, avg_fix_rounds, and mis_change_rate.
    """
    test_results = state.test_results

    total_tests = len(test_results)
    passed_tests = sum(1 for tr in test_results if tr.get("passed", False))
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0

    light_passed = any(
        tr.get("phase") == "light_test" and tr.get("passed")
        for tr in reversed(test_results)
    )
    full_passed = any(
        tr.get("phase") == "full_test" and tr.get("passed")
        for tr in reversed(test_results)
    )
    first_pass_rate = 100.0 if (light_passed and full_passed) else 0.0

    avg_fix_rounds = (
        state.fix_rounds / max(1, len([log for log in state.logs if "fix" in log]))
        if state.fix_rounds > 0
        else 0.0
    )

    fix_entries = len(
        [
            fa
            for fa in state.failure_analysis
            if fa.get("failure_type") == "wrong_code_modified"
        ]
    )
    total_changes = max(1, len(state.test_results))
    mis_change_rate = fix_entries / total_changes * 100

    return {
        "pass_rate": round(pass_rate, 2),
        "first_pass_rate": round(first_pass_rate, 2),
        "avg_fix_rounds": round(avg_fix_rounds, 2),
        "mis_change_rate": round(mis_change_rate, 2),
    }


async def _generate_report(metrics: dict, state: AgentState) -> dict:
    """Generate a comprehensive delivery report using LLM.

    Args:
        metrics: Calculated performance metrics.
        state: The current AgentState.

    Returns:
        A dict with project_summary, metrics, test_summary, recommendations,
        and delivered_files.
    """
    llm = _build_llm()
    chain = DELIVER_REPORT_PROMPT | llm
    response = await chain.ainvoke(
        {
            "performance_metrics": json.dumps(metrics, ensure_ascii=False),
            "test_results": json.dumps(
                state.test_results, ensure_ascii=False, default=str
            ),
            "phase_history": json.dumps(state.phase_history, ensure_ascii=False),
            "failure_analysis": json.dumps(state.failure_analysis, ensure_ascii=False),
        }
    )
    try:
        return json.loads(str(response.content))
    except json.JSONDecodeError:
        return {
            "project_summary": f"Project based on: {state.clarified_requirement[:200]}",
            "metrics": metrics,
            "test_summary": f"{len(state.test_results)} test phases completed",
            "recommendations": ["Review generated code for correctness"],
            "delivered_files": [],
        }


def _gather_delivered_files() -> list:
    """Scan the workspace directory to list all delivered files.

    Returns:
        A list of relative file paths in the workspace.
    """
    workdir = settings.WORKSPACE_DIR
    if not os.path.exists(workdir):
        return []
    files = []
    for root, _, filenames in os.walk(workdir):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            relative = os.path.relpath(full_path, workdir)
            files.append(relative)
    return sorted(files)


async def deliver_phase(state: AgentState) -> dict:
    """Execute the delivery phase.

    Calculates final performance metrics, generates a delivery report,
    and marks the workflow as complete.

    Args:
        state: The current AgentState.

    Returns:
        A dict with performance_metrics, should_continue=False,
        phase_history, and logs.
    """
    logger.info("Entering delivery phase")

    metrics = _calculate_metrics(state)
    logger.info(f"Performance metrics calculated: {metrics}")

    report = await _generate_report(metrics, state)
    logger.info("Delivery report generated")

    delivered_files = _gather_delivered_files()
    report["delivered_files"] = delivered_files

    report_path = os.path.join(settings.WORKSPACE_DIR, "DELIVERY_REPORT.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Delivery report saved to {report_path}")

    print("\n" + "=" * 60)
    print("DELIVERY REPORT")
    print("=" * 60)
    print(f"  Pass Rate:       {metrics['pass_rate']}%")
    print(f"  First-Pass Rate: {metrics['first_pass_rate']}%")
    print(f"  Avg Fix Rounds:  {metrics['avg_fix_rounds']}")
    print(f"  Mis-Change Rate: {metrics['mis_change_rate']}%")
    print(f"  Files Delivered: {len(delivered_files)}")
    print("=" * 60)
    print(f"Full report saved to: {report_path}")

    return {
        "performance_metrics": metrics,
        "should_continue": False,
        "phase_history": state.phase_history + ["deliver"],
        "logs": state.logs
        + [
            f"[deliver] Pass rate: {metrics['pass_rate']}%, "
            f"First-pass: {metrics['first_pass_rate']}%, "
            f"Fix rounds: {metrics['avg_fix_rounds']}"
        ],
    }
