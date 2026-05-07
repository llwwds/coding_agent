"""
Entry point for the Coding Agent.

Loads configuration, initializes the LLM and tools, builds the LangGraph
state graph, accepts user input for requirements, initializes AgentState,
and executes the complete workflow via graph.ainvoke().
"""

import asyncio
import sys

from langchain_openai import ChatOpenAI

from config import settings
from state import AgentState
from agent import build_graph
from tool import ALL_TOOLS
from logger import get_logger

logger = get_logger("main")


async def main() -> None:
    """Main async entry point for the Coding Agent.

    Performs the full lifecycle:
        1. Validate configuration
        2. Initialize LLM
        3. Build and compile the state graph
        4. Accept user requirement input
        5. Initialize AgentState
        6. Execute graph.ainvoke(state)
        7. Display final results

    Raises:
        SystemExit: If configuration is invalid.
        RuntimeError: If the workflow encounters an unrecoverable error.
    """
    print("=" * 60)
    print("  Coding Agent - LangChain-based Autonomous Agent")
    print("=" * 60)
    print()

    try:
        settings.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)

    logger.info("Configuration loaded successfully")
    logger.info(f"Model: {settings.MODEL_NAME}")
    logger.info(f"Workspace: {settings.WORKSPACE_DIR}")

    _ = ChatOpenAI(
        model=settings.MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.3,
    )
    logger.info("LLM initialized")

    print(f"Tools loaded: {len(ALL_TOOLS)} tools available")
    for tool in ALL_TOOLS:
        print(f"  - {tool.name}: {tool.description[:60]}...")
    print()

    print("Enter your coding requirement (type 'done' on a new line to finish):")
    print("-" * 40)

    lines = []
    while True:
        line = input("> ")
        if line.strip().lower() == "done":
            break
        lines.append(line)

    requirement = "\n".join(lines).strip()
    if not requirement:
        print("No requirement provided. Exiting.")
        return

    logger.info(f"Requirement received ({len(requirement)} chars)")

    graph = build_graph()
    logger.info("State graph compiled")

    state = AgentState(requirement=requirement)
    logger.info(f"Initial state created, starting from phase: {state.current_phase}")

    print()
    print("Starting workflow...")
    print("-" * 40)

    try:
        final_state_dict = await graph.ainvoke(state)
        final_state = AgentState(
            **{
                k: v
                for k, v in final_state_dict.items()
                if k in AgentState.model_fields
            }
        )
    except Exception as e:
        logger.error(f"Workflow execution error: {e}")
        print(f"ERROR during workflow: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  WORKFLOW COMPLETE")
    print("=" * 60)

    if final_state.human_intervention:
        print("\n  *** HUMAN INTERVENTION REQUIRED ***")
        print(f"  Max fix rounds ({final_state.max_fix_rounds}) exceeded.")
        print(f"  Phase history: {' -> '.join(final_state.phase_history)}")
    else:
        print(f"\n  Phase history: {' -> '.join(final_state.phase_history)}")
        if final_state.performance_metrics:
            metrics = final_state.performance_metrics
            print(f"  Pass Rate:        {metrics.get('pass_rate', 'N/A')}%")
            print(f"  First-Pass Rate:  {metrics.get('first_pass_rate', 'N/A')}%")
            print(f"  Avg Fix Rounds:   {metrics.get('avg_fix_rounds', 'N/A')}")
            print(f"  Mis-Change Rate:  {metrics.get('mis_change_rate', 'N/A')}%")

    print(f"\n  Logs: {len(final_state.logs)} entries")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
