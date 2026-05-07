"""
Requirement analysis and clarification phase.

Input:
    - User's original requirement string (in AgentState.requirement)

Output:
    - Clarified/refined requirement (AgentState.clarified_requirement)
    - Technical plan (AgentState.tech_plan)
    - Structured task template (AgentState.task_template)

Processing logic:
    1. Receive user requirement
    2. LLM analyzes requirement, identifies tech stack and functional boundaries
    3. Generate structured task template (input, target files, test commands, acceptance criteria)
    4. Interact with user to clarify ambiguities
    5. Loop until requirement and tech plan are both clear
"""

import json
import asyncio

from langchain_openai import ChatOpenAI
from state import AgentState
from prompts import (
    REQUIREMENT_ANALYSIS_PROMPT,
    CLARIFY_REQUIREMENT_PROMPT,
    TASK_TEMPLATE_PROMPT,
)
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


async def _analyze_requirement(requirement: str) -> dict:
    """Use LLM to analyze the initial requirement.

    Args:
        requirement: The raw user requirement string.

    Returns:
        A dict containing the analysis with summary, tech_stack,
        features, boundaries, risks, dependencies, and questions.
    """
    llm = _build_llm()
    chain = REQUIREMENT_ANALYSIS_PROMPT | llm
    response = await chain.ainvoke({"requirement": requirement})
    try:
        return json.loads(str(response.content))
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON, using raw text")
        return {
            "summary": response.content,
            "tech_stack": [],
            "features": [],
            "boundaries": "",
            "risks": [],
            "dependencies": [],
            "questions_for_clarification": [],
        }


async def _clarify_requirement(
    original: str, analysis: dict, user_response: str
) -> dict:
    """Clarify requirement based on user feedback.

    Args:
        original: The original requirement string.
        analysis: The previous analysis dict.
        user_response: User's response to clarification questions.

    Returns:
        A dict with clarified_requirement, remaining_ambiguities, and is_clear flag.
    """
    llm = _build_llm()
    chain = CLARIFY_REQUIREMENT_PROMPT | llm
    response = await chain.ainvoke(
        {
            "original_requirement": original,
            "analysis": json.dumps(analysis, ensure_ascii=False),
            "user_response": user_response,
        }
    )
    try:
        return json.loads(str(response.content))
    except json.JSONDecodeError:
        return {
            "clarified_requirement": response.content,
            "remaining_ambiguities": [],
            "is_clear": True,
        }


async def _generate_task_template(clarified: str, tech_plan: str) -> dict:
    """Generate a structured task template.

    Args:
        clarified: The clarified requirement string.
        tech_plan: The technical plan string.

    Returns:
        A dict with tasks list and overall acceptance criteria.
    """
    llm = _build_llm()
    chain = TASK_TEMPLATE_PROMPT | llm
    response = await chain.ainvoke(
        {
            "clarified_requirement": clarified,
            "tech_plan": tech_plan,
        }
    )
    try:
        return json.loads(str(response.content))
    except json.JSONDecodeError:
        return {
            "tasks": [],
            "overall_acceptance": ["Code compiles and runs without errors"],
        }


async def _interact_with_user(questions: list, prompt_text: str = "") -> str:
    """Present questions to the user and collect their response.

    In a CLI context, prints questions and reads from stdin.
    In an async context, uses asyncio to avoid blocking.

    Args:
        questions: List of clarification questions to ask the user.
        prompt_text: Additional prompt text to display.

    Returns:
        The user's response string.
    """
    if not questions:
        return ""

    print("\n" + "=" * 60)
    print("Requirement Clarification")
    print("=" * 60)
    if prompt_text:
        print(prompt_text)
    print("\nPlease answer the following questions to clarify the requirement:\n")

    for i, question in enumerate(questions, 1):
        print(f"  {i}. {question}")

    print("\n" + "-" * 40)
    print("Enter your response (type 'done' on a new line to finish):")
    print("-" * 40)

    loop = asyncio.get_event_loop()
    lines = []
    while True:
        line = await loop.run_in_executor(None, input, "> ")
        if line.strip().lower() == "done":
            break
        lines.append(line)

    user_response = "\n".join(lines)
    logger.info(f"User clarification response received ({len(user_response)} chars)")
    return user_response


async def requirement_phase(state: AgentState) -> dict:
    """Execute the requirement analysis and clarification phase.

    This function is designed to be called repeatedly until the requirement
    is clear. It tracks whether this is the first call or a subsequent
    clarification round via the state.

    Args:
        state: The current AgentState.

    Returns:
        A dict of partial state updates including clarified_requirement,
        tech_plan, task_template, phase_history, and logs.
    """
    logger.info("Entering requirement analysis phase")

    if not state.clarified_requirement:
        analysis = await _analyze_requirement(state.requirement)
        logger.info(
            f"Requirement analysis complete: {analysis.get('summary', '')[:100]}"
        )

        tech_plan = json.dumps(
            {
                "tech_stack": analysis.get("tech_stack", []),
                "boundaries": analysis.get("boundaries", ""),
                "risks": analysis.get("risks", []),
                "dependencies": analysis.get("dependencies", []),
            },
            ensure_ascii=False,
        )

        questions = analysis.get("questions_for_clarification", [])

        if questions:
            user_response = await _interact_user(questions)
            if user_response:
                clarification = await _clarify_requirement(
                    state.requirement, analysis, user_response
                )
                clarified_text = clarification.get(
                    "clarified_requirement", user_response
                )
                is_clear = clarification.get("is_clear", True)
            else:
                clarified_text = state.requirement
                is_clear = True
        else:
            clarified_text = state.requirement
            is_clear = True

        if is_clear and clarified_text:
            task_template = await _generate_task_template(clarified_text, tech_plan)
            logger.info("Task template generated successfully")
            return {
                "clarified_requirement": clarified_text,
                "tech_plan": tech_plan,
                "task_template": task_template,
                "phase_history": state.phase_history + ["requirement"],
                "logs": state.logs
                + ["[requirement] Analysis complete, requirement clarified"],
            }
        else:
            logger.info("Requirement still unclear, requesting clarification")
            return {
                "clarified_requirement": "",
                "tech_plan": tech_plan,
                "task_template": {},
                "phase_history": state.phase_history + ["requirement"],
                "logs": state.logs + ["[requirement] Further clarification needed"],
            }
    else:
        task_template = await _generate_task_template(
            state.clarified_requirement, state.tech_plan
        )
        logger.info("Task template generated from clarified requirement")
        return {
            "task_template": task_template,
            "phase_history": state.phase_history + ["requirement"],
            "logs": state.logs + ["[requirement] Task template finalized"],
        }


async def _interact_user(questions: list) -> str:
    """Handle user interaction for requirement clarification.

    Args:
        questions: List of clarification questions.

    Returns:
        User response string.
    """
    return await _interact_with_user(questions)
