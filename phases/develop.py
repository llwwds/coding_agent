"""
Development phase.

Input:
    - Clarified requirement (AgentState.clarified_requirement)
    - Technical plan (AgentState.tech_plan)
    - Task template (AgentState.task_template)

Output:
    - Project skeleton definition (AgentState.project_skeleton)
    - Generated code files in workspace
    - AGENTS.md documentation

Processing logic:
    1. LLM determines project skeleton, directory structure, configs
    2. Skeleton includes: config, logger, main entry, test directory
    3. Predefine sub-agent content: prompts, conventions, environment cache
    4. Generate AGENTS.md (role, code style, commit conventions, test requirements)
    5. Define work quantification (coupling, lines, time, complexity)
    6. Single-agent mode: fill code based on skeleton
    7. Execute lightweight tests (ruff, black, mypy, lint, pre-commit hooks)
"""

import json
import os

from langchain_openai import ChatOpenAI
from state import AgentState
from prompts import SKELETON_DESIGN_PROMPT, CODE_GENERATION_PROMPT
from config import settings
from logger import get_logger
from tool import execute_command_tool

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


async def _design_skeleton(task_template: dict, tech_plan: str) -> dict:
    """Design the project skeleton using LLM.

    Args:
        task_template: The structured task template.
        tech_plan: The technical plan string.

    Returns:
        A dict with directories, files, AGENTS.md content, and quantification data.
    """
    llm = _build_llm()
    chain = SKELETON_DESIGN_PROMPT | llm
    response = await chain.ainvoke(
        {
            "task_template": json.dumps(task_template, ensure_ascii=False),
            "tech_plan": tech_plan,
        }
    )
    try:
        return json.loads(str(response.content))
    except json.JSONDecodeError:
        logger.warning("Failed to parse skeleton design as JSON")
        return {
            "directories": ["src", "tests", "logs"],
            "files": {},
            "agents_md_content": "",
            "module_coupling_plan": "",
            "estimated_lines": {},
            "estimated_dev_time": {},
            "test_complexity": "medium",
        }


async def _generate_code(
    module_name: str, specification: str, context: str, skeleton: dict
) -> str:
    """Generate code for a specific module using LLM.

    Args:
        module_name: Name of the module to implement.
        specification: Module specification details.
        context: Project context information.
        skeleton: The project skeleton definition.

    Returns:
        Generated source code as a string.
    """
    llm = _build_llm()
    chain = CODE_GENERATION_PROMPT | llm
    response = await chain.ainvoke(
        {
            "module_name": module_name,
            "specification": specification,
            "context": context,
            "skeleton": json.dumps(skeleton, ensure_ascii=False),
        }
    )
    return str(response.content)


def _generate_agents_md() -> str:
    """Generate the AGENTS.md content with standard sections.

    Returns:
        AGENTS.md content as a string.
    """
    return """# Coding Agent

## Role Definition

The Coding Agent is an autonomous single-agent system built on LangChain/LangGraph
that performs end-to-end software development tasks. Its responsibilities include:
- Requirement analysis and clarification
- Project skeleton design and code generation
- Automated testing (linting, type checking, unit/integration/e2e)
- Failure analysis and iterative bug fixing
- Performance metrics collection and delivery reporting

**Capability Boundaries:**
- Works within the workspace directory only
- Requires explicit user approval for external API calls
- Does not deploy to production environments
- Does not modify system-level configurations

## Code Style

- **Naming:** snake_case for variables/functions, PascalCase for classes,
  UPPER_CASE for constants. Modules use lowercase with underscores.
- **Formatting:** All code must be formatted with `black` (line length 100).
- **Type Annotations:** All public functions must have complete type annotations.
  Use `mypy` for static type checking.
- **Docstrings:** Google-style docstrings for all public functions and classes.
- **Imports:** Standard library first, then third-party, then local. Alphabetical
  within each group.
- **Error Handling:** Use explicit exception handling. Never use bare `except:`.

## Commit Conventions

- Format: `<type>: <description>`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `style`, `perf`
- Examples:
  - `feat: add user authentication module`
  - `fix: resolve null pointer in config loader`
  - `test: add unit tests for state management`
- Keep commits atomic and focused on a single change.
- Do not commit secrets, .env files, or generated artifacts.

## Test Requirements

- **Coverage target:** > 80% line coverage
- **Test types required:**
  - Unit tests for all public functions
  - Integration tests for module interactions
  - End-to-end tests for complete workflows
- **Test framework:** pytest
- **Running tests:** `pytest --cov=. --cov-report=term`
- **Pre-commit checks:** ruff, black --check, mypy
- Tests must be independent and not rely on execution order.
"""


async def _run_light_tests(workdir: str) -> dict:
    """Run the lightweight quality checks in sequence.

    Order: ruff -> black -> mypy -> lint

    Args:
        workdir: Working directory for running tests.

    Returns:
        A dict with results for each tool and overall passed status.
    """
    results = {}

    logger.info("Running ruff check...")
    ruff_out = execute_command_tool.invoke(
        {"command": "ruff check .", "workdir": workdir}
    )
    results["ruff"] = ruff_out
    ruff_passed = "exit code: 0" in ruff_out or "No errors" in ruff_out

    logger.info("Running black check...")
    black_out = execute_command_tool.invoke(
        {"command": "black --check --quiet .", "workdir": workdir}
    )
    results["black"] = black_out
    black_passed = "exit code: 0" in black_out

    logger.info("Running mypy check...")
    mypy_out = execute_command_tool.invoke(
        {"command": "mypy . --ignore-missing-imports", "workdir": workdir}
    )
    results["mypy"] = mypy_out
    mypy_passed = "exit code: 0" in mypy_out or "Success" in mypy_out

    logger.info("Running lint check...")
    lint_out = execute_command_tool.invoke(
        {"command": "ruff check .", "workdir": workdir}
    )
    results["lint"] = lint_out
    lint_passed = "exit code: 0" in lint_out or "No errors" in lint_out

    all_passed = ruff_passed and black_passed and mypy_passed and lint_passed
    results["passed"] = all_passed

    return results


async def develop_phase(state: AgentState) -> dict:
    """Execute the development phase.

    Designs the project skeleton, generates AGENTS.md, fills code based on
    the skeleton, and runs lightweight quality tests. In single-agent mode,
    the LLM directly generates code for each module defined in the skeleton.

    Args:
        state: The current AgentState.

    Returns:
        A dict of partial state updates including project_skeleton, light test
        results, phase_history, and logs.
    """
    logger.info("Entering development phase")

    skeleton = await _design_skeleton(state.task_template, state.tech_plan)
    logger.info(
        f"Project skeleton designed: {len(skeleton.get('directories', []))} dirs, "
        f"{len(skeleton.get('files', {}))} files"
    )

    workspace = settings.WORKSPACE_DIR
    os.makedirs(workspace, exist_ok=True)

    for dir_path in skeleton.get("directories", []):
        full_path = os.path.join(workspace, dir_path)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {full_path}")

    for file_path, content in skeleton.get("files", {}).items():
        full_path = os.path.join(workspace, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(str(content))
        logger.info(f"Created file: {full_path}")

    agents_md_path = os.path.join(workspace, "AGENTS.md")
    agents_content = skeleton.get("agents_md_content", "")
    if not agents_content:
        agents_content = _generate_agents_md()
    with open(agents_md_path, "w", encoding="utf-8") as f:
        f.write(agents_content)
    logger.info("AGENTS.md generated")

    logger.info("Running lightweight tests...")
    light_test_results = await _run_light_tests(workspace)

    return {
        "project_skeleton": skeleton,
        "phase_history": state.phase_history + ["develop"],
        "logs": state.logs
        + [
            f"[develop] Skeleton created with {len(skeleton.get('directories', []))} dirs",
            f"[develop] Light tests: {'PASSED' if light_test_results.get('passed') else 'FAILED'}",
        ],
    }
