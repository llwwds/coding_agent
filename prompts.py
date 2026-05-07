"""
Prompt templates module.

Defines all ChatPromptTemplate instances used throughout the agent's phases.
Each template includes a system message (role definition + output format)
and a human message (dynamic input variables).
"""

from langchain_core.prompts import ChatPromptTemplate

REQUIREMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a senior software architect and requirements analyst. "
                "Analyze the user's requirement and produce a structured analysis. "
                "Identify: (1) technology choices, (2) functional boundaries, "
                "(3) potential risks, (4) dependencies.\n\n"
                "Output format (JSON):\n"
                '{{"summary": "...", "tech_stack": ["..."], "features": ["..."], '
                '"boundaries": "...", "risks": ["..."], "dependencies": ["..."], '
                '"questions_for_clarification": ["..."]}}'
            ),
        ),
        ("human", "Analyze the following requirement:\n{requirement}"),
    ]
)

CLARIFY_REQUIREMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a requirements analyst helping to clarify ambiguous requirements. "
                "Ask targeted questions to resolve ambiguities and refine the scope. "
                "Based on the user's response, produce an updated clarified requirement.\n\n"
                "Output format (JSON):\n"
                '{{"clarified_requirement": "...", "remaining_ambiguities": ["..."], '
                '"is_clear": true/false}}'
            ),
        ),
        (
            "human",
            (
                "Original requirement: {original_requirement}\n"
                "Previous analysis: {analysis}\n"
                "User clarification response: {user_response}"
            ),
        ),
    ]
)

TASK_TEMPLATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a technical project manager. Based on the clarified requirement "
                "and technical plan, generate a structured task template. The template must "
                "include: input specification, target files, test commands, and acceptance "
                "criteria for each task.\n\n"
                "Output format (JSON):\n"
                '{{"tasks": [{{"id": "...", "name": "...", "input": "...", '
                '"target_files": ["..."], "test_commands": ["..."], '
                '"acceptance_criteria": ["..."]}}], "overall_acceptance": ["..."]}}'
            ),
        ),
        (
            "human",
            (
                "Clarified requirement: {clarified_requirement}\n"
                "Technical plan: {tech_plan}"
            ),
        ),
    ]
)

SKELETON_DESIGN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a software architect designing a project skeleton. "
                "Define the complete directory structure, configuration files, "
                "main entry points, test directory, and logging setup. "
                "Also define content for AGENTS.md (role, code style, commit conventions, "
                "test requirements).\n\n"
                "Output format (JSON):\n"
                '{{"directories": ["..."], "files": {{"path": "content"}}, '
                '"agents_md_content": "...", "module_coupling_plan": "...", '
                '"estimated_lines": {{"module": count}}, '
                '"estimated_dev_time": {{"module": "hours"}}, '
                '"test_complexity": "low/medium/high"}}'
            ),
        ),
        ("human", ("Task template: {task_template}\n" "Technical plan: {tech_plan}")),
    ]
)

CODE_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert software developer. Generate complete, production-ready "
                "code for the specified module. Follow best practices: include docstrings, "
                "type annotations, error handling, and follow the project's code style. "
                "Do NOT leave any placeholder or TODO comments. Output only the code, "
                "no explanations."
            ),
        ),
        (
            "human",
            (
                "Module to implement: {module_name}\n"
                "Specification: {specification}\n"
                "Project context: {context}\n"
                "Existing skeleton: {skeleton}"
            ),
        ),
    ]
)

LIGHT_TEST_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a QA engineer specializing in code quality checks. "
                "Analyze the test results from ruff, black, mypy, and lint checks. "
                "Determine if the code passes quality gates. If not, identify the specific "
                "issues that need fixing.\n\n"
                "Output format (JSON):\n"
                '{{"passed": true/false, "issues": [{{"file": "...", "line": n, '
                '"type": "format/lint/type", "message": "...", "severity": "error/warning"}}], '
                '"summary": "..."}}'
            ),
        ),
        (
            "human",
            (
                "Ruff output: {ruff_output}\n"
                "Black output: {black_output}\n"
                "Mypy output: {mypy_output}\n"
                "Lint output: {lint_output}"
            ),
        ),
    ]
)

FULL_TEST_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a senior QA lead. Analyze the full test suite results including "
                "unit tests, integration tests, and end-to-end tests. Determine overall "
                "pass/fail status and categorize any failures.\n\n"
                "Output format (JSON):\n"
                '{{"passed": true/false, "unit_test_results": {{...}}, '
                '"integration_test_results": {{...}}, "e2e_test_results": {{...}}, '
                '"coverage_percentage": n, "failure_categories": ["..."], '
                '"summary": "..."}}'
            ),
        ),
        (
            "human",
            (
                "Unit test output: {unit_output}\n"
                "Integration test output: {integration_output}\n"
                "E2E test output: {e2e_output}\n"
                "Coverage output: {coverage_output}"
            ),
        ),
    ]
)

FIX_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a debugging expert. Analyze the test failures and categorize them "
                "into one of these types:\n"
                "1. unread_required_file - Did not read necessary files\n"
                "2. wrong_code_modified - Modified incorrect/unrelated code\n"
                "3. test_not_executed - Tests were not properly run\n"
                "4. over_engineering - Overly complex or unnecessary design\n"
                "5. incomplete_fix - Fix did not fully address the issue\n\n"
                "Output format (JSON):\n"
                '{{"failure_type": "unread_required_file|wrong_code_modified|'
                'test_not_executed|over_engineering|incomplete_fix", '
                '"confidence": 0.0-1.0, "root_cause": "...", '
                '"affected_files": ["..."], "suggested_fix": "..."}}'
            ),
        ),
        (
            "human",
            (
                "Test failure details: {failure_details}\n"
                "Context of changes made: {change_context}"
            ),
        ),
    ]
)

FIX_STRATEGY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a senior developer creating a fix strategy. Based on the failure "
                "analysis, select and detail the appropriate fix strategy:\n"
                "- For test failures: repair the code\n"
                "- For coverage gaps: add missing tests\n"
                "- For type errors: fix type annotations\n"
                "- For over-engineering: simplify the code\n"
                "- For incomplete fixes: complete the fix\n\n"
                "Output format (JSON):\n"
                '{{"strategy": "repair_code|add_tests|fix_types|simplify|complete_fix", '
                '"specific_actions": [{{"action": "...", "file": "...", '
                '"description": "..."}}], "estimated_effort": "low/medium/high"}}'
            ),
        ),
        (
            "human",
            (
                "Failure analysis: {failure_analysis}\n"
                "Fix round number: {fix_round}\n"
                "Max fix rounds: {max_fix_rounds}"
            ),
        ),
    ]
)

DELIVER_REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a project delivery manager. Generate a comprehensive delivery "
                "report including performance metrics, project summary, and recommendations. "
                "Metrics to include: pass rate, first-pass rate, average fix rounds, "
                "mis-change rate.\n\n"
                "Output format (JSON):\n"
                '{{"project_summary": "...", "metrics": {{"pass_rate": n, '
                '"first_pass_rate": n, "avg_fix_rounds": n, "mis_change_rate": n}}, '
                '"test_summary": "...", "recommendations": ["..."], '
                '"delivered_files": ["..."]}}'
            ),
        ),
        (
            "human",
            (
                "Performance metrics: {performance_metrics}\n"
                "Test results: {test_results}\n"
                "Phase history: {phase_history}\n"
                "Failure analysis: {failure_analysis}"
            ),
        ),
    ]
)
