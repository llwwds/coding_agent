# Coding Agent

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
