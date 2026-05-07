# Coding Agent

## Role Definition

The Coding Agent is an autonomous single-agent system built on LangChain/LangGraph
that performs end-to-end software development tasks. Its responsibilities include:

- **Requirement Analysis**: Analyze user requirements, identify technical constraints,
  and generate structured task templates.
- **Code Generation**: Design project skeletons and generate production-ready code
  based on specifications.
- **Automated Testing**: Execute quality checks (linting, type checking, formatting)
  and comprehensive test suites (unit, integration, end-to-end).
- **Failure Analysis & Repair**: Categorize test failures into 5 types and apply
  targeted fix strategies with a bounded retry loop.
- **Delivery & Reporting**: Collect performance metrics (pass rate, first-pass rate,
  fix rounds, mis-change rate) and generate delivery reports.

### Capability Boundaries

- Operates within the designated workspace directory only
- Does not deploy to production environments
- Does not modify system-level configurations
- Requires explicit user input for requirement clarifications
- May require human intervention if fix rounds exceed threshold

## Code Style

### Naming Conventions
- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Modules**: lowercase with underscores

### Formatting
- All code formatted with `black` (line length 100)
- Consistent indentation (4 spaces, no tabs)
- Trailing newline at end of files

### Type Annotations
- All public functions must have complete type annotations
- Use `mypy` for static type checking (`mypy . --ignore-missing-imports`)
- Prefer explicit types over `Any`

### Docstrings
- Google-style docstrings for all public functions and classes
- Must include: `Args`, `Returns`, `Raises` sections
- Module-level docstrings describing purpose and behavior

### Imports
- Standard library first, then third-party, then local
- Alphabetical order within each group
- No wildcard imports (`from module import *`)

### Error Handling
- Use explicit exception types (never bare `except:`)
- Log exceptions with appropriate context
- Return meaningful error messages to the user

## Commit Conventions

### Format
```
<type>: <description>
```

### Types
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code restructuring without behavior change
- `test` - Adding or modifying tests
- `docs` - Documentation changes
- `chore` - Maintenance tasks
- `style` - Formatting, whitespace changes
- `perf` - Performance improvements

### Rules
- Commits must be atomic (single logical change)
- Do not commit secrets (.env, credentials, API keys)
- Do not commit generated artifacts (build outputs, caches)
- Commit messages should explain "why" not just "what"

### Examples
```
feat: add user authentication module
fix: resolve null pointer in config loader
test: add unit tests for state management
docs: update API reference with new endpoints
```

## Test Requirements

### Coverage
- Target: > 80% line coverage
- Measured with `pytest --cov=. --cov-report=term`

### Test Types
1. **Unit Tests**: Test individual functions and methods in isolation
2. **Integration Tests**: Test module interactions and data flow
3. **End-to-End Tests**: Test complete workflows from input to output

### Quality Gates
- `ruff check .` - Linting
- `black --check .` - Formatting
- `mypy . --ignore-missing-imports` - Type checking
- `pytest --tb=short` - Test execution

### Pre-commit Hooks
All quality checks must pass before code can be committed:
1. ruff (lint)
2. black (format)
3. mypy (type check)
4. pytest (tests)

### Test Independence
- Tests must not depend on execution order
- Each test should set up its own fixtures
- Use mocking for external dependencies
