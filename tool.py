"""
Custom toolset module.

Defines all LangChain tools used by the Coding Agent for file operations,
command execution, testing, version control, and logging.
"""

import os
import subprocess
from typing import Optional

from langchain_core.tools import tool


@tool
def read_file_tool(file_path: str, offset: int = 0, limit: Optional[int] = None) -> str:
    """Read the contents of a file at the specified path.

    Args:
        file_path: Absolute or relative path to the file to read.
        offset: Line number to start reading from (0-indexed).
        limit: Maximum number of lines to read. Reads all if None.

    Returns:
        The file contents as a string, or an error message if the file
        cannot be read.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if offset > 0:
            lines = lines[offset:]
        if limit is not None:
            lines = lines[:limit]
        return "".join(lines)
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"


@tool
def write_file_tool(file_path: str, content: str) -> str:
    """Write content to a file, creating directories if needed.

    Args:
        file_path: Absolute or relative path to the target file.
        content: The string content to write to the file.

    Returns:
        A success message or error description.
    """
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {file_path}"
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except OSError as e:
        return f"Error writing file {file_path}: {str(e)}"


@tool
def edit_file_tool(file_path: str, old_string: str, new_string: str) -> str:
    """Perform an exact string replacement in a file.

    Only the first occurrence of old_string is replaced. If old_string
    appears multiple times, an error is returned asking for more context.

    Args:
        file_path: Path to the file to edit.
        old_string: The exact text to find and replace.
        new_string: The replacement text.

    Returns:
        A success message or error description.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if old_string not in content:
            return f"Error: old_string not found in {file_path}"

        count = content.count(old_string)
        if count > 1:
            return (
                f"Error: Found {count} matches for old_string in {file_path}. "
                "Please provide more surrounding context to make it unique."
            )

        new_content = content.replace(old_string, new_string, 1)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Successfully edited {file_path}"

    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except Exception as e:
        return f"Error editing file {file_path}: {str(e)}"


@tool
def execute_command_tool(
    command: str, workdir: Optional[str] = None, timeout: int = 120
) -> str:
    """Execute a shell command and return the result.

    Args:
        command: The shell command to execute.
        workdir: Working directory for the command. Defaults to current directory.
        timeout: Maximum execution time in seconds (default: 120).

    Returns:
        Combined stdout and stderr output from the command.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        output += f"\n[exit code: {result.returncode}]"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds: {command}"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
def run_test_tool(test_type: str, workdir: str) -> str:
    """Run code quality and test tools on the project.

    Supported test_type values:
        - ruff: Run ruff linter
        - black: Run black formatter check
        - mypy: Run mypy type checker
        - pytest: Run pytest test suite
        - all: Run all of the above in sequence

    Args:
        test_type: The type of test to run (ruff/black/mypy/pytest/all).
        workdir: Working directory to run tests in.

    Returns:
        The combined output of the test commands.
    """
    commands = {
        "ruff": "ruff check .",
        "black": "black --check .",
        "mypy": "mypy .",
        "pytest": "pytest --tb=short",
    }

    if test_type == "all":
        results = []
        for name, cmd in commands.items():
            output = execute_command_tool.invoke({"command": cmd, "workdir": workdir})
            results.append(f"=== {name} ===\n{output}")
        return "\n\n".join(results)
    elif test_type in commands:
        return execute_command_tool.invoke(
            {"command": commands[test_type], "workdir": workdir}
        )
    else:
        return f"Error: Unknown test type '{test_type}'. Valid: ruff, black, mypy, pytest, all"


@tool
def git_commit_tool(message: str, files: Optional[str] = None) -> str:
    """Stage and commit changes to the git repository.

    Args:
        message: The commit message.
        files: Optional space-separated list of files to add. If None, adds all.

    Returns:
        Output from the git commands.
    """
    try:
        if files:
            add_result = execute_command_tool.invoke({"command": f"git add {files}"})
        else:
            add_result = execute_command_tool.invoke({"command": "git add ."})

        if "Error" in add_result:
            return add_result

        commit_result = execute_command_tool.invoke(
            {"command": f'git commit -m "{message}"'}
        )
        return f"Add result:\n{add_result}\nCommit result:\n{commit_result}"

    except Exception as e:
        return f"Error during git commit: {str(e)}"


@tool
def log_tool(module_name: str, level: str, message: str) -> str:
    """Record an operation log entry using the project's logger.

    Args:
        module_name: Name of the module generating the log (e.g., 'agent', 'develop').
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        message: The log message content.

    Returns:
        Confirmation message with the logged entry.
    """
    try:
        from logger import get_logger

        logger = get_logger(module_name)
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(message)
        return f"Logged [{level}] {module_name}: {message}"
    except Exception as e:
        return f"Error logging message: {str(e)}"


ALL_TOOLS = [
    read_file_tool,
    write_file_tool,
    edit_file_tool,
    execute_command_tool,
    run_test_tool,
    git_commit_tool,
    log_tool,
]
