"""Tests for tool.py - LangChain custom tools."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestReadFileTool:
    """Tests for the read_file_tool."""

    def test_read_existing_file(self):
        """Test reading an existing file."""
        from tool import read_file_tool

        content = "line1\nline2\nline3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            result = read_file_tool.invoke({"file_path": path})
            assert "line1" in result
            assert "line2" in result
            assert "line3" in result
        finally:
            os.unlink(path)

    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        from tool import read_file_tool

        result = read_file_tool.invoke({"file_path": "/nonexistent/file.txt"})
        assert "Error" in result

    def test_read_with_offset(self):
        """Test reading a file with line offset."""
        from tool import read_file_tool

        content = "line1\nline2\nline3\nline4\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            result = read_file_tool.invoke({"file_path": path, "offset": 2})
            assert "line3" in result
            assert "line1" not in result
        finally:
            os.unlink(path)

    def test_read_with_limit(self):
        """Test reading a file with line limit."""
        from tool import read_file_tool

        content = "line1\nline2\nline3\nline4\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            result = read_file_tool.invoke({"file_path": path, "limit": 2})
            lines = result.strip().split("\n")
            assert len(lines) == 2
        finally:
            os.unlink(path)


class TestWriteFileTool:
    """Tests for the write_file_tool."""

    def test_write_new_file(self):
        """Test writing content to a new file."""
        from tool import write_file_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_output.txt")
            result = write_file_tool.invoke(
                {"file_path": file_path, "content": "hello world"}
            )
            assert "Successfully wrote" in result
            assert os.path.exists(file_path)
            with open(file_path, "r") as f:
                assert f.read() == "hello world"

    def test_write_creates_directories(self):
        """Test that write creates parent directories if needed."""
        from tool import write_file_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "subdir", "nested", "file.txt")
            result = write_file_tool.invoke({"file_path": file_path, "content": "test"})
            assert "Successfully wrote" in result
            assert os.path.exists(file_path)

    def test_overwrite_existing_file(self):
        """Test overwriting an existing file."""
        from tool import write_file_tool

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("original")
            path = f.name

        try:
            write_file_tool.invoke({"file_path": path, "content": "updated"})
            with open(path, "r") as f:
                assert f.read() == "updated"
        finally:
            os.unlink(path)


class TestEditFileTool:
    """Tests for the edit_file_tool."""

    def test_edit_single_occurrence(self):
        """Test editing a file with a single match."""
        from tool import edit_file_tool

        content = "def foo():\n    return 1\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            result = edit_file_tool.invoke(
                {
                    "file_path": path,
                    "old_string": "return 1",
                    "new_string": "return 2",
                }
            )
            assert "Successfully edited" in result
            with open(path, "r") as f:
                assert "return 2" in f.read()
        finally:
            os.unlink(path)

    def test_edit_nonexistent_file(self):
        """Test editing a file that doesn't exist."""
        from tool import edit_file_tool

        result = edit_file_tool.invoke(
            {
                "file_path": "/nonexistent/edit.txt",
                "old_string": "a",
                "new_string": "b",
            }
        )
        assert "Error" in result

    def test_edit_multiple_matches(self):
        """Test editing when string appears multiple times."""
        from tool import edit_file_tool

        content = "x = 1\nx = 1\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            result = edit_file_tool.invoke(
                {
                    "file_path": path,
                    "old_string": "x = 1",
                    "new_string": "x = 2",
                }
            )
            assert "Found" in result and "matches" in result
        finally:
            os.unlink(path)


class TestExecuteCommandTool:
    """Tests for the execute_command_tool."""

    def test_execute_simple_command(self):
        """Test executing a simple shell command."""
        from tool import execute_command_tool

        result = execute_command_tool.invoke({"command": "echo hello"})
        assert "hello" in result
        assert "exit code: 0" in result

    def test_execute_command_with_failure(self):
        """Test executing a command that fails."""
        from tool import execute_command_tool

        result = execute_command_tool.invoke(
            {"command": "python -c 'import sys; sys.exit(1)'"}
        )
        assert "exit code: 1" in result

    def test_execute_command_with_workdir(self):
        """Test executing a command in a specific directory."""
        from tool import execute_command_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            result = execute_command_tool.invoke(
                {
                    "command": "pwd",
                    "workdir": tmpdir,
                }
            )
            assert tmpdir in result


class TestRunTestTool:
    """Tests for the run_test_tool."""

    def test_unknown_test_type(self):
        """Test running an unknown test type."""
        from tool import run_test_tool

        result = run_test_tool.invoke({"test_type": "unknown", "workdir": "."})
        assert "Error" in result

    def test_ruff_test(self):
        """Test running ruff check."""
        from tool import run_test_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "test.py")
            with open(py_file, "w") as f:
                f.write("x = 1\n")

            result = run_test_tool.invoke({"test_type": "ruff", "workdir": tmpdir})
            assert "ruff" in result.lower() or "exit code" in result


class TestLogTool:
    """Tests for the log_tool."""

    def test_log_info(self):
        """Test logging an info message."""
        from tool import log_tool

        result = log_tool.invoke(
            {
                "module_name": "test_module",
                "level": "INFO",
                "message": "test log message",
            }
        )
        assert "Logged" in result


class TestAllTools:
    """Tests for the ALL_TOOLS list."""

    def test_all_tools_list(self):
        """Test that ALL_TOOLS contains 7 tools."""
        from tool import ALL_TOOLS

        assert len(ALL_TOOLS) == 7
        tool_names = [t.name for t in ALL_TOOLS]
        assert "read_file_tool" in tool_names
        assert "write_file_tool" in tool_names
        assert "edit_file_tool" in tool_names
        assert "execute_command_tool" in tool_names
        assert "run_test_tool" in tool_names
        assert "git_commit_tool" in tool_names
        assert "log_tool" in tool_names
