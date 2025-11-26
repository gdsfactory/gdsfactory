"""Tests for install module."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pygit2 import GitError, Repository, clone_repository

from gdsfactory.install import clone_or_update_repository


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestCloneOrUpdateRepository:
    """Test clone_or_update_repository function."""

    def test_clone_new_repository(self, temp_dir: Path) -> None:
        """Test cloning a new repository."""
        repo_path = temp_dir / "new_repo"
        test_url = "https://github.com/octocat/Hello-World.git"

        # Clone the repository
        clone_or_update_repository(test_url, repo_path)

        # Verify it was cloned
        assert repo_path.exists()
        assert (repo_path / ".git").exists()

        # Verify it's a valid git repository
        repo = Repository(repo_path)
        assert not repo.is_empty

    def test_update_existing_repository(self, temp_dir: Path) -> None:
        """Test updating an existing repository."""
        repo_path = temp_dir / "existing_repo"
        test_url = "https://github.com/octocat/Hello-World.git"

        # Clone the repository first
        clone_repository(test_url, str(repo_path))
        assert repo_path.exists()

        # Call clone_or_update_repository again
        clone_or_update_repository(test_url, repo_path)

        # Verify it still exists and is a valid repository
        assert repo_path.exists()
        repo = Repository(repo_path)
        assert not repo.is_empty

    def test_non_git_directory_raises_error(self, temp_dir: Path) -> None:
        """Test that a non-git directory raises an error."""
        non_git_path = temp_dir / "not_a_repo"
        non_git_path.mkdir()
        (non_git_path / "some_file.txt").write_text("not a git repo")

        test_url = "https://github.com/octocat/Hello-World.git"

        # Should raise ValueError
        with pytest.raises(ValueError, match="not a git repository"):
            clone_or_update_repository(test_url, non_git_path)

    def test_repository_with_no_remotes(self, temp_dir: Path) -> None:
        """Test handling repository with no remotes configured."""
        repo_path = temp_dir / "no_remote_repo"

        # Create a git repository without remotes
        from pygit2 import Signature, init_repository

        repo = init_repository(str(repo_path))

        # Create an initial commit
        tree = repo.TreeBuilder().write()
        author = Signature("Test User", "test@example.com")
        repo.create_commit(
            "refs/heads/main",
            author,
            author,
            "Initial commit",
            tree,
            [],
        )
        repo.set_head("refs/heads/main")

        test_url = "https://github.com/octocat/Hello-World.git"

        # Should not raise an error, just continue with existing state
        clone_or_update_repository(test_url, repo_path)

        # Verify repository still exists
        assert repo_path.exists()
        reopened_repo = Repository(repo_path)
        assert not reopened_repo.is_empty

    @patch("gdsfactory.install.Repository")
    def test_fetch_error_handling(self, mock_repo_class: Mock, temp_dir: Path) -> None:
        """Test error handling when fetch fails."""
        repo_path = temp_dir / "fetch_error_repo"
        repo_path.mkdir()

        # Create a mock repository that raises an error on fetch
        mock_repo = Mock()
        mock_remote = Mock()
        mock_remote.fetch.side_effect = GitError("Fetch failed")
        mock_repo.remotes = [mock_remote]
        mock_repo.head_is_unborn = False
        mock_repo_class.return_value = mock_repo

        test_url = "https://github.com/octocat/Hello-World.git"

        # Should not raise an error, just print and continue
        clone_or_update_repository(test_url, repo_path)

        # Verify fetch was attempted
        mock_remote.fetch.assert_called_once()
