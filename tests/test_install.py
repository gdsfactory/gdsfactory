"""Tests for install module."""

from pathlib import Path

import pytest
from pygit2 import Repository, clone_repository

from gdsfactory.install import clone_or_update_repository


class TestCloneOrUpdateRepository:
    """Test clone_or_update_repository function."""

    test_url: str = "https://github.com/octocat/Hello-World.git"

    def test_clone_new_repository(self, tmp_path: Path) -> None:
        """Test cloning a new repository."""
        repo_path = tmp_path / "new_repo"

        clone_or_update_repository(self.test_url, repo_path)

        assert repo_path.exists()
        assert (repo_path / ".git").exists()

        repo = Repository(repo_path)
        assert not repo.is_empty

    def test_update_existing_repository(self, tmp_path: Path) -> None:
        """Test updating an existing repository."""
        repo_path = tmp_path / "existing_repo"

        clone_repository(self.test_url, str(repo_path))
        assert repo_path.exists()

        clone_or_update_repository(self.test_url, repo_path)

        assert repo_path.exists()
        repo = Repository(repo_path)
        assert not repo.is_empty

    def test_non_git_directory_raises_error(self, tmp_path: Path) -> None:
        """Test that a non-git directory raises an error."""
        non_git_path = tmp_path / "not_a_repo"
        non_git_path.mkdir()
        (non_git_path / "some_file.txt").write_text("not a git repo")

        with pytest.raises(ValueError, match="not a git repository"):
            clone_or_update_repository(self.test_url, non_git_path)
