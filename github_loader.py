"""
GitHub Loader — ingests source files from any GitHub repo.
Filters by extension, respects size limits, handles pagination.
"""

import logging
from dataclasses import dataclass, field
from typing import Generator
from github import Github, GithubException
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings

logger = logging.getLogger(__name__)

# File types we care about for code understanding
SUPPORTED_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".rs", ".java", ".cpp", ".c", ".h",
    ".md", ".mdx", ".txt", ".yaml", ".yml",
    ".json", ".toml", ".env.example", ".sh",
    ".sql", ".graphql", ".proto",
}

MAX_FILE_SIZE_BYTES = 100_000  # skip files > 100KB (minified/generated)


@dataclass
class RepoFile:
    """Represents a single file ingested from GitHub."""
    path: str
    content: str
    repo: str
    sha: str
    url: str
    extension: str
    size_bytes: int
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "source": "github",
            "repo": self.repo,
            "path": self.path,
            "sha": self.sha,
            "url": self.url,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            **self.metadata,
        }


class GitHubLoader:
    def __init__(self, token: str = None, repo_name: str = None):
        self.token = token or settings.github_token
        self.repo_name = repo_name or settings.github_target_repo
        self.client = Github(self.token)
        self._repo = None

    @property
    def repo(self):
        if not self._repo:
            self._repo = self.client.get_repo(self.repo_name)
        return self._repo

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_file_content(self, file_content) -> str:
        """Decode file content with retry logic."""
        return file_content.decoded_content.decode("utf-8", errors="replace")

    def _is_supported(self, path: str, size: int) -> bool:
        """Filter out unsupported or oversized files."""
        if size > MAX_FILE_SIZE_BYTES:
            return False
        ext = "." + path.split(".")[-1] if "." in path else ""
        return ext in SUPPORTED_EXTENSIONS

    def stream_files(
        self,
        branch: str = None,
        path: str = "",
        max_files: int = None,
    ) -> Generator[RepoFile, None, None]:
        """
        Lazily stream files from the repo.
        Use as: for f in loader.stream_files(): process(f)
        """
        max_files = max_files or settings.max_files_per_ingest
        branch = branch or self.repo.default_branch
        count = 0

        logger.info(f"Starting ingestion: {self.repo_name} @ {branch}")

        try:
            contents = self.repo.get_contents(path, ref=branch)
        except GithubException as e:
            logger.error(f"Failed to get repo contents: {e}")
            return

        # BFS traversal of repo tree
        queue = list(contents)
        while queue and count < max_files:
            item = queue.pop(0)

            if item.type == "dir":
                try:
                    queue.extend(self.repo.get_contents(item.path, ref=branch))
                except GithubException:
                    logger.warning(f"Skipping dir {item.path}")
                continue

            if not self._is_supported(item.path, item.size):
                continue

            try:
                content = self._get_file_content(item)
                ext = "." + item.path.split(".")[-1] if "." in item.path else ""

                yield RepoFile(
                    path=item.path,
                    content=content,
                    repo=self.repo_name,
                    sha=item.sha,
                    url=item.html_url,
                    extension=ext,
                    size_bytes=item.size,
                )
                count += 1
                logger.debug(f"[{count}/{max_files}] Loaded: {item.path}")

            except Exception as e:
                logger.warning(f"Failed to load {item.path}: {e}")
                continue

        logger.info(f"Ingestion complete: {count} files loaded from {self.repo_name}")

    def get_repo_summary(self) -> dict:
        """Quick metadata about the target repo."""
        r = self.repo
        return {
            "name": r.full_name,
            "description": r.description,
            "language": r.language,
            "stars": r.stargazers_count,
            "default_branch": r.default_branch,
            "topics": r.get_topics(),
            "last_push": r.pushed_at.isoformat() if r.pushed_at else None,
        }
