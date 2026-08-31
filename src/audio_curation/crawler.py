"""Polite, provenance-first acquisition for explicitly licensed OpenSLR corpora."""
from __future__ import annotations

import hashlib
import json
import re
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup


USER_AGENT = "AudioDataCurationResearchBot/1.0 (contact: repository owner)"
MANIFEST_COLUMNS = (
    "id", "path", "speaker_id", "sentence", "locale", "source_split",
    "source_dataset", "source_page_url", "source_archive_url", "source_member",
    "source_license", "source_audio_sha256",
)


@dataclass(frozen=True)
class CrawlPolicy:
    user_agent: str = USER_AGENT
    min_delay_sec: float = 5.0
    timeout_sec: float = 60.0
    max_download_bytes: int = 600_000_000


class RobotsDenied(RuntimeError):
    """Raised when robots.txt does not permit a requested URL."""


class PoliteSession:
    """HTTP session that checks robots.txt and enforces per-host delays."""

    def __init__(self, policy: CrawlPolicy, sleep: Callable[[float], None] = time.sleep):
        self.policy = policy
        self.sleep = sleep
        self.session = requests.Session()
        self.session.headers["User-Agent"] = policy.user_agent
        self._robots: dict[str, tuple[RobotFileParser, str, float]] = {}
        self._last_request: dict[str, float] = {}

    @staticmethod
    def _origin(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _load_robots(self, url: str) -> tuple[RobotFileParser, str, float]:
        origin = self._origin(url)
        if origin in self._robots:
            return self._robots[origin]
        robots_url = f"{origin}/robots.txt"
        response = self.session.get(robots_url, timeout=self.policy.timeout_sec)
        response.raise_for_status()
        text = response.text
        parser = RobotFileParser(robots_url)
        parser.parse(text.splitlines())
        declared = parser.crawl_delay(self.policy.user_agent)
        if declared is None:
            declared = parser.crawl_delay("*")
        delay = max(self.policy.min_delay_sec, float(declared or 0.0))
        self._robots[origin] = (parser, text, delay)
        return self._robots[origin]

    def get(self, url: str, *, stream: bool = False) -> requests.Response:
        parser, _, delay = self._load_robots(url)
        if not parser.can_fetch(self.policy.user_agent, url):
            raise RobotsDenied(f"robots.txt disallows {url}")
        origin = self._origin(url)
        elapsed = time.monotonic() - self._last_request.get(origin, 0.0)
        if elapsed < delay:
            self.sleep(delay - elapsed)
        response = self.session.get(url, timeout=self.policy.timeout_sec, stream=stream)
        self._last_request[origin] = time.monotonic()
        response.raise_for_status()
        return response

    def robots_evidence(self) -> dict[str, dict]:
        return {
            origin: {
                "url": f"{origin}/robots.txt",
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "effective_delay_sec": delay,
                "text": text,
            }
            for origin, (_, text, delay) in self._robots.items()
        }


def parse_openslr_page(html: str, page_url: str) -> dict:
    """Parse an OpenSLR resource page without relying on brittle CSS positions."""
    soup = BeautifulSoup(html, "html.parser")
    def field(name: str) -> str:
        label = soup.find("b", string=re.compile(rf"^\s*{re.escape(name)}:\s*$", re.I))
        if label is None or label.parent is None:
            return ""
        value = label.parent.get_text(" ", strip=True)
        return re.sub(rf"^\s*{re.escape(name)}:\s*", "", value, flags=re.I).strip()
    links: list[dict[str, object]] = []
    by_filename: dict[str, dict[str, object]] = {}
    for anchor in soup.find_all("a", href=True):
        url = urljoin(page_url, anchor["href"])
        filename = PurePosixPath(urlparse(url).path).name
        if filename.endswith((".tar.gz", ".tgz")):
            if filename not in by_filename:
                item: dict[str, object] = {"filename": filename, "url": url, "mirrors": [url]}
                by_filename[filename] = item
                links.append(item)
            elif url not in by_filename[filename]["mirrors"]:
                by_filename[filename]["mirrors"].append(url)
    return {
        "identifier": field("Identifier"),
        "summary": field("Summary"),
        "category": field("Category"),
        "license": field("License"),
        "archives": links,
    }


def _safe_member(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    return not member.issym() and not member.islnk() and not path.is_absolute() and ".." not in path.parts


def _transcripts(archive: tarfile.TarFile) -> dict[str, str]:
    output: dict[str, str] = {}
    for member in archive.getmembers():
        if not _safe_member(member) or not member.isfile() or not member.name.endswith(".trans.txt"):
            continue
        handle = archive.extractfile(member)
        if handle is None:
            continue
        for line in handle.read().decode("utf-8").splitlines():
            clip_id, separator, sentence = line.partition(" ")
            if separator:
                output[clip_id] = sentence.strip()
    return output


def extract_librispeech_archive(
    archive_path: Path, output_dir: Path, *, max_clips: int | None, source: dict
) -> pd.DataFrame:
    """Safely extract a deterministic prefix of aligned FLAC clips."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        transcripts = _transcripts(archive)
        members = sorted(
            (m for m in archive.getmembers() if _safe_member(m) and m.isfile() and m.name.endswith(".flac")),
            key=lambda m: m.name,
        )
        if max_clips is not None:
            members = members[:max_clips]
        for member in members:
            clip_id = Path(member.name).stem
            if clip_id not in transcripts:
                continue
            target = output_dir / f"{clip_id}.flac"
            handle = archive.extractfile(member)
            if handle is None:
                continue
            payload = handle.read()
            target.write_bytes(payload)
            speaker_id = clip_id.split("-", 1)[0]
            rows.append({
                "id": clip_id,
                "path": str(target.resolve()),
                "speaker_id": speaker_id,
                "sentence": transcripts[clip_id],
                "locale": "en",
                "source_split": archive_path.name.removesuffix(".tar.gz"),
                "source_dataset": source["identifier"],
                "source_page_url": source["page_url"],
                "source_archive_url": source["archive_url"],
                "source_member": member.name,
                "source_license": source["license"],
                "source_audio_sha256": hashlib.sha256(payload).hexdigest(),
            })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def count_audio_members(archive_path: Path) -> int:
    """Count safe regular FLAC members before applying the extraction cap."""
    with tarfile.open(archive_path, "r:gz") as archive:
        return sum(
            _safe_member(member) and member.isfile() and member.name.endswith(".flac")
            for member in archive.getmembers()
        )


def download_file(client: PoliteSession, url: str, target: Path) -> tuple[int, str]:
    """Stream a bounded download and return byte count and SHA-256."""
    response = client.get(url, stream=True)
    declared = int(response.headers.get("content-length", "0") or 0)
    if declared > client.policy.max_download_bytes:
        raise ValueError(f"declared download size {declared} exceeds limit")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > client.policy.max_download_bytes:
                    raise ValueError("download exceeded byte limit")
                digest.update(chunk)
                handle.write(chunk)
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return size, digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
