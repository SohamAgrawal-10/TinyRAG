
from pathlib import Path
import re


def chunk_markdown(
    file_path: Path,
    root_dir: Path,
    max_lines: int = 10,
    overlap: int = 2,
) -> list[dict]:
    """Split Markdown into heading-aware chunks with source metadata."""
    if max_lines < 1:
        raise ValueError("max_lines must be at least 1")

    if overlap < 0 or overlap >= max_lines:
        raise ValueError("overlap must be between 0 and max_lines - 1")

    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    relative_path = file_path.relative_to(root_dir).as_posix()

    sections = []
    current_heading = None
    current_lines = []
    section_start = 1
    in_code_block = False

    for line_number, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block

        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)

        if heading_match and not in_code_block:
            if current_lines:
                sections.append(
                    (current_heading, section_start, current_lines)
                )

            current_heading = heading_match.group(2)
            current_lines = [line]
            section_start = line_number
        else:
            if not current_lines:
                section_start = line_number
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, section_start, current_lines))

    chunks = []

    for heading, start_line, section_lines in sections:
        start = 0

        while start < len(section_lines):
            end = min(start + max_lines, len(section_lines))
            content = "\n".join(section_lines[start:end]).strip()

            if content:
                absolute_start = start_line + start
                absolute_end = start_line + end - 1

                chunks.append({
                    "chunk_id": f"{relative_path}:{absolute_start:04d}",
                    "source": relative_path,
                    "heading": heading,
                    "start_line": absolute_start,
                    "end_line": absolute_end,
                    "text": content,
                })

            if end == len(section_lines):
                break

            start = end - overlap

    return chunks


def load_markdown(root_dir: Path) -> list[dict]:
    """Read and chunk every Markdown file under a directory."""
    root_dir = root_dir.resolve()
    chunks = []

    for file_path in sorted(root_dir.rglob("*.md")):
        if file_path.is_file():
            chunks.extend(chunk_markdown(file_path, root_dir))

    return chunks
