from pathlib import Path


def validate_expected_html_filenames(expected_html_files: list[str]) -> None:
    """Validate the expected HTML filenames before starting an agent run."""

    for filename in expected_html_files:
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Expected HTML filenames must be non-empty strings.")

        if filename != Path(filename).name or Path(filename).suffix.lower() not in {".html", ".htm"}:
            raise ValueError(f"Invalid expected HTML filename: {filename!r}")


def find_missing_html_artifacts(expected_html_files: list[str], generated_files: list[str]) -> list[str]:
    """Return expected HTML filenames that are missing or empty."""

    available_files: set[str] = set()

    for file_path in generated_files:
        path = Path(file_path)

        try:
            if path.is_file() and path.stat().st_size > 0:
                available_files.add(path.name)

        except OSError:
            continue

    return [filename for filename in expected_html_files if filename not in available_files]


