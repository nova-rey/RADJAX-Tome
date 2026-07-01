from __future__ import annotations

from collections.abc import Sequence


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not headers:
        raise ValueError("markdown table requires at least one header")
    widths = [len(str(header)) for header in headers]
    rendered_rows = [[str(cell) for cell in row] for row in rows]
    for row in rendered_rows:
        if len(row) != len(headers):
            raise ValueError("markdown table row width does not match headers")
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(values: Sequence[object]) -> str:
        cells = [
            f" {str(value).ljust(widths[index])} " for index, value in enumerate(values)
        ]
        return "|" + "|".join(cells) + "|"

    separator = "|" + "|".join(f" {'-' * width} " for width in widths) + "|"
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rendered_rows)
    return "\n".join(lines)


def status_line(**fields: object) -> str:
    return " ".join(f"{key}={value}" for key, value in fields.items())
