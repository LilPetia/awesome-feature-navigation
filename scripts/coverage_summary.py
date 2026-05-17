from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from dataclasses import dataclass
from json import dumps
from pathlib import Path


@dataclass(frozen=True)
class ModuleCoverage:
    name: str
    covered: int
    total: int

    @property
    def percent(self) -> float:
        if self.total == 0:
            return 100.0
        return self.covered / self.total * 100.0


def read_int(element: ET.Element, attribute: str) -> int:
    value = element.attrib.get(attribute, "0")
    return int(float(value))


def module_coverage(class_element: ET.Element) -> ModuleCoverage:
    filename = class_element.attrib.get("filename", class_element.attrib.get("name", "unknown"))
    lines_element = class_element.find("lines")
    if lines_element is None:
        return ModuleCoverage(filename, 0, 0)

    line_elements = list(lines_element.findall("line"))
    covered = sum(1 for line in line_elements if read_int(line, "hits") > 0)
    return ModuleCoverage(filename, covered, len(line_elements))


def build_summary(report_path: Path) -> str:
    coverage = read_coverage(report_path)
    covered, total, percent, modules = coverage

    lines = [
        "## Coverage",
        "",
        f"Covered lines: **{covered}/{total}**",
        f"Total coverage: **{percent:.1f}%**",
        "",
        "| Module | Covered lines | Coverage |",
        "| --- | ---: | ---: |",
    ]

    for module in modules[:10]:
        lines.append(f"| `{module.name}` | {module.covered}/{module.total} | {module.percent:.1f}% |")

    return "\n".join(lines) + "\n"


def read_coverage(report_path: Path) -> tuple[int, int, float, list[ModuleCoverage]]:
    root = ET.parse(report_path).getroot()
    total = read_int(root, "lines-valid")
    covered = read_int(root, "lines-covered")
    percent = 100.0 if total == 0 else covered / total * 100.0

    modules = [module_coverage(class_element) for class_element in root.findall(".//class")]
    modules = [module for module in modules if module.total > 0]
    modules.sort(key=lambda module: module.percent)

    return covered, total, percent, modules


def badge_color(percent: float) -> str:
    if percent >= 90:
        return "brightgreen"
    if percent >= 80:
        return "green"
    if percent >= 70:
        return "yellowgreen"
    if percent >= 60:
        return "yellow"
    if percent >= 50:
        return "orange"
    return "red"


def write_badge(report_path: Path, badge_path: Path) -> None:
    covered, total, percent, _ = read_coverage(report_path)
    badge_path.parent.mkdir(parents=True, exist_ok=True)
    badge = {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{percent:.1f}%",
        "color": badge_color(percent),
        "cacheSeconds": 300,
    }
    badge_path.write_text(dumps(badge, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote coverage badge to {badge_path}: {covered}/{total} lines, {percent:.1f}%")


def write_github_summary(summary: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path is None:
        return

    with Path(summary_path).open("a", encoding="utf-8") as file:
        file.write(summary)


def parse_args(argv: list[str]) -> tuple[Path, Path | None]:
    parser = ArgumentParser(description="Write a GitHub Actions coverage summary from coverage.xml.")
    parser.add_argument("report", nargs="?", default="coverage.xml")
    parser.add_argument("--badge", type=Path, help="Optional Shields endpoint JSON file to update.")
    args = parser.parse_args(argv[1:])
    return Path(args.report), args.badge


def main(argv: list[str]) -> int:
    report_path, badge_path = parse_args(argv)
    if not report_path.exists():
        print(f"Coverage report not found: {report_path}", file=sys.stderr)
        return 1

    summary = build_summary(report_path)
    print(summary)
    write_github_summary(summary)
    if badge_path is not None:
        write_badge(report_path, badge_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
