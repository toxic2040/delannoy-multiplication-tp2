#!/usr/bin/env python3
"""Run the complete Delannoy theorem replay with fail-closed release checks."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SHA256SUMS"
COMPONENT_TIMEOUT_SECONDS = 900
COMPONENTS = (
    ("structural identities", "verify_structural.py", None),
    (
        "compact dimension step",
        "verify_dimension_step_compact.py",
        "results/dimension_step_compact_certificate.json",
    ),
    (
        "unbounded dimension step",
        "verify_dimension_step_unbounded.py",
        "results/dimension_step_unbounded_certificate.json",
    ),
    (
        "lower base strip",
        "verify_lower_strip.py",
        "results/lower_strip_certificate.json",
    ),
    (
        "upper base strip",
        "verify_upper_strip.py",
        "results/upper_strip_certificate.json",
    ),
)
CERTIFICATE_SCHEMAS = {
    "results/dimension_step_compact_certificate.json": "dimension-step-compact-v1",
    "results/dimension_step_unbounded_certificate.json": "dimension-step-unbounded-v1",
    "results/lower_strip_certificate.json": "lower-strip-e1-v1",
    "results/upper_strip_certificate.json": "delannoy-upper-strip-v1",
}
EXPECTED_MANIFEST_PATHS = tuple(
    sorted(
        {
            ".gitignore",
            "LICENSE",
            "LICENSE-CC-BY-4.0",
            "README.md",
            "requirements.txt",
            "verify_all.py",
            *(script for _, script, _ in COMPONENTS),
            *(certificate for _, _, certificate in COMPONENTS if certificate),
        }
    )
)
FORBIDDEN_PATH_MARKERS = (
    b"/home/",
    b"/Users/",
    b"../",
    b"~/",
)
MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  ([!-~]+)")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def inside_checkout(path: Path) -> bool:
    try:
        path.resolve().relative_to(HERE)
    except ValueError:
        return False
    return True


def ignored_inventory_path(relative: Path) -> bool:
    return bool(relative.parts) and relative.parts[0] in {
        ".git",
        ".ruff_cache",
        "__pycache__",
    }


def checkout_inventory() -> set[str]:
    inventory: set[str] = set()
    for path in HERE.rglob("*"):
        relative = path.relative_to(HERE)
        if ignored_inventory_path(relative):
            continue
        if path.is_symlink() or path.is_file():
            inventory.add(relative.as_posix())
    return inventory


def parse_manifest(text: str) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    failures: list[str] = []
    lines = text.splitlines()
    if not lines:
        return entries, ["manifest is empty"]
    if text and not text.endswith("\n"):
        failures.append("manifest lacks its final newline")
    for number, line in enumerate(lines, 1):
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            failures.append(f"malformed manifest line {number}")
            continue
        digest, name = match.groups()
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or name != pure.as_posix():
            failures.append(f"unsafe manifest path on line {number}: {name}")
            continue
        if name in entries:
            failures.append(f"duplicate manifest path: {name}")
            continue
        entries[name] = digest
    if list(entries) != sorted(entries):
        failures.append("manifest paths are not sorted")
    return entries, failures


def validate_manifest(text: str, check_inventory: bool = True) -> list[str]:
    entries, failures = parse_manifest(text)
    expected = set(EXPECTED_MANIFEST_PATHS)
    actual = set(entries)
    for missing in sorted(expected - actual):
        failures.append(f"manifest omits {missing}")
    for extra in sorted(actual - expected):
        failures.append(f"manifest contains unexpected path {extra}")
    for name in sorted(expected & actual):
        path = HERE / name
        if path.is_symlink():
            failures.append(f"release path is a symlink: {name}")
            continue
        if not path.is_file() or not inside_checkout(path):
            failures.append(f"missing or escaping release file: {name}")
            continue
        if sha256(path.read_bytes()) != entries[name]:
            failures.append(f"hash mismatch: {name}")
    if check_inventory:
        allowed = expected | {MANIFEST.name}
        for extra in sorted(checkout_inventory() - allowed):
            failures.append(f"unmanifested release file: {extra}")
    return failures


def validate_certificate(
    payload: bytes, expected_schema: str, script_name: str
) -> list[str]:
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"certificate is not valid UTF-8 JSON: {error}"]
    failures: list[str] = []
    if not isinstance(record, dict):
        failures.append("certificate root is not an object")
        return failures
    if record.get("schema") != expected_schema:
        failures.append(f"certificate schema is not {expected_schema}")
    if record.get("passed") is not True:
        failures.append("certificate does not record passed=true")
    gates = record.get("gates")
    if gates is not None and (
        not isinstance(gates, list)
        or not gates
        or not all(
            isinstance(gate, dict) and gate.get("passed") is True for gate in gates
        )
    ):
        failures.append("certificate contains a failed or malformed gate list")
    mutations = record.get("mutation_controls")
    if mutations is not None and not isinstance(mutations, dict):
        failures.append("certificate mutation controls are malformed")
    elif isinstance(mutations, dict):
        failed_boolean_controls = sorted(
            name
            for name, value in mutations.items()
            if isinstance(value, bool) and not value
        )
        if failed_boolean_controls:
            failures.append(
                "certificate records unrejected mutations: "
                + ", ".join(failed_boolean_controls)
            )
    producer = record.get("producer_sha256", record.get("source_sha256"))
    if producer is not None:
        actual_producer = sha256((HERE / script_name).read_bytes())
        if producer != actual_producer:
            failures.append("certificate producer hash does not match its script")
    claimed_body = record.get("body_sha256")
    if claimed_body is not None:
        body = dict(record)
        del body["body_sha256"]
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        if claimed_body != sha256(encoded):
            failures.append("certificate body hash is invalid")
    return failures


def preflight() -> tuple[str, list[str]]:
    failures: list[str] = []
    if len(sys.argv) != 1:
        failures.append("verify_all.py has no quick, smoke, or skip mode")
    try:
        sympy_version = version("sympy")
    except PackageNotFoundError:
        failures.append("required dependency is missing: sympy==1.14.0")
    else:
        if sympy_version != "1.14.0":
            failures.append(f"SymPy version is {sympy_version}, expected 1.14.0")
    try:
        manifest_text = MANIFEST.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        return "", [*failures, f"cannot read SHA256SUMS: {error}"]
    failures.extend(validate_manifest(manifest_text))
    for _, script_name, _ in COMPONENTS:
        try:
            content = (HERE / script_name).read_bytes()
        except OSError as error:
            failures.append(f"cannot read component {script_name}: {error}")
            continue
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in content:
                failures.append(
                    f"nonportable path marker {marker.decode('ascii')!r} in {script_name}"
                )
    return manifest_text, failures


def mutated_manifest_is_rejected(manifest_text: str) -> bool:
    entries, failures = parse_manifest(manifest_text)
    if failures or not entries:
        return False
    first_name = next(iter(entries))
    old = entries[first_name]
    replacement = ("0" if old[0] != "0" else "1") + old[1:]
    mutated = manifest_text.replace(
        f"{old}  {first_name}", f"{replacement}  {first_name}", 1
    )
    return bool(validate_manifest(mutated, check_inventory=False))


def run_component(
    label: str,
    script_name: str,
    certificate_name: str | None,
    temporary_root: Path,
) -> tuple[bytes | None, str | None]:
    print(f"\n=== {label}: {script_name} ===", flush=True)
    command = [sys.executable, "-I", "-B", str(HERE / script_name)]
    temporary_certificate: Path | None = None
    if certificate_name is not None:
        temporary_certificate = temporary_root / certificate_name
        temporary_certificate.parent.mkdir(parents=True, exist_ok=True)
        command.extend(("--output", str(temporary_certificate)))
    environment = {"LC_ALL": "C"}
    try:
        completed = subprocess.run(
            command,
            cwd=HERE,
            env=environment,
            check=False,
            timeout=COMPONENT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, f"{script_name} exceeded {COMPONENT_TIMEOUT_SECONDS} seconds"
    except OSError as error:
        return None, f"could not launch {script_name}: {error}"
    if completed.returncode != 0:
        return None, f"{script_name} exited {completed.returncode}"
    if temporary_certificate is None:
        return None, None
    try:
        return temporary_certificate.read_bytes(), None
    except OSError as error:
        return None, f"{script_name} did not produce its certificate: {error}"


def main() -> int:
    manifest_text, failures = preflight()
    if failures:
        print("VERIFY_ALL: PREFLIGHT FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    if not mutated_manifest_is_rejected(manifest_text):
        print("VERIFY_ALL: PREFLIGHT FAILED")
        print("  - manifest mutation was not rejected")
        return 1

    certificate_mutations_rejected = 0
    with tempfile.TemporaryDirectory(prefix="delannoy-replay-") as temporary:
        temporary_root = Path(temporary)
        for label, script_name, certificate_name in COMPONENTS:
            generated, failure = run_component(
                label, script_name, certificate_name, temporary_root
            )
            if failure is not None:
                print(f"VERIFY_ALL: {failure}")
                return 1
            if certificate_name is None:
                continue
            if generated is None:
                print(f"VERIFY_ALL: missing generated certificate for {script_name}")
                return 1
            schema_failures = validate_certificate(
                generated, CERTIFICATE_SCHEMAS[certificate_name], script_name
            )
            if schema_failures:
                print(f"VERIFY_ALL: invalid generated certificate {certificate_name}")
                for failure in schema_failures:
                    print(f"  - {failure}")
                return 1
            banked = (HERE / certificate_name).read_bytes()
            if generated != banked:
                print(f"VERIFY_ALL: rebuilt certificate differs: {certificate_name}")
                return 1
            record = json.loads(generated)
            record["passed"] = False
            mutated = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
            if validate_certificate(
                mutated, CERTIFICATE_SCHEMAS[certificate_name], script_name
            ):
                certificate_mutations_rejected += 1
            else:
                print(
                    f"VERIFY_ALL: certificate mutation was not rejected: {certificate_name}"
                )
                return 1

    print(
        "\nVERIFY_ALL: PASS "
        f"({len(COMPONENTS)} full components, {len(CERTIFICATE_SCHEMAS)} byte-stable "
        f"certificates, {1 + certificate_mutations_rejected} rejected release mutations)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
