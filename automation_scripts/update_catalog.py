#!/usr/bin/env python3
"""Insert a new release into the root catalog.json for the QuPath Extension Manager.

The catalog follows the QuPath extension-catalog-model schema
(https://github.com/qupath/extension-catalog-model). This script prepends a new
release entry to the GelGenie extension's `releases` list, leaving older entries
in place so users on older QuPath versions can still resolve a compatible jar.

Intended to be called from the release workflow, e.g.:

    python automation_scripts/update_catalog.py \
        --version 2.0.0 \
        --repo mattaq31/GelGenie \
        --qupath-min v0.7.0

but it is fully runnable by hand for local testing. It is idempotent: re-running
with a version that already exists in the catalog updates that entry in place
rather than duplicating it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Repo-root catalog, resolved relative to this script (scripts/ -> repo root).
CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog.json"

# The jar base name is fixed by qupathExtension.name in build.gradle.kts.
JAR_BASENAME = "qupath-extension-gelgenie"
# Name of the extension entry within the catalog to update.
EXTENSION_NAME = "QuPath GelGenie extension"


def build_release(version: str, repo: str, qupath_min: str) -> dict:
    """Construct a single release entry matching the catalog schema."""
    tag = f"v{version}"
    base = f"https://github.com/{repo}/releases/download/{tag}"
    return {
        "name": tag,
        "main_url": f"{base}/{JAR_BASENAME}-{version}.jar",
        "required_dependency_urls": None,
        "optional_dependency_urls": None,
        "javadoc_urls": [f"{base}/{JAR_BASENAME}-{version}-javadoc.jar"],
        "version_range": {
            "min": qupath_min,
            "max": None,
            "excludes": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True,
                        help="Extension version without leading 'v', e.g. 2.0.0")
    parser.add_argument("--repo", required=True,
                        help="GitHub owner/repo that hosts the release, e.g. mattaq31/GelGenie")
    parser.add_argument("--qupath-min", default="v0.7.0",
                        help="Minimum compatible QuPath version (default: v0.7.0)")
    args = parser.parse_args()

    catalog = json.loads(CATALOG_PATH.read_text())

    extensions = catalog.get("extensions", [])
    ext = next((e for e in extensions if e.get("name") == EXTENSION_NAME), None)
    if ext is None:
        raise SystemExit(f"Could not find extension {EXTENSION_NAME!r} in {CATALOG_PATH}")

    release = build_release(args.version, args.repo, args.qupath_min)
    releases = ext.setdefault("releases", [])

    # Idempotent: replace an existing entry for this tag, else prepend the new one.
    existing = next((i for i, r in enumerate(releases) if r.get("name") == release["name"]), None)
    if existing is not None:
        releases[existing] = release
        print(f"Updated existing catalog entry for {release['name']}")
    else:
        releases.insert(0, release)
        print(f"Prepended new catalog entry for {release['name']}")

    CATALOG_PATH.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"Wrote {CATALOG_PATH}")


if __name__ == "__main__":
    main()
