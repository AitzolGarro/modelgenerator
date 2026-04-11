"""
Standalone SQLite migration: add animation parameter columns to the jobs table.

Usage (from project root):
    python -m backend.scripts.migrate_anim_params [--db-path PATH]

Safe to run multiple times — each ALTER TABLE is wrapped in a try/except that
silently ignores "duplicate column name" errors from SQLite.
"""

import argparse
import sqlite3
from pathlib import Path


# Columns to add: (column_name, sql_type)
_NEW_COLUMNS: list[tuple[str, str]] = [
    ("num_frames",           "INTEGER"),
    ("anim_inference_steps", "INTEGER"),
    ("anim_guidance_scale",  "REAL"),
    ("anim_resolution",      "TEXT"),
    ("enhance_animation",    "INTEGER"),
    ("enhance_personality",  "TEXT"),
    ("enhance_intensity",    "REAL"),
]


def migrate(db_path: Path) -> None:
    """Add animation parameter columns to the jobs table (idempotent)."""
    print(f"Migrating database: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        for col_name, col_type in _NEW_COLUMNS:
            try:
                conn.execute(
                    f"ALTER TABLE jobs ADD COLUMN {col_name} {col_type}"
                )
                conn.commit()
                print(f"  ✓ Added column: {col_name} {col_type}")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" in str(exc).lower():
                    print(f"  · Skipping {col_name} (already exists)")
                else:
                    raise
    finally:
        conn.close()
    print("Migration complete.")


def _find_default_db() -> Path:
    """Locate the SQLite DB using the app's settings (if importable)."""
    try:
        from app.core.config import get_settings
        settings = get_settings()
        return settings.STORAGE_ROOT / "jobs.db"
    except Exception:
        pass
    # Fallback: look for the database relative to project root
    candidates = [
        Path("backend/storage/modelgenerator.db"),
        Path("storage/modelgenerator.db"),
        Path("data/jobs.db"),
        Path("backend/data/jobs.db"),
        Path("jobs.db"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("data/jobs.db")  # default even if not yet present


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add animation parameter columns to the jobs table."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to the SQLite database file (auto-detected if omitted)",
    )
    args = parser.parse_args()

    db_path = args.db_path or _find_default_db()
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}")
        raise SystemExit(1)

    migrate(db_path)


if __name__ == "__main__":
    main()
