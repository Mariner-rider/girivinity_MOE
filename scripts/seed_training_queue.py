"""Seed the PostgreSQL training queue (replaces the old SQLite seed step)."""
import datetime

from app.core.db import get_conn

SEEDS = [
    ("What is artificial intelligence",
     "Artificial intelligence is the simulation of human intelligence processes by machines."),
    ("How does machine learning work",
     "Machine learning enables systems to learn from experience without being explicitly programmed."),
    ("What is deep learning",
     "Deep learning uses artificial neural networks with representation learning."),
]


def main() -> None:
    ts = datetime.datetime.utcnow().isoformat()
    with get_conn() as conn:
        with conn.cursor() as db:
            db.executemany(
                "INSERT INTO training_queue (query, chunk_text, url, score, timestamp, status)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                [(q, c, "", 0.9, ts, "pending") for q, c in SEEDS],
            )
    print(f"Seeded {len(SEEDS)} training records into PostgreSQL")


if __name__ == "__main__":
    main()