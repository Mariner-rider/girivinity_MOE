from __future__ import annotations

import json
import logging
import multiprocessing
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from app.core.db import get_conn
from app.finetune.native_lora_trainer import NativeSelfTrainer
from app.finetune.update_gate import approve_model_update
from app.security.training_poison_guard import TrainingPoisonGuard

logger = logging.getLogger(__name__)


class SelfTrainer:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        st = cfg["modules"]["self_training"]
        tr = cfg["training"]
        self.db_path = Path(tr.get("queue_db") or "data/training.db")
        self.interval_s = int(st["interval_seconds"])
        self.threshold = int(st["chunk_threshold"])
        self.base_model = Path(st["base_model_path"])
        self.adapters_dir = Path(st["adapters_dir"])
        self.training_dir = Path(st["training_queue_dir"])
        self.event_log = Path(st["event_log_path"])
        self.alerts_log = Path(st["alerts_log_path"])
        self.epochs = int(st["epochs"])
        self.lr = float(st["learning_rate"])
        self.loss_abort = float(st["loss_abort_threshold"])
        self._init_db()

    def queue(self, query: str, chunks: list[dict]) -> None:
        """Write chunks to PostgreSQL queue. Fast — never blocks caller."""
        ts = datetime.utcnow().isoformat()
        with get_conn() as conn:
            with conn.cursor() as db:
                db.executemany(
                    "INSERT INTO training_queue "
                    "(query, chunk_text, url, score, timestamp, status) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (
                            query,
                            c.get("text", ""),
                            c.get("url", ""),
                            c.get("score", 0.0),
                            ts,
                            "pending",
                        )
                        for c in chunks
                    ],
                )

    @classmethod
    def start(cls) -> multiprocessing.Process:
        """Launch the daemon process. Call once at FastAPI startup."""
        instance = cls()
        p = multiprocessing.Process(target=instance._run_daemon, daemon=True)
        p.start()
        Path(".self_trainer.pid").write_text(str(p.pid))
        logger.info("SelfTrainer daemon started PID=%s", p.pid)
        return p

    def _run_daemon(self) -> None:
        logger.info("SelfTrainer running, interval=%ds", self.interval_s)
        while True:
            try:
                self._check_and_train()
            except Exception as exc:
                logger.error("SelfTrainer error: %s", exc)
            time.sleep(self.interval_s)

    def _check_and_train(self) -> None:
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute("SELECT COUNT(*) FROM training_queue WHERE status='pending'")
                count = db.fetchone()[0]

        if count < self.threshold:
            logger.info("SelfTrainer: %d pending < threshold %d, skipping", count, self.threshold)
            return

        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute("SELECT id, query, chunk_text, url FROM training_queue WHERE status='pending'")
                rows = db.fetchall()

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self.training_dir / f"{ts}.jsonl"

        # --- G-06: Scan every chunk through TrainingPoisonGuard before writing ---
        guard = TrainingPoisonGuard()
        clean_rows: list = []
        quarantine_count = 0
        for row in rows:
            row_id, query, chunk_text, url = row
            scan = guard.scan_chunk(chunk_text, url or "", query or "")
            if scan.is_poisoned:
                quarantine_count += 1
                logger.warning(
                    "SelfTrainer: quarantined row id=%s url=%s signals=%s confidence=%.3f",
                    row_id, url, scan.signals, scan.confidence,
                )
            else:
                clean_rows.append(row)

        if quarantine_count:
            logger.warning(
                "SelfTrainer: %d/%d chunks quarantined by PoisonGuard",
                quarantine_count, len(rows),
            )

        if not clean_rows:
            logger.warning("SelfTrainer: all chunks quarantined — aborting training cycle")
            return

        with open(dataset_path, "w", encoding="utf-8") as f:
            for _, query, chunk_text, url in clean_rows:
                f.write(
                    json.dumps(
                        {
                            "instruction": f"What do you know about: {query}",
                            "response": chunk_text,
                            "source_url": url,
                        }
                    )
                    + "\n"
                )

        success = self._run_lora_update(dataset_path, ts)

        if success:
            ids = [r[0] for r in clean_rows]
            placeholders = ",".join(["%s"] * len(ids))
            with get_conn() as conn:
                with conn.cursor() as db:
                    db.execute(
                        f"UPDATE training_queue SET status='trained' WHERE id IN ({placeholders})",
                        ids,
                    )
            self._log_event(ts, len(clean_rows), "success")

    def _run_lora_update(self, dataset_path: Path, version: str) -> bool:
        """Fine-tune the NATIVE GirivinityModel with LoRA on attention +
        MoE experts + router (no PEFT, no HuggingFace). Same contract as
        before: train on the JSONL, save a versioned adapter, flip the
        'latest' symlink, honour loss_abort."""
        final_dir = self.base_model / "final"
        if not (final_dir / "model.pt").exists():
            logger.warning(
                "Base model not found at %s — skipping. Build it first with: "
                "python -m app.training.pretrain --config config.yaml",
                final_dir,
            )
            return False

        try:
            adapter_out = self.adapters_dir / version
            latest_link = self.adapters_dir / "latest"

            trainer = NativeSelfTrainer(
                base_model_path=str(final_dir),
                adapter_dir=str(adapter_out),
                lr=self.lr,
                epochs=self.epochs,
                resume_from=str(latest_link.resolve())
                if latest_link.is_symlink() else None,
            )
            avg_loss = trainer.train_on_jsonl(str(dataset_path))

            if avg_loss > self.loss_abort:
                self._log_alert(version, avg_loss)
                return False

            # --- G-06: Quality gate — require loss improvement before promoting adapter ---
            try:
                gate_passed = approve_model_update(
                    baseline_score=self.loss_abort,
                    candidate_score=avg_loss,
                    benchmark_name="training_loss",
                    min_delta=0.0,
                )
            except Exception as gate_exc:
                logger.warning("SelfTrainer: update_gate raised %s — blocking promotion", gate_exc)
                gate_passed = False

            if not gate_passed:
                logger.warning(
                    "SelfTrainer: adapter promotion blocked by quality gate (avg_loss=%.4f)",
                    avg_loss,
                )
                return False

            if latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(adapter_out.resolve())
            return True

        except Exception as exc:
            logger.error("LoRA update failed: %s", exc)
            return False
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS training_queue (
                        id         SERIAL PRIMARY KEY,
                        query      TEXT    NOT NULL,
                        chunk_text TEXT    NOT NULL,
                        url        TEXT    DEFAULT '',
                        score      REAL    DEFAULT 0.0,
                        timestamp  TEXT    NOT NULL,
                        status     TEXT    DEFAULT 'pending'
                    )
                """
                )

    def _log_event(self, ts: str, count: int, status: str) -> None:
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.event_log, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": ts,
                        "chunks_trained": count,
                        "adapter_version": ts,
                        "status": status,
                    }
                )
                + "\n"
            )

    def _log_alert(self, ts: str, loss: float) -> None:
        self.alerts_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.alerts_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": ts, "alert": "loss_abort", "loss": round(loss, 4)}) + "\n")