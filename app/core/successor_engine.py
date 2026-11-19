from __future__ import annotations
import json
import logging
import multiprocessing
import time
from datetime import datetime
from pathlib import Path

import yaml

from app.core.db import get_conn
from app.core.model_generation_policy import GenerationPolicy

logger = logging.getLogger(__name__)


def _quantize_successor(weights_path: str, output_path: str, quant_type: str = "Q4_K_M") -> bool:
    try:
        from model.quantise import export_to_gguf

        export_to_gguf(weights_path, output_path, quant_type)
        gguf_path = Path(output_path) / "model.gguf"
        if gguf_path.exists():
            logger.info(
                "Successor quantization completed: weights=%s output=%s quant_type=%s",
                weights_path,
                output_path,
                quant_type,
            )
            return True
        logger.error(
            "Successor quantization did not produce GGUF: weights=%s output=%s",
            weights_path,
            output_path,
        )
        return False
    except Exception as exc:
        logger.error("Successor quantization failed: %s", exc)
        return False


class SuccessorEngine:
    def __init__(self) -> None:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        se = cfg["successor_engine"]
        tr = cfg["training"]
        self.check_interval_s = int(se["check_interval_seconds"])
        self.kb_threshold = int(se.get("knowledge_base_threshold", se.get("chunk_threshold", 100000)))
        self.quality_threshold = float(se.get("quality_score_threshold", se.get("quality_threshold", 3.5)))
        self.versions_dir = Path(se["versions_dir"])
        self.notifications_path = Path(se["notifications_path"])
        self.corpus_dir = Path(se.get("corpus_dir", se.get("training_root", "data/successor_training")))
        self.db_path = Path(tr.get("queue_db") or "data/training.db")
        self.active_link = Path(se.get("active_model_symlink", "models/active"))
        self.auto_quantize = bool(se.get("auto_quantize", True))
        self.quant_type = str(se.get("quant_type", "Q4_K_M"))
        self.generation_policy = GenerationPolicy()

    @classmethod
    def start(cls) -> multiprocessing.Process:
        instance = cls()
        p = multiprocessing.Process(target=instance._run_daemon, daemon=True)
        p.start()
        Path(".successor_engine.pid").write_text(str(p.pid))
        logger.info("SuccessorEngine daemon started PID=%s", p.pid)
        return p

    def log_feedback(self, user_id: str, score: float) -> None:
        """Called from chat endpoint when user rates a response (1-5)."""
        with get_conn() as conn:
            with conn.cursor() as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS feedback "
                    "(id SERIAL PRIMARY KEY, "
                    "user_id TEXT, score REAL, timestamp TEXT)"
                )
                db.execute(
                    "INSERT INTO feedback (user_id, score, timestamp) "
                    "VALUES (%s, %s, %s)",
                    (user_id, score, datetime.utcnow().isoformat()),
                )

    def get_notifications(self) -> list[dict]:
        if not self.notifications_path.exists():
            return []
        notifications = []
        with open(self.notifications_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    notifications.append(json.loads(line))
        return notifications

    def approve_successor(self, version: str) -> bool:
        version_dir = self.versions_dir / version
        if not version_dir.exists():
            logger.error("Version %s not found", version)
            return False
        self.active_link.parent.mkdir(parents=True, exist_ok=True)
        if self.active_link.is_symlink() or self.active_link.exists():
            self.active_link.unlink()
        self.active_link.symlink_to(version_dir.resolve())
        self._update_notification_status(version, "approved")
        logger.info("Successor %s approved and set as active", version)
        return True

    def reject_successor(self, version: str) -> bool:
        self._update_notification_status(version, "rejected")
        logger.info("Successor %s rejected", version)
        return True

    def _run_daemon(self) -> None:
        logger.info("SuccessorEngine running, interval=%ds", self.check_interval_s)
        while True:
            try:
                self._check_thresholds()
            except Exception as exc:
                logger.error("SuccessorEngine error: %s", exc)
            time.sleep(self.check_interval_s)

    def _check_thresholds(self) -> None:
        kb_count = self._count_trained_chunks()
        avg_score = self._rolling_quality_score()

        kb_trigger = kb_count >= self.kb_threshold
        quality_trigger = 0 < avg_score < self.quality_threshold

        if not (kb_trigger or quality_trigger):
            logger.info(
                "SuccessorEngine: kb=%d/%d quality=%.2f/%.2f — no trigger",
                kb_count,
                self.kb_threshold,
                avg_score,
                self.quality_threshold,
            )
            return

        reason = []
        if kb_trigger:
            reason.append(f"kb_chunks={kb_count}>={self.kb_threshold}")
        if quality_trigger:
            reason.append(f"quality={avg_score:.2f}<{self.quality_threshold}")
        next_generation = self.generation_policy.get_next_generation()
        if self.generation_policy._is_extrapolated(next_generation):
            logger.info(
                "Beyond defined roadmap — extrapolating next generation: %s",
                next_generation.name,
            )
        logger.info("SuccessorEngine triggered: %s", ", ".join(reason))

        self._build_successor()

    def _build_successor(self) -> None:
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        exported = self._export_corpus(version)
        if exported is None:
            return
        corpus_path, eval_path = exported

        version_dir = self.versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        success = self._run_full_training(corpus_path, version_dir)
        if not success:
            logger.error("Full training failed for version %s", version)
            return

        perplexity = self._evaluate(version_dir, eval_path)
        if perplexity == float("inf"):
            logger.error("Evaluation failed for version %s — discarding", version)
            return

        prev_version, prev_perplexity = self._get_current_model_stats()
        first_model = prev_perplexity == float("inf")

        if not first_model and perplexity >= prev_perplexity:
            logger.info(
                "New model perplexity %.2f not better than %.2f — discarding",
                perplexity,
                prev_perplexity,
            )
            return

        improvement = None
        if not first_model and prev_perplexity > 0:
            improvement = round((prev_perplexity - perplexity) / prev_perplexity * 100, 2)

        # Persist so the NEXT successor has a real baseline to beat.
        (version_dir / "perplexity.txt").write_text(str(perplexity))

        quantization_status = "quantization_skipped"
        if self.auto_quantize:
            quantized = _quantize_successor(
                weights_path=str(version_dir / "final"),
                output_path=str(version_dir),
                quant_type=self.quant_type,
            )
            quantization_status = "quantized" if quantized else "quantization_failed"

        self._write_notification(
            version=version,
            previous_version=prev_version,
            improvement_percent=improvement,
            trained_on_chunks=self._count_trained_chunks(),
            perplexity=perplexity,
            quantization_status=quantization_status,
        )

    def _export_corpus(self, version: str) -> tuple[Path, Path] | None:
        corpus_dir = self.corpus_dir / version
        corpus_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = corpus_dir / "corpus.jsonl"
        eval_path = corpus_dir / "eval.jsonl"
        try:
            with get_conn() as conn:
                with conn.cursor() as db:
                    db.execute("SELECT query, chunk_text FROM training_queue WHERE status='trained'")
                    rows = db.fetchall()
        except Exception:
            logger.warning("training_queue table not found")
            return None
        if not rows:
            logger.warning("No trained chunks to export")
            return None

        records = [
            {"instruction": f"What do you know about: {q}", "response": c}
            for q, c in rows
        ]
        # Hold out the most recent 10% as the perplexity evaluation set.
        eval_count = max(1, int(len(records) * 0.1)) if len(records) >= 20 else 0
        eval_records = records[-eval_count:] if eval_count else []
        train_records = records[:-eval_count] if eval_count else records

        with open(corpus_path, "w", encoding="utf-8") as f:
            for rec in train_records:
                f.write(json.dumps(rec) + "\n")
        with open(eval_path, "w", encoding="utf-8") as f:
            for rec in eval_records:
                f.write(json.dumps(rec) + "\n")
        logger.info(
            "Exported %d train / %d eval chunks to %s",
            len(train_records), len(eval_records), corpus_dir,
        )
        return corpus_path, eval_path

    def _run_full_training(self, corpus_path: Path, output_dir: Path) -> bool:
        try:
            import torch
            from model.architecture import GirivinityConfig
            from model.train import train

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            on_gpu = device.type == "cuda"
            if on_gpu:
                logger.info("Training on GPU: %s", torch.cuda.get_device_name(0))
                scaler = torch.cuda.amp.GradScaler()
                _ = scaler
            else:
                logger.info("Training on CPU — this will be slow, consider a GPU instance")

            # Build the NEXT generation's architecture. If the current model
            # is dense this enables the default MoE preset; if it is already
            # MoE this doubles the expert count (grow_experts path), unless
            # the roadmap entry for the next generation overrides the shape.
            current_cfg = GirivinityConfig.from_yaml()
            successor_cfg = self.generation_policy.successor_config(current_cfg)
            logger.info(
                "Successor architecture: dim=%d layers=%d moe=%s routed=%s activated=%s shared=%s",
                successor_cfg.dim,
                successor_cfg.n_layers,
                successor_cfg.moe_enabled,
                successor_cfg.n_routed_experts,
                successor_cfg.n_activated_experts,
                successor_cfg.n_shared_experts,
            )

            try:
                train(
                    data_path=str(corpus_path),
                    tokeniser_path="models/tokeniser/tokeniser.json",
                    output_dir=str(output_dir),
                    epochs=3,
                    batch_size=4,
                    lr=3e-4,
                    grad_accum=8,
                    model_config=successor_cfg,
                )
                return True
            finally:
                if on_gpu:
                    torch.cuda.empty_cache()
        except Exception as exc:
            logger.error("Full training error: %s", exc)
            return False

    def _evaluate(self, model_dir: Path, eval_path: Path | None = None) -> float:
        """Perplexity on the held-out eval split written by _export_corpus."""
        import math

        import torch
        import torch.nn.functional as F
        from tokenizers import Tokenizer

        from model.architecture import GirivinityConfig, GirivinityModel

        weights = model_dir / "final" / "model.pt"
        if not weights.exists():
            return float("inf")
        if eval_path is None or not Path(eval_path).exists():
            logger.error("No eval split found — cannot compute perplexity")
            return float("inf")

        cfg_path = model_dir / "final" / "config.yaml"
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text()) or {}
            fields = {k: v for k, v in data.items() if k in GirivinityConfig.__dataclass_fields__}
            cfg = GirivinityConfig(**fields)
        else:
            cfg = GirivinityConfig.from_yaml()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GirivinityModel(cfg).to(device)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.eval()

        tok = Tokenizer.from_file("models/tokeniser/tokeniser.json")
        samples: list[list[int]] = []
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = (
                    f"### Instruction:\n{rec['instruction']}\n\n"
                    f"### Response:\n{rec['response']}"
                )
                ids = tok.encode(text).ids[:512]
                if len(ids) > 8:
                    samples.append(ids)
        if not samples:
            return float("inf")

        losses: list[float] = []
        with torch.no_grad():
            for ids in samples[:256]:
                x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
                output = model(x)
                logits = output[0]
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
                losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        perplexity = math.exp(min(avg_loss, 20.0))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Eval perplexity on %d samples: %.4f", len(losses), perplexity)
        return perplexity

    def _count_trained_chunks(self) -> int:
        try:
            with get_conn() as conn:
                with conn.cursor() as db:
                    db.execute("SELECT COUNT(*) FROM training_queue WHERE status='trained'")
                    return db.fetchone()[0]
        except Exception:
            return 0

    def _rolling_quality_score(self) -> float:
        try:
            with get_conn() as conn:
                with conn.cursor() as db:
                    db.execute(
                        "SELECT AVG(score) FROM "
                        "(SELECT score FROM feedback "
                        " ORDER BY id DESC LIMIT 100) recent_feedback"
                    )
                    row = db.fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception:
            return 0.0

    def _get_current_model_stats(self) -> tuple[str, float]:
        if not self.active_link.exists():
            return "none", float("inf")
        try:
            name = self.active_link.resolve().name
            perplexity_file = self.active_link / "perplexity.txt"
            if perplexity_file.exists():
                return name, float(perplexity_file.read_text().strip())
            return name, float("inf")
        except Exception:
            return "unknown", float("inf")

    def _write_notification(
        self,
        version: str,
        previous_version: str,
        improvement_percent: float,
        trained_on_chunks: int,
        perplexity: float,
        quantization_status: str = "quantization_failed",
    ) -> None:
        self.notifications_path.parent.mkdir(parents=True, exist_ok=True)
        notification = {
            "type": "successor_ready",
            "version": version,
            "previous_version": previous_version,
            "improvement_percent": improvement_percent,
            "trained_on_chunks": trained_on_chunks,
            "perplexity": round(perplexity, 4),
            "quantization_status": quantization_status,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "awaiting_admin_approval",
        }
        with open(self.notifications_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(notification) + "\n")
        logger.info("Notification written for version %s", version)

    def _update_notification_status(self, version: str, status: str) -> None:
        if not self.notifications_path.exists():
            return
        lines = self.notifications_path.read_text(encoding="utf-8").splitlines()
        updated = []
        for line in lines:
            try:
                n = json.loads(line)
                if n.get("version") == version:
                    n["status"] = status
                updated.append(json.dumps(n))
            except json.JSONDecodeError:
                updated.append(line)
        self.notifications_path.write_text("\n".join(updated) + "\n", encoding="utf-8")