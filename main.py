"""
GK web API: upload image(s), return brand predictions + chat-style text.

Contract is batch-shaped (`results[]`) so we can raise MAX_UPLOAD_FILES later
without changing clients much.

Run from repo root (BZAN583):
  pip install -r gk_web/backend/requirements.txt
  uvicorn gk_web.backend.main:app --reload --host 127.0.0.1 --port 8000

Or:
  python -m uvicorn gk_web.backend.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

logger = logging.getLogger("gk_web")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Raise when the UI supports multi-select / zip / folder upload.
MAX_UPLOAD_FILES = 1

# Input guardrails (BZAN 583 build.pdf: length limits, validation before heavy work).
MAX_IMAGE_PIXELS = 35_000_000  # decompression / pixel bomb guard (~35 MP)
# Tool-style guardrail: bounded request rate per client IP (local dev; put proxy in prod).
RATE_LIMIT_WINDOW_SEC = 60.0
RATE_LIMIT_MAX_REQUESTS = 40

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
STILL_SHOPPING = "Still shopping — we couldn't finish that look just yet. Try another photo in a moment."

ResultStatus = Literal["ok", "still_shopping", "error", "partial"]

_rate_lock = threading.Lock()
_rate_buckets: dict[str, list[float]] = defaultdict(list)
_db_lock = threading.Lock()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
RUN_TAG = os.getenv("GK_RUN_TAG", datetime.utcnow().strftime("%Y%m%d-%H%M%S")).strip()
TUNING_VERSION = os.getenv("GK_TUNING_VERSION", "").strip()
try:
    CHAT_REALLY_LOW_SIM_THRESHOLD = float(os.getenv("GK_CHAT_REALLY_LOW_SIM_THRESHOLD", "0.62"))
except ValueError:
    CHAT_REALLY_LOW_SIM_THRESHOLD = 0.62


def _ollama_enabled() -> bool:
    v = os.getenv("GK_OLLAMA_ENABLE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _ollama_quick_take(result: dict[str, Any]) -> str | None:
    """
    Optional local LLM gloss on top of retrieval output (Ollama). Does not change predictions.
    Set GK_OLLAMA_ENABLE=1 and run `ollama serve` with OLLAMA_MODEL pulled.
    """
    if not _ollama_enabled():
        return None
    base = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    model = (os.getenv("OLLAMA_MODEL", "llama3.2") or "llama3.2").strip()
    try:
        timeout = float(os.getenv("OLLAMA_TIMEOUT_SEC", "15"))
    except ValueError:
        timeout = 15.0

    brand = result.get("predicted_brand", "unknown")
    sim = result.get("similarity")
    same = result.get("same_photo")
    item = result.get("item_match")
    decision = result.get("decision_label", "")
    prompt = (
        "You explain fashion match results to shoppers.\n"
        "The app compared one user photo only to two scraped brand image catalogs "
        "(FrankiesBikinis and MirchiByKim), using similarity scores — not web search.\n\n"
        f"Facts: predicted_catalog_side={brand}, best_similarity={sim}, "
        f"same_photo={same}, similar_item_signal={item}, decision_label={decision}.\n\n"
        "Write exactly one short sentence (30 words max). Friendly. "
        "Do not claim the clothing is definitely that brand. "
        "Do not mention AI, APIs, embeddings, or Ollama. "
        "Output the sentence only, no quotes."
    )
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as e:
        logger.warning("Ollama quick-take skipped (%s): %s", base, e)
        return None
    text = (body.get("response") or "").strip()
    if not text:
        return None
    return " ".join(text.split())


def _client_ip(request: Request) -> str:
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def _rate_limit_check(client_ip: str) -> None:
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets[client_ip]
        cutoff = now - RATE_LIMIT_WINDOW_SEC
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Too many requests from this client. Please wait a minute and try again.",
            )
        bucket.append(now)


def _decoded_image_guard(path: Path, original_filename: str) -> dict[str, Any] | None:
    """
    Reject non-images and huge decoded bitmaps before TensorFlow work.
    Returns an error result dict, or None if OK.
    """
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    except ImportError:
        return None

    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
    except OSError:
        return {
            "filename": original_filename,
            "status": "error",
            "message": "Could not read a valid image file.",
            "predicted_brand": None,
            "raw": None,
            "detail_code": "invalid_image_decode",
        }
    except Exception:
        logger.exception("image decode guard %s", path)
        return {
            "filename": original_filename,
            "status": "error",
            "message": STILL_SHOPPING,
            "predicted_brand": None,
            "raw": None,
            "detail_code": "image_guard_exception",
        }

    pixels = w * h
    if pixels > MAX_IMAGE_PIXELS or w < 2 or h < 2:
        return {
            "filename": original_filename,
            "status": "error",
            "message": "Image dimensions are not supported (too large or too small).",
            "predicted_brand": None,
            "raw": None,
            "detail_code": "image_dimensions_rejected",
        }
    return None


def _ensure_repo_on_path() -> None:
    import sys

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_on_path()

import gen_ai_project.gk_two_brand_matcher as gk  # noqa: E402

app = FastAPI(title="GK Brand Matcher API", version="0.1.0")


def _db_enabled() -> bool:
    return bool(DATABASE_URL)


def _ensure_results_table() -> None:
    """Create Postgres table once (safe to call repeatedly)."""
    if not _db_enabled():
        return
    try:
        import psycopg

        ddl = """
        CREATE TABLE IF NOT EXISTS gk_prediction_results (
            id BIGSERIAL PRIMARY KEY,
            request_id TEXT NOT NULL,
            image_index INTEGER NOT NULL,
            run_tag TEXT NULL,
            tuning_version TEXT NULL,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            predicted_brand TEXT NULL,
            predicted_score DOUBLE PRECISION NULL,
            latency_ms DOUBLE PRECISION NOT NULL,
            feedback_correct BOOLEAN NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
                # Safe migration for existing tables created before run/tuning tracking.
                cur.execute("ALTER TABLE gk_prediction_results ADD COLUMN IF NOT EXISTS run_tag TEXT NULL;")
                cur.execute("ALTER TABLE gk_prediction_results ADD COLUMN IF NOT EXISTS tuning_version TEXT NULL;")
            conn.commit()
    except Exception:
        logger.exception("Failed to initialize Postgres table")


def _extract_predicted_score(raw: Any) -> float | None:
    """Map matcher payload to a score value we can trend in UI."""
    if not isinstance(raw, dict):
        return None
    sim = raw.get("similarity")
    if sim is None:
        return None
    try:
        return float(sim)
    except (TypeError, ValueError):
        return None


def _insert_result_row(
    request_id: str,
    image_index: int,
    filename: str,
    status: str,
    predicted_brand: str | None,
    predicted_score: float | None,
    latency_ms: float,
    run_tag: str,
    tuning_version: str,
) -> int | None:
    if not _db_enabled():
        return None
    try:
        import psycopg

        with _db_lock:
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO gk_prediction_results (
                            request_id, image_index, run_tag, tuning_version, filename, status, predicted_brand,
                            predicted_score, latency_ms
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            request_id,
                            image_index,
                            run_tag,
                            tuning_version,
                            filename,
                            status,
                            predicted_brand,
                            predicted_score,
                            latency_ms,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
        return int(row[0]) if row else None
    except Exception:
        logger.exception("Failed to insert result row")
        return None


def _read_recent_rows(limit: int = 200, run_tag: str | None = None) -> dict[str, Any]:
    if not _db_enabled():
        return {"rows": [], "working_accuracy": None, "feedback_count": 0}
    try:
        import psycopg

        where_sql = "WHERE run_tag = %s" if run_tag else ""
        row_params: tuple[Any, ...] = (run_tag, limit) if run_tag else (limit,)
        agg_params: tuple[Any, ...] = (run_tag,) if run_tag else ()

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        id, request_id, image_index, run_tag, tuning_version, filename, status, predicted_brand,
                        predicted_score, latency_ms, feedback_correct, created_at, updated_at
                    FROM gk_prediction_results
                    {where_sql}
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    row_params,
                )
                rows = cur.fetchall()
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) FILTER (WHERE feedback_correct IS NOT NULL) AS feedback_count,
                        AVG(CASE WHEN feedback_correct THEN 1.0 ELSE 0.0 END) AS working_accuracy
                    FROM gk_prediction_results
                    {where_sql}
                    """,
                    agg_params,
                )
                agg = cur.fetchone()
        return {
            "rows": [
                {
                    "id": int(r[0]),
                    "request_id": str(r[1]),
                    "image_index": int(r[2]),
                    "run_tag": r[3],
                    "tuning_version": r[4],
                    "filename": str(r[5]),
                    "status": str(r[6]),
                    "predicted_brand": r[7],
                    "predicted_score": float(r[8]) if r[8] is not None else None,
                    "latency_ms": float(r[9]),
                    "feedback_correct": r[10],
                    "created_at": r[11].isoformat() if r[11] else None,
                    "updated_at": r[12].isoformat() if r[12] else None,
                }
                for r in rows
            ],
            "feedback_count": int(agg[0] or 0) if agg else 0,
            "working_accuracy": float(agg[1]) if agg and agg[1] is not None else None,
        }
    except Exception:
        logger.exception("Failed to read Postgres rows")
        return {"rows": [], "working_accuracy": None, "feedback_count": 0}


def _set_feedback(row_id: int, correct: bool) -> bool:
    if not _db_enabled():
        return False
    try:
        import psycopg

        with _db_lock:
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE gk_prediction_results
                        SET feedback_correct = %s, updated_at = NOW()
                        WHERE id = %s
                        """,
                        (correct, row_id),
                    )
                    updated = cur.rowcount
                conn.commit()
        return bool(updated)
    except Exception:
        logger.exception("Failed to write feedback")
        return False


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _results_summary(run_tag: str | None = None) -> dict[str, Any]:
    if not _db_enabled():
        return {
            "db_enabled": False,
            "run_tag_filter": run_tag,
            "total_rows": 0,
            "feedback_count": 0,
            "working_accuracy": None,
            "latency_ms": {"p50": None, "p95": None, "mean": None},
            "by_predicted_brand": [],
            "available_run_tags": [],
        }

    try:
        import psycopg

        where_clause = "WHERE run_tag = %s" if run_tag else ""
        params: tuple[Any, ...] = (run_tag,) if run_tag else ()

        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT predicted_brand, latency_ms, feedback_correct
                    FROM gk_prediction_results
                    {where_clause}
                    ORDER BY id DESC
                    """,
                    params,
                )
                rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT DISTINCT run_tag
                    FROM gk_prediction_results
                    WHERE run_tag IS NOT NULL AND run_tag <> ''
                    ORDER BY run_tag DESC
                    """
                )
                tag_rows = cur.fetchall()
    except Exception:
        logger.exception("Failed to compute summary")
        return {
            "db_enabled": True,
            "run_tag_filter": run_tag,
            "error": "summary_query_failed",
        }

    total_rows = len(rows)
    latencies = sorted(float(r[1]) for r in rows if r[1] is not None)
    feedback_values = [r[2] for r in rows if r[2] is not None]
    feedback_count = len(feedback_values)
    working_accuracy = (
        float(sum(1 for x in feedback_values if bool(x)) / feedback_count)
        if feedback_count
        else None
    )

    brand_stats: dict[str, dict[str, Any]] = {}
    for predicted_brand, _lat, feedback_correct in rows:
        key = str(predicted_brand or "unknown")
        if key not in brand_stats:
            brand_stats[key] = {
                "predicted_brand": key,
                "count": 0,
                "feedback_count": 0,
                "correct_count": 0,
                "accuracy": None,
            }
        b = brand_stats[key]
        b["count"] += 1
        if feedback_correct is not None:
            b["feedback_count"] += 1
            if bool(feedback_correct):
                b["correct_count"] += 1

    for b in brand_stats.values():
        if b["feedback_count"] > 0:
            b["accuracy"] = float(b["correct_count"] / b["feedback_count"])

    by_predicted_brand = sorted(
        brand_stats.values(),
        key=lambda x: (-int(x["count"]), str(x["predicted_brand"])),
    )

    mean_latency = (sum(latencies) / len(latencies)) if latencies else None
    available_tags = [str(r[0]) for r in tag_rows if r and r[0]]

    return {
        "db_enabled": True,
        "run_tag_filter": run_tag,
        "total_rows": total_rows,
        "feedback_count": feedback_count,
        "working_accuracy": working_accuracy,
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "mean": float(mean_latency) if mean_latency is not None else None,
        },
        "by_predicted_brand": by_predicted_brand,
        "available_run_tags": available_tags,
    }


def _effective_tuning_version() -> str:
    if TUNING_VERSION:
        return TUNING_VERSION
    return (
        f"k{gk.BRAND_VOTE_TOP_K}"
        f"_m{gk.BRAND_VOTE_PER_BRAND_TOP_M}"
        f"_w{gk.BRAND_VOTE_MEAN_WEIGHT:.2f}-{gk.BRAND_VOTE_MAX_WEIGHT:.2f}"
        f"_countnorm{getattr(gk, 'BRAND_COUNT_NORM_EXP', 0):.2f}"
        f"_minority{getattr(gk, 'BRAND_MINORITY_SCORE_MULT', 1.0):.2f}"
    )

_ROOT_HELP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GK API (port 8000)</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 40rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
    code { background: #f4f4f4; padding: 0.15rem 0.35rem; border-radius: 4px; }
    a { color: #0645ad; }
    .box { border: 1px solid #ccc; border-radius: 8px; padding: 1rem; margin: 1rem 0; background: #fafafa; }
  </style>
</head>
<body>
  <h1>GK Brand Matcher — API server</h1>
  <p>This address (<strong>port 8000</strong>) is the <em>backend API only</em>. There is no full web app here, so the root URL used to show <code>{"detail":"Not Found"}</code> — that was expected.</p>
  <div class="box">
    <h2>Open the website (frontend)</h2>
    <p>From the project folder, run the Vite dev server, then open <strong>port 5173</strong>:</p>
    <ol>
      <li><code>cd gk_web/frontend</code></li>
      <li><code>npm install</code> (first time only)</li>
      <li><code>npm run dev</code></li>
      <li>Browser: <a href="http://127.0.0.1:5173">http://127.0.0.1:5173</a></li>
    </ol>
    <p>If you see <code>ERR_CONNECTION_REFUSED</code> on 5173, the Vite process is not running (or Node/npm is not installed).</p>
  </div>
  <div class="box">
    <h2>API links</h2>
    <ul>
      <li><a href="/docs">Interactive API docs (Swagger)</a></li>
      <li><a href="/api/health">GET /api/health</a></li>
      <li><a href="/api/capabilities">GET /api/capabilities</a></li>
      <li><code>POST /api/analyze</code> — multipart field <code>files</code> (used by the React app)</li>
    </ul>
    <p>Optional: set <code>GK_OLLAMA_ENABLE=1</code> with <a href="https://ollama.com" target="_blank" rel="noopener">Ollama</a> running locally to append a short “Quick take” sentence after each match (predictions still come from embeddings).</p>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root_help() -> str:
    """So opening http://127.0.0.1:8000/ explains the two-server dev setup."""
    return _ROOT_HELP_HTML


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj


def _display_brand_for_chat(brand: str) -> str:
    """Human-readable brand labels for chat copy (internal ids stay unchanged in JSON)."""
    mapping = {
        "FrankiesBikinis": "Frankies Bikinis",
        "MirchiByKim": "Mirchi by Kim",
    }
    return mapping.get(brand, brand)


def _assistant_message(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return str(result.get("user_message") or STILL_SHOPPING)
    brand = _display_brand_for_chat(str(result.get("predicted_brand") or "unknown"))
    sim = result.get("similarity")
    same = result.get("same_photo") is True
    item = result.get("item_match") is True
    sim_f = float(sim) if sim is not None else None
    sim_s = f"{sim_f:.4f}" if sim_f is not None else "n/a"

    # Copy is tuned to retrieval reality: we only ever compare to Frankies + Mirchi scraped
    # catalogs — high cosine similarity means "closest aesthetic neighbor," not proven provenance.
    parts: list[str] = []
    if same:
        parts.append(
            f"**Strong catalog alignment** — nearest indexed images point to **{brand}** "
            f"(similarity {sim_s}). Treat this as a confident match to something already in our scraped set."
        )
    elif item:
        parts.append(
            "**We haven’t shopped this exact piece yet**, but it visually resembles listings we’ve seen "
            f"more on the **{brand}** side of our closet (similarity {sim_s}). "
            "That’s a **similar-style hint**, not confirmation that the item is from that brand."
        )
    elif sim_f is not None and sim_f >= CHAT_REALLY_LOW_SIM_THRESHOLD:
        parts.append(
            f"**Closest catalog side currently leans {brand}** (similarity {sim_s}). "
            "We did not hit the softer same-item threshold, but this is still a directional style recommendation."
        )
    else:
        parts.append(
            "**Hmm — haven’t shopped that exact vibe yet.** "
            f"The nearest vibe among brands we track skews toward **{brand}** "
            f"(similarity {sim_s}) — use it only as a loose starting point."
        )
        parts.append(
            f"(Strict same-photo threshold is {float(result.get('threshold') or gk.SAME_PHOTO_THRESHOLD):.2f}; "
            f"softer same-item threshold is {float(result.get('item_match_threshold') or gk.ITEM_MATCH_THRESHOLD):.2f}.)"
        )

    decision = result.get("decision_label")
    if decision:
        parts.append(f"Signal: **{decision}**.")

    warn = result.get("catalog_warning")
    if warn:
        parts.append(str(warn))
    return " ".join(parts)


def _one_item_from_matcher(filename: str, result: dict[str, Any]) -> dict[str, Any]:
    """Normalize matcher dict -> one API `results[]` element."""
    safe_raw = _json_safe(result)
    if not result.get("ok"):
        return {
            "filename": filename,
            "status": "still_shopping",
            "message": str(result.get("user_message") or STILL_SHOPPING),
            "predicted_brand": result.get("predicted_brand"),
            "raw": safe_raw,
            "detail_code": result.get("error"),
        }
    msg = _assistant_message(result)
    quick = _ollama_quick_take(result)
    if quick:
        msg = f"{msg}\n\n**Quick take:** {quick}"
        safe_raw = dict(safe_raw) if isinstance(safe_raw, dict) else safe_raw
        if isinstance(safe_raw, dict):
            safe_raw["ollama_quick_take"] = quick

    return {
        "filename": filename,
        "status": "ok",
        "message": msg,
        "predicted_brand": result.get("predicted_brand"),
        "raw": safe_raw,
        "detail_code": None,
    }


def analyze_bytes(filename: str, raw: bytes) -> dict[str, Any]:
    """
    Run matcher on in-memory upload. Used by /api/analyze and available for
    future workers / queues without duplicating temp-file logic.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return {
            "filename": filename,
            "status": "error",
            "message": f"Unsupported file type. Use: {', '.join(sorted(ALLOWED_SUFFIXES))}",
            "predicted_brand": None,
            "raw": None,
            "detail_code": "bad_suffix",
        }

    if len(raw) > MAX_UPLOAD_BYTES:
        return {
            "filename": filename,
            "status": "error",
            "message": "File too large (max 15 MB).",
            "predicted_brand": None,
            "raw": None,
            "detail_code": "too_large",
        }

    tmp_name = f"gk_upload_{uuid.uuid4().hex}{suffix}"
    tmp_path = Path(tempfile.gettempdir()) / tmp_name

    try:
        tmp_path.write_bytes(raw)
    except OSError:
        logger.exception("write temp %s", filename)
        return {
            "filename": filename,
            "status": "error",
            "message": STILL_SHOPPING,
            "predicted_brand": None,
            "raw": None,
            "detail_code": "temp_write_failed",
        }

    bad = _decoded_image_guard(tmp_path, filename)
    if bad:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return bad

    try:
        result = gk.query_same_photo(tmp_path, threshold=gk.SAME_PHOTO_THRESHOLD)
    except Exception:
        logger.exception("query_same_photo %s", filename)
        return {
            "filename": filename,
            "status": "error",
            "message": STILL_SHOPPING,
            "predicted_brand": None,
            "raw": None,
            "detail_code": "matcher_exception",
        }
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return _one_item_from_matcher(filename, result)


def _aggregate_status(results: list[dict[str, Any]]) -> ResultStatus:
    statuses = [r.get("status") for r in results]
    if not statuses:
        return "error"
    ok_count = sum(1 for s in statuses if s == "ok")
    if ok_count == len(statuses):
        return "ok"
    if ok_count > 0:
        return "partial"
    if all(s == "still_shopping" for s in statuses):
        return "still_shopping"
    return "error"


def _chat_summary(results: list[dict[str, Any]]) -> str:
    """Single string for the chat bubble; scales to many results."""
    if len(results) == 1:
        return str(results[0].get("message") or "")
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        name = r.get("filename") or f"image_{i}"
        lines.append(f"**Photo {i}** ({name}): {r.get('message', '')}")
    return "\n\n".join(lines)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    """Public limits and safety knobs for clients (avoids hard-coding duplicates)."""
    return {
        "schema_version": 1,
        "max_upload_files": MAX_UPLOAD_FILES,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
        "max_image_pixels": MAX_IMAGE_PIXELS,
        "rate_limit_per_minute": RATE_LIMIT_MAX_REQUESTS,
        "run_tag": RUN_TAG,
        "tuning_version": _effective_tuning_version(),
        "chat_really_low_similarity_threshold": CHAT_REALLY_LOW_SIM_THRESHOLD,
        "operations": ["analyze_image"],
        "ollama_quick_take": {
            "enabled": _ollama_enabled(),
            "url": os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            "model": (os.getenv("OLLAMA_MODEL", "llama3.2") or "llama3.2").strip(),
        },
    }


@app.on_event("startup")
def _startup_init() -> None:
    _ensure_results_table()


@app.get("/api/results")
def list_results(limit: int = 200, run_tag: str | None = None) -> dict[str, Any]:
    capped = max(1, min(limit, 1000))
    tag = run_tag.strip() if run_tag else None
    payload = _read_recent_rows(capped, run_tag=tag)
    payload["db_enabled"] = _db_enabled()
    return payload


@app.get("/api/results/summary")
def results_summary(run_tag: str | None = None) -> dict[str, Any]:
    return _results_summary(run_tag=run_tag.strip() if run_tag else None)


@app.post("/api/results/{row_id}/feedback")
def set_result_feedback(row_id: int, body: dict[str, Any]) -> dict[str, Any]:
    if "correct" not in body:
        raise HTTPException(status_code=400, detail="Missing boolean `correct` in request body.")
    correct = bool(body["correct"])
    ok = _set_feedback(row_id=row_id, correct=correct)
    if not ok:
        raise HTTPException(
            status_code=404 if _db_enabled() else 400,
            detail="Could not update feedback (row missing or DATABASE_URL not configured).",
        )
    payload = _read_recent_rows(200)
    payload["db_enabled"] = _db_enabled()
    return payload


@app.post("/api/analyze")
async def analyze(request: Request, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """
    Multipart form field `files` (repeat the field for each image).

    Today MAX_UPLOAD_FILES is 1; the response is still `results: [...]`
    so clients can loop without a special case.
    """
    _rate_limit_check(_client_ip(request))

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded (use form field `files`).")

    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"At most {MAX_UPLOAD_FILES} image(s) per request for now. "
                    "Multi-image batches will use this same endpoint."
                ),
                "max_upload_files": MAX_UPLOAD_FILES,
                "received": len(files),
            },
        )

    request_id = uuid.uuid4().hex
    tuning_version = _effective_tuning_version()
    results: list[dict[str, Any]] = []
    for upload in files:
        fn = upload.filename or "upload"
        if not upload.filename:
            results.append(
                {
                    "filename": fn,
                    "status": "error",
                    "message": "Missing filename.",
                    "predicted_brand": None,
                    "raw": None,
                    "detail_code": "missing_filename",
                }
            )
            continue

        try:
            raw = await upload.read()
        except Exception:
            logger.exception("read upload %s", fn)
            results.append(
                {
                    "filename": fn,
                    "status": "error",
                    "message": STILL_SHOPPING,
                    "predicted_brand": None,
                    "raw": None,
                    "detail_code": "upload_read_failed",
                }
            )
            continue

        t0 = time.perf_counter()
        item = analyze_bytes(fn, raw)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        item["latency_ms"] = latency_ms
        item["predicted_score"] = _extract_predicted_score(item.get("raw"))
        item["index"] = len(results)
        row_id = _insert_result_row(
            request_id=request_id,
            image_index=item["index"],
            filename=fn,
            status=str(item.get("status") or "error"),
            predicted_brand=item.get("predicted_brand"),
            predicted_score=item.get("predicted_score"),
            latency_ms=latency_ms,
            run_tag=RUN_TAG,
            tuning_version=tuning_version,
        )
        item["result_id"] = row_id
        item["run_tag"] = RUN_TAG
        item["tuning_version"] = tuning_version
        results.append(item)

    agg = _aggregate_status(results)
    summary = _chat_summary(results)

    out: dict[str, Any] = {
        "schema_version": 1,
        "limits": {"max_upload_files": MAX_UPLOAD_FILES},
        "request_id": request_id,
        "run_tag": RUN_TAG,
        "tuning_version": tuning_version,
        "status": agg,
        "message": summary,
        "results": results,
    }
    # Flat fields for simple / legacy clients (first image only).
    if results:
        out["predicted_brand"] = results[0].get("predicted_brand")
        out["detail_code"] = results[0].get("detail_code")
    else:
        out["predicted_brand"] = None
        out["detail_code"] = None

    return out
