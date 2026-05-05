"""
gk_two_brand_matcher.py

Focused GK pipeline for 2 brands:
1) Crawl configured brand URLs with BeautifulSoup
2) Download likely product images daily into brand-specific folders
3) Build a local neural embedding index
4) Compare a user image against disk catalog and return same-photo yes/no

Notes:
- This script is designed for local use and incremental daily refresh.
- Respect website Terms/robots and avoid abusive request rates.
- Product image detection is heuristic; tune filters as needed.

Optional scrape tuning (environment variables):
- GK_REQUEST_SLEEP_S — seconds between HTTP attempts (default 0.45).
- GK_MAX_EXTRA_PAGES — max extra product/collection links per seed URL (default 40).
- GK_MAX_CANDIDATES_PER_PAGE — max image URLs to try per page; Shopify JSON can list
  thousands, each with a sleep + request (default 400). Set 0 for no limit (very slow).
- GK_REQUEST_TIMEOUT_S — per-request timeout seconds (default 20).

Brand vote tuning (query/embed path; helps when one catalog is much larger):
- GK_BRAND_VOTE_TOP_K — neighbors considered for brand aggregation (default 24).
- GK_BRAND_VOTE_PER_BRAND_TOP_M — max similarities kept per brand before averaging (default 4).
- GK_BRAND_VOTE_MEAN_WEIGHT / GK_BRAND_VOTE_MAX_WEIGHT — blend for brand score (defaults 0.75 / 0.25;
  normalized to sum to 1). More max weight favors the single strongest neighbor per brand.
- GK_BRAND_MINORITY_SCORE_MULT — multiply final vote score for the brand with fewer indexed images
  (default 1.0 = off). Try 1.08–1.15 to counter catalog-size skew without re-indexing.

Dependencies:
    pip install requests beautifulsoup4 pillow numpy tensorflow
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup, Tag
from PIL import Image

# Scrape/crawl does not use TensorFlow. Lazy-load so `python run_gk_two_brand.py --mode scrape`
# starts immediately instead of sitting silent for minutes while TF initializes.
_tf_module = None


def _lazy_tf():
    global _tf_module
    if _tf_module is None:
        import tensorflow as tf

        _tf_module = tf
    return _tf_module


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ----------------------------
# Paths and constants
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # BZAN583
FRANKIES_DIR = BASE_DIR / "FrankiesBikinis"
MIRCHI_DIR = BASE_DIR / "MirchiByKim"
META_DIR = BASE_DIR / "gk_two_brand_meta"
META_DIR.mkdir(exist_ok=True)

MANIFEST_PATH = META_DIR / "image_manifest.json"
INDEX_PATH = META_DIR / "embedding_index.npz"
LAST_RUN_PATH = META_DIR / "last_daily_refresh.json"
ERROR_LOG_PATH = META_DIR / "query_errors.jsonl"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

REQUEST_TIMEOUT_S = _env_float("GK_REQUEST_TIMEOUT_S", 20.0)
REQUEST_SLEEP_S = _env_float("GK_REQUEST_SLEEP_S", 0.45)  # polite crawl interval
MIN_IMAGE_BYTES = 15_000
MIN_SIDE_PIXELS = 300
MAX_EXTRA_PRODUCT_PAGES_PER_SEED = _env_int("GK_MAX_EXTRA_PAGES", 40)
# Shopify themes embed huge JSON blobs; without a cap, one page can mean 10k+ sleeps.
MAX_CANDIDATES_PER_PAGE = _env_int("GK_MAX_CANDIDATES_PER_PAGE", 400)  # 0 = unlimited

# Similarity threshold for "same photo" (allow mild crops/resizes)
SAME_PHOTO_THRESHOLD = 0.88
# Softer threshold for "same item / likely same product listing family"
ITEM_MATCH_THRESHOLD = 0.80
BRAND_VOTE_PER_BRAND_TOP_M = _env_int("GK_BRAND_VOTE_PER_BRAND_TOP_M", 4)
BRAND_VOTE_TOP_K = _env_int("GK_BRAND_VOTE_TOP_K", 24)
_w_mean = _env_float("GK_BRAND_VOTE_MEAN_WEIGHT", 0.75)
_w_max = _env_float("GK_BRAND_VOTE_MAX_WEIGHT", 0.25)
_w_sum = _w_mean + _w_max
if _w_sum <= 0:
    BRAND_VOTE_MEAN_WEIGHT, BRAND_VOTE_MAX_WEIGHT = 0.75, 0.25
else:
    BRAND_VOTE_MEAN_WEIGHT = _w_mean / _w_sum
    BRAND_VOTE_MAX_WEIGHT = _w_max / _w_sum
BRAND_MINORITY_SCORE_MULT = _env_float("GK_BRAND_MINORITY_SCORE_MULT", 1.0)
# Penalize brands with much larger catalogs to reduce size-driven vote bias.
# 0.0 = off, 0.5 = divide by sqrt(count), 1.0 = divide by count.
BRAND_COUNT_NORM_EXP = _env_float("GK_BRAND_COUNT_NORM_EXP", 0.5)

# Allow common commerce/CDN image hosts (important for Shopify sites).
TRUSTED_IMAGE_HOST_SNIPPETS = [
    "cdn.shopify.com",
    "shopify.com",
    "shopifycdn",
    "cloudfront.net",
    "imgix.net",
]


# ----------------------------
# Brand URL configs
# ----------------------------
FRANKIES_URLS = [
    "https://frankiesbikinis.com/",
    "https://frankiesbikinis.com/collections/new-arrivals",
    "https://frankiesbikinis.com/collections/best-sellers",
    "https://frankiesbikinis.com/collections/back-in-stock",
    "https://frankiesbikinis.com/collections/wishes",
    "https://frankiesbikinis.com/collections/new-arrivals-swimwear",
    "https://frankiesbikinis.com/collections/new-arrivals-clothing",
    "https://frankiesbikinis.com/collections/festival-edit-26",
    "https://frankiesbikinis.com/collections/spring-break-edit-2026",
    "https://frankiesbikinis.com/collections/spring-florals",
    "https://frankiesbikinis.com/collections/beaded",
    "https://frankiesbikinis.com/pages/francescas-favorites",
    "https://frankiesbikinis.com/pages/rewards",
    "https://frankiesbikinis.com/collections/swimwear",
    "https://frankiesbikinis.com/collections/bikinis",
    "https://frankiesbikinis.com/collections/one-piece",
    "https://frankiesbikinis.com/collections/swim-tops",
    "https://frankiesbikinis.com/collections/swim-bottoms",
    "https://frankiesbikinis.com/collections/the-forever-collection",
    "https://frankiesbikinis.com/pages/fit-guide",
    "https://frankiesbikinis.com/collections/clothing",
    "https://frankiesbikinis.com/collections/dresses-rompers",
    "https://frankiesbikinis.com/collections/sweaters-knits",
    "https://frankiesbikinis.com/collections/matching-sets",
    "https://frankiesbikinis.com/collections/accessories",
    "https://frankiesbikinis.com/collections/tops",
    "https://frankiesbikinis.com/collections/bottoms",
    "https://frankiesbikinis.com/collections/spring-26-drop-2",
    "https://frankiesbikinis.com/collections/valentines-collection",
    "https://frankiesbikinis.com/collections/lexxola-x-frankies-bikinis",
    "https://frankiesbikinis.com/collections/spring-26",
    "https://frankiesbikinis.com/collections/holiday-house",
    "https://frankiesbikinis.com/collections/wrapped-in-frankies",
    "https://frankiesbikinis.com/collections/resort-26",
    "https://frankiesbikinis.com/collections/september-knits",
    "https://frankiesbikinis.com/products/gift-card?variant=21338122309",
    "https://frankiesbikinis.com/pages/our-story",
    "https://frankiesbikinis.com/blogs/blog",
    "https://frankiesbikinis.com/pages/lookbooks",
    "https://frankiesbikinis.com/collections/sale",
    "https://frankiesbikinis.com/collections/new-to-sale",
    "https://frankiesbikinis.com/collections/sale-swimwear",
    "https://frankiesbikinis.com/collections/sale-clothing",
    "https://frankiesbikinis.com/collections/bella-hadid-x-frankies-bikinis",
]

MIRCHI_URLS = [
    "https://mirchibykim.com/",
    "https://mirchibykim.com/collections/frontpage",
    "https://mirchibykim.com/collections/denim",
    "https://mirchibykim.com/collections/bags",
    "https://mirchibykim.com/collections/skirts",
    "https://mirchibykim.com/collections/tops",
    "https://mirchibykim.com/collections/sets",
    "https://mirchibykim.com/collections/dresses",
    "https://mirchibykim.com/collections/tees-tanks",
    "https://mirchibykim.com/collections/jackets",
    "https://mirchibykim.com/collections/bottoms",
    "https://mirchibykim.com/collections/mirchi2k",
    "https://mirchibykim.com/collections/capsule-drop",
    "https://mirchibykim.com/collections/mirchi-girl-drop",
    "https://mirchibykim.com/collections/chapter-ii",
    "https://mirchibykim.com/collections/collection-i",
    "https://mirchibykim.com/pages/behind-the-brand",
    "https://mirchibykim.com/pages/craftsmanship",
    "https://mirchibykim.com/pages/sustainable-packaging",
    "https://mirchibykim.com/pages/press",
    "https://mirchibykim.com/pages/photo-journal",
    "https://mirchibykim.com/pages/customization",
]


@dataclass
class BrandConfig:
    name: str
    base_domain: str
    urls: list[str]
    out_dir: Path


BRANDS = [
    BrandConfig("FrankiesBikinis", "frankiesbikinis.com", FRANKIES_URLS, FRANKIES_DIR),
    BrandConfig("MirchiByKim", "mirchibykim.com", MIRCHI_URLS, MIRCHI_DIR),
]
BRAND_NAME_SET = {b.name for b in BRANDS}


def apply_quick_scrape_profile() -> None:
    """
    Smaller/faster crawl for local testing (call before scrape).
    Overrides module globals; does not change environment variables.
    """
    global REQUEST_SLEEP_S, MAX_EXTRA_PRODUCT_PAGES_PER_SEED, MAX_CANDIDATES_PER_PAGE
    REQUEST_SLEEP_S = 0.2
    MAX_EXTRA_PRODUCT_PAGES_PER_SEED = 12
    MAX_CANDIDATES_PER_PAGE = 120


def scrape_limits_summary() -> str:
    return (
        f"sleep={REQUEST_SLEEP_S}s, timeout={REQUEST_TIMEOUT_S}s, "
        f"max_extra_pages_per_seed={MAX_EXTRA_PRODUCT_PAGES_PER_SEED}, "
        f"max_candidates_per_page={MAX_CANDIDATES_PER_PAGE or 'unlimited'}"
    )


# ----------------------------
# Utilities
# ----------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"images": [], "by_hash": {}}


def save_manifest(m: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")


def safe_ext_from_url(url: str) -> str:
    p = urlparse(url).path.lower()
    if p.endswith(".png"):
        return ".png"
    if p.endswith(".webp"):
        return ".webp"
    if p.endswith(".jpeg") or p.endswith(".jpg"):
        return ".jpg"
    return ".jpg"


def parse_srcset_first(srcset: str) -> Optional[str]:
    # Typical format: "url1 480w, url2 768w"
    if not srcset:
        return None
    first = srcset.split(",")[0].strip()
    if not first:
        return None
    return first.split(" ")[0].strip()


def likely_product_context(img_tag: Tag) -> bool:
    # Heuristic: include if product-ish classes/attributes appear
    text_blob = " ".join(
        [
            str(img_tag.get("class", "")),
            str(img_tag.get("id", "")),
            str(img_tag.get("alt", "")),
            str(img_tag.parent.get("class", "") if img_tag.parent else ""),
            str(img_tag.parent.get("id", "") if img_tag.parent else ""),
        ]
    ).lower()
    positive = [
        "product",
        "card",
        "grid",
        "gallery",
        "collection",
        "item",
        "media",
        "tile",
    ]
    negative = [
        "logo",
        "icon",
        "sprite",
        "payment",
        "flag",
        "social",
        "avatar",
    ]
    return any(k in text_blob for k in positive) and not any(k in text_blob for k in negative)


def normalize_img_url(page_url: str, src: str) -> Optional[str]:
    if not src:
        return None
    src = src.strip()
    if src.startswith("data:"):
        return None
    full = urljoin(page_url, src)
    if not full.startswith(("http://", "https://")):
        return None
    return full


def is_trusted_image_url(url: str, domain_hint: str) -> bool:
    u = url.lower()
    if domain_hint.lower() in u:
        return True
    return any(snippet in u for snippet in TRUSTED_IMAGE_HOST_SNIPPETS)


def likely_product_page_url(url: str) -> bool:
    u = url.lower()
    return (
        "/products/" in u
        or "product" in u
        or "/collections/" in u
    )


def fetch_html(session: requests.Session, url: str) -> Optional[str]:
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT_S)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None


def extract_candidate_image_urls(page_url: str, html: str, domain_hint: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    seen: set[str] = set()
    cap = MAX_CANDIDATES_PER_PAGE

    def _room() -> bool:
        return cap <= 0 or len(out) < cap

    for img in soup.find_all("img"):
        if not _room():
            break
        if not isinstance(img, Tag):
            continue
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-original")
            or parse_srcset_first(img.get("srcset", ""))
        )
        full = normalize_img_url(page_url, src or "")
        if not full:
            continue

        # Domain and format filtering first
        if not is_trusted_image_url(full, domain_hint):
            continue
        if re.search(r"\.(svg|gif)(\?|$)", full.lower()):
            continue

        # Product context heuristic
        if not likely_product_context(img):
            # keep some obvious Shopify product image patterns anyway
            if "products" not in full.lower() and "shopify" not in full.lower():
                continue

        if full not in seen:
            seen.add(full)
            out.append(full)

    # Also parse image URLs from script JSON blobs (common in Shopify themes).
    # This catches product media not always present in <img> tags in initial HTML.
    for script in soup.find_all("script"):
        if not _room():
            break
        txt = script.get_text(" ", strip=True)
        if not txt:
            continue
        # Broad regex for image URLs in embedded JSON/JS.
        for m in re.finditer(r"https?://[^\s\"']+\.(?:jpg|jpeg|png|webp)(?:\?[^\s\"']*)?", txt, flags=re.I):
            if not _room():
                break
            full = m.group(0)
            if not is_trusted_image_url(full, domain_hint):
                continue
            if re.search(r"\.(svg|gif)(\?|$)", full.lower()):
                continue
            if full not in seen:
                seen.add(full)
                out.append(full)
    return out


def extract_candidate_product_links(page_url: str, html: str, domain_hint: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a"):
        if not isinstance(a, Tag):
            continue
        href = (a.get("href") or "").strip()
        if not href:
            continue
        full = urljoin(page_url, href)
        if not full.startswith(("http://", "https://")):
            continue
        if domain_hint not in full.lower():
            continue
        if not likely_product_page_url(full):
            continue
        # Avoid non-product assets/pages
        if any(k in full.lower() for k in ["/account", "/cart", "/search", "/policies", "/blogs", "/pages/"]):
            continue
        if full not in seen:
            seen.add(full)
            out.append(full)
    return out


def download_candidates_for_page(
    session: requests.Session,
    source_page_url: str,
    candidates: list[str],
    brand: BrandConfig,
    by_hash: dict[str, Any],
    images_meta: list[dict[str, Any]],
    stats: dict[str, int],
) -> None:
    for img_url in candidates:
        data = download_image(session, img_url)
        time.sleep(REQUEST_SLEEP_S)
        if not data:
            continue
        if not image_ok_for_catalog(data):
            continue

        h = sha256_bytes(data)
        if h in by_hash:
            stats["skipped_dup"] += 1
            continue

        ext = safe_ext_from_url(img_url)
        out_path = brand.out_dir / f"{h}{ext}"
        out_path.write_bytes(data)

        meta = {
            "sha256": h,
            "brand": brand.name,
            "source_page": source_page_url,
            "source_img_url": img_url,
            "local_path": str(out_path),
            "downloaded_at": datetime.utcnow().isoformat(),
        }
        images_meta.append(meta)
        by_hash[h] = meta
        stats["downloaded_new"] += 1


def download_image(session: requests.Session, url: str) -> Optional[bytes]:
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT_S)
        if r.status_code != 200:
            return None
        ctype = (r.headers.get("content-type") or "").lower()
        if "image" not in ctype:
            return None
        data = r.content
        if len(data) < MIN_IMAGE_BYTES:
            return None
        return data
    except Exception:
        return None


def image_ok_for_catalog(data: bytes) -> bool:
    try:
        with Image.open(Path(os.devnull)) as _:
            pass
    except Exception:
        # intentional no-op for PIL import sanity
        pass
    try:
        from io import BytesIO

        with Image.open(BytesIO(data)) as im:
            w, h = im.size
            if min(w, h) < MIN_SIDE_PIXELS:
                return False
            return True
    except Exception:
        return False


def crawl_brand(brand: BrandConfig, manifest: dict[str, Any]) -> dict[str, int]:
    brand.out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    stats = {
        "pages_ok": 0,
        "img_candidates": 0,
        "downloaded_new": 0,
        "skipped_dup": 0,
        "extra_pages_ok": 0,
    }
    by_hash = manifest.get("by_hash", {})
    images_meta = manifest.get("images", [])

    n_seeds = len(brand.urls)
    for si, url in enumerate(brand.urls, start=1):
        print(
            f"    [{brand.name}] seed {si}/{n_seeds}: {url[:88]}{'…' if len(url) > 88 else ''}",
            flush=True,
        )
        html = fetch_html(session, url)
        time.sleep(REQUEST_SLEEP_S)
        if not html:
            continue
        stats["pages_ok"] += 1

        candidates = extract_candidate_image_urls(url, html, brand.base_domain)
        stats["img_candidates"] += len(candidates)
        print(
            f"      → {len(candidates)} image URL(s) to try on this page "
            f"(new_so_far={stats['downloaded_new']}, dups_skipped={stats['skipped_dup']})",
            flush=True,
        )
        download_candidates_for_page(session, url, candidates, brand, by_hash, images_meta, stats)

        # Aggressive pass: follow likely product links from this page.
        extra_links = extract_candidate_product_links(url, html, brand.base_domain)
        extras = extra_links[:MAX_EXTRA_PRODUCT_PAGES_PER_SEED]
        if extras:
            print(f"      → following {len(extras)} extra product/collection link(s)", flush=True)
        for ei, extra_url in enumerate(extras, start=1):
            extra_html = fetch_html(session, extra_url)
            time.sleep(REQUEST_SLEEP_S)
            if not extra_html:
                continue
            stats["extra_pages_ok"] += 1
            extra_candidates = extract_candidate_image_urls(extra_url, extra_html, brand.base_domain)
            stats["img_candidates"] += len(extra_candidates)
            print(
                f"        extra {ei}/{len(extras)}: {len(extra_candidates)} image URL(s)",
                flush=True,
            )
            download_candidates_for_page(
                session,
                extra_url,
                extra_candidates,
                brand,
                by_hash,
                images_meta,
                stats,
            )

    manifest["images"] = images_meta
    manifest["by_hash"] = by_hash
    return stats


# ----------------------------
# Embedding index + matching
# ----------------------------
def get_embedding_model():
    # Lightweight pretrained CNN feature extractor.
    tf = _lazy_tf()
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        pooling="avg",
        weights="imagenet",
    )
    return base


def load_and_preprocess(path: Path) -> np.ndarray:
    tf = _lazy_tf()
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def embedding_for_path(model, path: Path) -> np.ndarray:
    x = load_and_preprocess(path)
    emb = model.predict(x, verbose=0).reshape(-1).astype(np.float32)
    norm = np.linalg.norm(emb) + 1e-12
    return emb / norm


def embedding_views_for_path(model, path: Path) -> list[np.ndarray]:
    """
    Multi-view query embeddings to be more robust to crops and framing changes.
    """
    with Image.open(path) as raw:
        im = raw.convert("RGB")
        w, h = im.size
        # Full frame
        variants = [im]
        # Center crop (~85%) to tolerate whitespace/background shifts.
        cw, ch = int(w * 0.85), int(h * 0.85)
        if cw > 32 and ch > 32:
            left = max((w - cw) // 2, 0)
            top = max((h - ch) // 2, 0)
            variants.append(im.crop((left, top, left + cw, top + ch)))
        # Horizontal flip to reduce orientation sensitivity.
        variants.append(im.transpose(Image.FLIP_LEFT_RIGHT))

    tf = _lazy_tf()
    out: list[np.ndarray] = []
    for v in variants:
        arr = np.array(v.resize((224, 224), Image.BILINEAR), dtype=np.float32)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        x = np.expand_dims(arr, axis=0)
        emb = model.predict(x, verbose=0).reshape(-1).astype(np.float32)
        out.append(emb / (np.linalg.norm(emb) + 1e-12))
    return out


def build_embedding_index() -> dict[str, Any]:
    return build_embedding_index_for_brands(None)


def build_embedding_index_for_brands(selected_brands: Optional[set[str]]) -> dict[str, Any]:
    print("Building embedding index (loading TensorFlow + MobileNetV2 — first time can take 1–3 min)…", flush=True)
    model = get_embedding_model()
    entries = []
    vectors = []

    for brand in BRANDS:
        if selected_brands and brand.name not in selected_brands:
            continue
        brand.out_dir.mkdir(parents=True, exist_ok=True)
        for p in brand.out_dir.glob("*"):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            try:
                emb = embedding_for_path(model, p)
                vectors.append(emb)
                entries.append({"brand": brand.name, "path": str(p), "sha256": sha256_file(p)})
            except Exception:
                continue

    if not vectors:
        idx = {"vectors": np.zeros((0, 1280), dtype=np.float32), "entries": []}
    else:
        idx = {"vectors": np.vstack(vectors), "entries": entries}

    np.savez_compressed(
        INDEX_PATH,
        vectors=idx["vectors"],
        entries_json=np.array([json.dumps(e) for e in idx["entries"]], dtype=object),
    )
    return idx


def load_embedding_index() -> dict[str, Any]:
    if not INDEX_PATH.exists():
        return build_embedding_index()
    data = np.load(INDEX_PATH, allow_pickle=True)
    vectors = data["vectors"]
    entries = [json.loads(x) for x in data["entries_json"].tolist()]
    idx = {"vectors": vectors, "entries": entries}
    return ensure_full_brand_index(idx)


def _brand_has_disk_images(brand: BrandConfig) -> bool:
    if not brand.out_dir.exists():
        return False
    return any(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for p in brand.out_dir.glob("*"))


def ensure_full_brand_index(idx: dict[str, Any]) -> dict[str, Any]:
    """
    If index was built with --brand and now misses another brand that *does* have
    images on disk, rebuild a full index so query can pool across both folders.
    """
    counts = catalog_brand_counts(idx.get("entries", []))
    missing_with_files = [
        b.name
        for b in BRANDS
        if counts.get(b.name, 0) == 0 and _brand_has_disk_images(b)
    ]
    if not missing_with_files:
        return idx
    rebuilt = build_embedding_index()
    return rebuilt


def classify_decision_label(similarity: float, same_photo_threshold: float) -> str:
    if similarity >= same_photo_threshold:
        return "exact_or_near_exact"
    if similarity >= ITEM_MATCH_THRESHOLD:
        return "likely_same_item"
    return "uncertain"


def log_query_error(err: Exception, context: dict[str, Any]) -> str:
    ref = uuid.uuid4().hex[:10]
    payload = {
        "error_ref": ref,
        "ts_utc": datetime.utcnow().isoformat(),
        "context": context,
        "error_type": type(err).__name__,
        "error_message": str(err),
        "traceback": traceback.format_exc(),
    }
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return ref


def catalog_brand_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entries:
        b = e.get("brand", "unknown")
        counts[b] = counts.get(b, 0) + 1
    return counts


def query_same_photo(user_image_path: Path, threshold: float = SAME_PHOTO_THRESHOLD) -> dict[str, Any]:
    try:
        if not user_image_path.exists():
            return {"ok": False, "user_message": "still shopping", "error": "still_shopping"}

        index = load_embedding_index()
        if len(index["entries"]) == 0:
            return {"ok": False, "user_message": "still shopping", "error": "still_shopping"}

        # 1) Exact hash match shortcut
        user_hash = sha256_file(user_image_path)
        for e in index["entries"]:
            if e.get("sha256") == user_hash:
                return {
                    "ok": True,
                    "same_photo": True,
                    "item_match": True,
                    "decision_label": "exact_or_near_exact",
                    "reason": "exact_hash_match",
                    "best_match": e,
                    "similarity": 1.0,
                    "threshold": threshold,
                    "item_match_threshold": ITEM_MATCH_THRESHOLD,
                    "predicted_brand": e.get("brand"),
                }

        # 2) Neural embedding cosine similarity
        model = get_embedding_model()
        q_views = embedding_views_for_path(model, user_image_path)
        sims = None
        for q in q_views:
            v = index["vectors"] @ q  # vectors already L2-normalized
            sims = v if sims is None else np.maximum(sims, v)
        assert sims is not None
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_entry = index["entries"][best_i]

        # 3) Brand-level voting with per-brand cap to reduce catalog-size bias.
        n = len(index["entries"])
        k = min(BRAND_VOTE_TOP_K, n)
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        brand_scores: dict[str, list[float]] = {}
        for i in top_idx:
            b = index["entries"][int(i)].get("brand", "unknown")
            brand_scores.setdefault(b, []).append(float(sims[int(i)]))
        # Keep only each brand's strongest M neighbors, then average.
        brand_trimmed = {b: sorted(v, reverse=True)[:BRAND_VOTE_PER_BRAND_TOP_M] for b, v in brand_scores.items()}
        brand_mean_scores = {b: float(np.mean(v)) for b, v in brand_trimmed.items()}
        brand_max_scores = {b: float(np.max(v)) for b, v in brand_trimmed.items()}
        # Blend robustness (mean) + distinctiveness (max) for better generalization.
        brand_final_scores = {
            b: (BRAND_VOTE_MEAN_WEIGHT * brand_mean_scores[b])
            + (BRAND_VOTE_MAX_WEIGHT * brand_max_scores[b])
            for b in brand_mean_scores
        }
        counts = catalog_brand_counts(index["entries"])
        if BRAND_COUNT_NORM_EXP > 0:
            brand_final_scores = {
                b: float(brand_final_scores[b]) / max(float(counts.get(b, 1)) ** BRAND_COUNT_NORM_EXP, 1.0)
                for b in brand_final_scores
            }
        if BRAND_MINORITY_SCORE_MULT > 1.0 and len(brand_final_scores) >= 2:
            by_count = sorted(
                ((b, counts.get(b, 0)) for b in brand_final_scores),
                key=lambda t: t[1],
            )
            minority_brand = by_count[0][0]
            brand_final_scores = {
                b: (
                    float(brand_final_scores[b]) * BRAND_MINORITY_SCORE_MULT
                    if b == minority_brand
                    else float(brand_final_scores[b])
                )
                for b in brand_final_scores
            }
        predicted_brand = max(brand_final_scores.items(), key=lambda kv: kv[1])[0]

        missing_brands = [b.name for b in BRANDS if counts.get(b.name, 0) == 0]
        catalog_warning = None
        if missing_brands:
            catalog_warning = (
                "Catalog missing images for: "
                + ", ".join(missing_brands)
                + ". Run scrape/index again before trusting brand-level output."
            )

        return {
            "ok": True,
            "same_photo": bool(best_sim >= threshold),
            "item_match": bool(best_sim >= ITEM_MATCH_THRESHOLD),
            "decision_label": classify_decision_label(best_sim, threshold),
            "reason": "embedding_similarity",
            "similarity": best_sim,
            "threshold": threshold,
            "item_match_threshold": ITEM_MATCH_THRESHOLD,
            "predicted_brand": predicted_brand,
            "brand_vote_mean_similarity": brand_mean_scores,
            "brand_vote_method": (
                f"top_k={k}, per_brand_top_m={BRAND_VOTE_PER_BRAND_TOP_M}, "
                f"score={BRAND_VOTE_MEAN_WEIGHT:.3g}*mean+{BRAND_VOTE_MAX_WEIGHT:.3g}*max"
                + (
                    f", count_norm_exp={BRAND_COUNT_NORM_EXP}"
                    if BRAND_COUNT_NORM_EXP > 0
                    else ""
                )
                + (
                    f", minority_mult={BRAND_MINORITY_SCORE_MULT}"
                    if BRAND_MINORITY_SCORE_MULT > 1.0
                    else ""
                )
            ),
            "brand_vote_final_similarity": brand_final_scores,
            "catalog_brand_counts": counts,
            "catalog_warning": catalog_warning,
            "best_match": best_entry,
        }
    except Exception as err:
        ref = log_query_error(
            err,
            {
                "user_image_path": str(user_image_path),
                "threshold": threshold,
            },
        )
        return {
            "ok": False,
            "user_message": "still shopping",
            "error": "still_shopping",
            "debug_error_ref": ref,
        }


# ----------------------------
# Daily refresh orchestration
# ----------------------------
def refresh_all_brands_once() -> dict[str, Any]:
    return refresh_selected_brands_once(None)


def refresh_selected_brands_once(selected_brands: Optional[set[str]]) -> dict[str, Any]:
    manifest = load_manifest()
    run_stats = {"started_at": datetime.utcnow().isoformat(), "brands": {}}

    to_run = [b for b in BRANDS if not selected_brands or b.name in selected_brands]
    print(
        f"Scrape starting ({len(to_run)} brand(s)) — TensorFlow is not loaded until indexing.",
        flush=True,
    )
    print(f"  Crawl limits: {scrape_limits_summary()}", flush=True)
    for b in BRANDS:
        if selected_brands and b.name not in selected_brands:
            continue
        print(f"  Crawling {b.name}… (many URLs + polite delays; can take 30–90+ min)", flush=True)
        stats = crawl_brand(b, manifest)
        run_stats["brands"][b.name] = stats
        print(
            f"  {b.name} done: pages_ok={stats.get('pages_ok')} "
            f"downloaded_new={stats.get('downloaded_new')} skipped_dup={stats.get('skipped_dup')}",
            flush=True,
        )

    run_stats["finished_at"] = datetime.utcnow().isoformat()
    save_manifest(manifest)
    LAST_RUN_PATH.write_text(json.dumps(run_stats, indent=2), encoding="utf-8")

    # Rebuild index after crawl.
    idx = build_embedding_index_for_brands(selected_brands)
    run_stats["index_size"] = len(idx["entries"])
    LAST_RUN_PATH.write_text(json.dumps(run_stats, indent=2), encoding="utf-8")
    return run_stats


def run_daemon_daily(hour_24: int = 3, minute: int = 15) -> None:
    """
    Optional local daemon mode.
    For production reliability, prefer OS scheduler:
      - Windows Task Scheduler
      - cron on Linux/macOS
    """
    while True:
        now = datetime.now()
        should_run = now.hour == hour_24 and now.minute == minute
        already_ran_today = False

        if LAST_RUN_PATH.exists():
            try:
                j = json.loads(LAST_RUN_PATH.read_text(encoding="utf-8"))
                last = datetime.fromisoformat(j["finished_at"])
                already_ran_today = last.date() == now.date()
            except Exception:
                already_ran_today = False

        if should_run and not already_ran_today:
            stats = refresh_all_brands_once()
            print(json.dumps(stats, indent=2))

        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-brand scraper + same-photo matcher.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scrape_now", help="Crawl configured URLs and download product images.")
    s.add_argument(
        "--brand",
        choices=sorted(BRAND_NAME_SET),
        default=None,
        help="Optional: limit scraping to one brand.",
    )
    i = sub.add_parser("index_now", help="Build/rebuild neural embedding index from disk images.")
    i.add_argument(
        "--brand",
        choices=sorted(BRAND_NAME_SET),
        default=None,
        help="Optional: limit indexing to one brand.",
    )

    q = sub.add_parser("query", help="Compare user image against disk catalog and return same-photo yes/no.")
    q.add_argument("--image", required=True, help="Path to user image")
    q.add_argument("--threshold", type=float, default=SAME_PHOTO_THRESHOLD)

    d = sub.add_parser("daemon_daily", help="Run internal daily daemon mode.")
    d.add_argument("--hour", type=int, default=3)
    d.add_argument("--minute", type=int, default=15)

    args = parser.parse_args()

    if args.cmd == "scrape_now":
        selected = {args.brand} if getattr(args, "brand", None) else None
        stats = refresh_selected_brands_once(selected)
        print(json.dumps(stats, indent=2))
        return

    if args.cmd == "index_now":
        selected = {args.brand} if getattr(args, "brand", None) else None
        idx = build_embedding_index_for_brands(selected)
        print(json.dumps({"index_size": len(idx["entries"]), "index_path": str(INDEX_PATH)}, indent=2))
        return

    if args.cmd == "query":
        out = query_same_photo(Path(args.image), threshold=args.threshold)
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "daemon_daily":
        run_daemon_daily(hour_24=args.hour, minute=args.minute)
        return


if __name__ == "__main__":
    main()

