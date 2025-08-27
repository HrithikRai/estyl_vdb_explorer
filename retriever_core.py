from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import io
import json
import math
import os
import threading
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery

# ===========
# Constants
# ===========
WEAVIATE_URL = os.getenv("WEAVIATE_HOST", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
COLLECTION = os.getenv("WEAVIATE_COLLECTION", "Estyl_articles")

QUERY_PROPS = [
    "title","category","subcategory","subsubcategory","image_caption",
    "description","color","fabric","brand","unique_features"
]
RETURN_PROPS = [
    "title","gender","product_url","gcs_image_path","brand","price","category","color",
    "subcategory","subsubcategory","description","image_caption","unique_features"
]

CATEGORY_OPTIONS = [
    "Tops","Bottoms","Shoes","Outerwear","Jewelry & Watches","Accessories",
    "Activewear","Suits & Tailoring","Underwear","Bags","Dresses & One-Pieces","Lingerie/Underwear",
    "Small Leather Goods", "Swimwear"
]

OUTFIT_ORDER_BY_N = {
    2: ["Tops", "Bottoms"],
    3: ["Tops", "Bottoms", "Shoes"],
    4: ["Tops", "Bottoms", "Shoes", "Outerwear"],
    5: ["Tops", "Bottoms", "Shoes", "Outerwear", "Accessories"],
}

BUDGET_TIERS: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {
    "Tops": {"Budget": (0, 30), "Mid": (30, 70), "Premium": (70, 150), "Luxury": (150, None)},
    "Dresses & One-Pieces": {"Budget": (0, 100), "Mid": (100, 250), "Premium": (250, 600), "Luxury": (600, None)},
    "Bottoms": {"Budget": (0, 60), "Mid": (60, 130), "Premium": (130, 250), "Luxury": (250, None)},
    "Outerwear": {"Budget": (0, 120), "Mid": (120, 300), "Premium": (300, 700), "Luxury": (700, None)},
    "Suits & Tailoring": {"Budget": (0, 150), "Mid": (150, 350), "Premium": (350, 800), "Luxury": (800, None)},
    "Lingerie/Underwear": {"Budget": (0, 20), "Mid": (20, 50), "Premium": (50, 100), "Luxury": (100, None)},
    "Sleep & Lounge": {"Budget": (0, 40), "Mid": (40, 100), "Premium": (100, 200), "Luxury": (200, None)},
    "Activewear": {"Budget": (0, 40), "Mid": (40, 90), "Premium": (90, 180), "Luxury": (180, None)},
    "Swimwear": {"Budget": (0, 40), "Mid": (40, 100), "Premium": (100, 200), "Luxury": (200, None)},
    "Shoes": {"Budget": (0, 80), "Mid": (80, 150), "Premium": (150, 300), "Luxury": (300, None)},
    "Bags": {"Budget": (0, 80), "Mid": (80, 200), "Premium": (200, 500), "Luxury": (500, None)},
    "Small Leather Goods": {"Budget": (0, 40), "Mid": (40, 100), "Premium": (100, 200), "Luxury": (200, None)},
    "Accessories": {"Budget": (0, 30), "Mid": (30, 80), "Premium": (80, 150), "Luxury": (150, None)},
    "Jewelry & Watches": {"Budget": (0, 80), "Mid": (80, 200), "Premium": (200, 500), "Luxury": (500, None)},
}

# Default budget allocation weights
OUTFIT_WEIGHTS = {"Tops": 0.22, "Bottoms": 0.25, "Shoes": 0.30, "Outerwear": 0.35, "Accessories": 0.08}

# Canonicalization map for free-form category mentions
_CAT_SYNONYMS = {
    "tops": "Tops", "shirt": "Tops", "t-shirt": "Tops", "tee": "Tops", "blouse": "Tops",
    "bottoms": "Bottoms", "pants": "Bottoms", "trousers": "Bottoms", "jeans": "Bottoms", "skirt": "Bottoms", "shorts": "Bottoms",
    "shoes": "Shoes", "sneakers": "Shoes", "boots": "Shoes", "heels": "Shoes", "sandals": "Shoes", "loafers": "Shoes",
    "outerwear": "Outerwear", "jacket": "Outerwear", "coat": "Outerwear", "blazer": "Outerwear", "hoodie": "Outerwear",
    "accessories": "Accessories", "watch": "Accessories", "belt": "Accessories", "hat": "Accessories",
    "scarf": "Accessories", "sunglasses": "Accessories", "glasses": "Accessories", "tie": "Accessories",
}

# ===========
# Data models (I/O contracts for chatbot)
# ===========
@dataclass
class Product:
    uuid: str
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    properties: Dict[str, Any] = None  # must be JSON-serializable

@dataclass
class SingleSearchParams:
    text_query: str = ""
    image_bytes: Optional[bytes] = None
    gender: Optional[str] = None  # "male" | "female" | "unisex" | None/any
    categories: Optional[List[str]] = None  # canonical names
    brand_substr: Optional[str] = None
    budget_tier: Optional[str] = None  # "Budget"|"Mid"|"Premium"|"Luxury"
    user_budget: Optional[float] = None  # caps price_max when helpful
    limit: int = 12
    topk_for_rerank: int = 40
    offset: int = 0
    exclude_ids: Optional[Iterable[str]] = None
    reranker: str = "fashionclip"  # "fashionclip" | "lightweight"

@dataclass
class OutfitComposeParams:
    event_text: str = ""
    image_bytes: Optional[bytes] = None
    gender: Optional[str] = None
    brand_hint: Optional[str] = None
    selected_tier: str = "Mid"
    user_budget: float = 350.0
    num_outfits: int = 3
    articles: int = 3  # 2..5
    per_cat_candidates: int = 5
    reranker: str = "fashionclip"

@dataclass
class Outfit:
    items: List[Product]
    total_price: float
    budget_ok: bool

# ===========
# Thread-safe singletons (clients/models)
# ===========
_singleton_lock = threading.Lock()

@lru_cache(maxsize=1)
def get_weaviate_client() -> weaviate.WeaviateClient:
    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        raise RuntimeError("Missing WEAVIATE_HOST / WEAVIATE_API_KEY env vars.")
    with _singleton_lock:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        )

@lru_cache(maxsize=1)
def get_clip() -> Tuple[CLIPModel, CLIPProcessor, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

@lru_cache(maxsize=1)
def get_fashionclip() -> Tuple[CLIPModel, CLIPProcessor, CLIPTokenizer, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device).eval()
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    tokenizer = CLIPTokenizer.from_pretrained("patrickjohncyh/fashion-clip")
    return model, processor, tokenizer, device

# ===========
# Embeddings
# ===========
def _l2_normalize(v: np.ndarray, axis: Optional[int] = None, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True) + eps
    return (v / n).astype(np.float32)

def embed_text(text: str) -> np.ndarray:
    model, processor, device = get_clip()
    inputs = processor(text=[text or ""], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        vec = model.get_text_features(**inputs).cpu().numpy().flatten()
    return _l2_normalize(vec)

def embed_texts_batch(texts: Sequence[str]) -> np.ndarray:
    model, processor, device = get_clip()
    inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        vecs = model.get_text_features(**inputs).cpu().numpy()
    return _l2_normalize(vecs, axis=1)

def embed_image(image_bytes: bytes) -> np.ndarray:
    model, processor, device = get_clip()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vec = model.get_image_features(**inputs).cpu().numpy().flatten()
    return _l2_normalize(vec)

# ===========
# Filters / helpers
# ===========
def canonicalize_category(s: str) -> Optional[str]:
    if not s:
        return None
    k = str(s).strip()
    if k in CATEGORY_OPTIONS:
        return k
    return _CAT_SYNONYMS.get(k.lower())

def build_filters(
    gender: Optional[str],
    categories: Optional[Sequence[str]],
    price_min: Optional[float],
    price_max: Optional[float],
    brand_substr: Optional[str],
    exclude_ids: Optional[Iterable[str]] = None,
) -> Optional[Filter]:
    parts: List[Filter] = []
    if gender and gender.lower() != "any":
        parts.append(Filter.by_property("gender").equal(gender))
    if categories:
        cats = [c for c in categories if c]
        if cats:
            parts.append(Filter.any_of([Filter.by_property("category").equal(c) for c in cats]))
    if price_min is not None:
        parts.append(Filter.by_property("price").greater_or_equal(float(price_min)))
    if price_max is not None:
        parts.append(Filter.by_property("price").less_or_equal(float(price_max)))
    if brand_substr:
        parts.append(Filter.by_property("brand").like(f"*{brand_substr}*"))
    if exclude_ids:
        try:
            parts.append(Filter.by_id().not_in(list(exclude_ids)))
        except Exception:
            pass
    return Filter.all_of(parts) if parts else None

def get_tier_from_budget(category: str, budget: float) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    tiers = BUDGET_TIERS.get(category)
    if not tiers:
        return None, None, None
    for t, (lo, hi) in tiers.items():
        if (budget >= (lo or 0)) and (hi is None or budget <= hi):
            return t, lo, hi
    return None, None, None

def price_bounds_for(cat: str, tier: str) -> Tuple[Optional[float], Optional[float]]:
    if cat not in BUDGET_TIERS:
        return None, None
    return BUDGET_TIERS[cat][tier]

def compute_category_caps(cats_in_outfit: Sequence[str], total_budget: float) -> Dict[str, float]:
    weights = []
    for c in cats_in_outfit:
        weights.append(OUTFIT_WEIGHTS.get(c, 1.0))
    s = sum(weights) or 1.0
    return {c: float(total_budget * (w / s)) for c, w in zip(cats_in_outfit, weights)}

def normalize_price(x: Any) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x)
        for ch in ["€", "$", ",", " "]:
            s = s.replace(ch, "")
        return float(s)
    except Exception:
        return 0.0

def compose_rerank_text(p: Dict[str, Any]) -> str:
    return " | ".join([
        str(p.get("title", "")),
        str(p.get("brand", "")),
        str(p.get("category", "")),
        str(p.get("color", "")),
        str(p.get("image_caption", "")),
        str(p.get("unique_features", "")),
        str(p.get("description", ""))[:400],
    ])

# ===========
# Retrieval & Reranking
# ===========
def do_hybrid_search(
    text_query: str,
    text_vec: Optional[np.ndarray],
    image_vec: Optional[np.ndarray],
    limit: int,
    offset: int,
    filters: Optional[Filter],
) -> List[Any]:
    """
    Returns a list of Weaviate objects with .uuid, .properties, .metadata.score
    """
    client = get_weaviate_client()
    if image_vec is not None:
        # Visual → lean more on vector field, use image_vector
        effective_alpha = 0.60
        primary_vec = image_vec
        target_vector = "image_vector"
    else:
        # Text-only → keyword heavy (alpha≈0), use text_vector if provided
        effective_alpha = 0.0
        primary_vec = text_vec
        target_vector = "text_vector"

    objs = client.collections.get(COLLECTION).query.hybrid(
        query=text_query or "",
        vector=primary_vec.tolist() if primary_vec is not None else None,
        alpha=float(effective_alpha),
        target_vector=target_vector,
        query_properties=QUERY_PROPS,
        limit=int(limit),
        offset=int(offset),
        filters=filters,
        return_properties=RETURN_PROPS,
        return_metadata=MetadataQuery(score=True),
    )
    return objs.objects

def lightweight_rerank(objs: List[Any], query_text: str, query_img_vec: Optional[np.ndarray]) -> List[Any]:
    if not objs:
        return []
    texts = [compose_rerank_text(o.properties or {}) for o in objs]
    cand_vecs = embed_texts_batch(texts)  # (n, d)
    q = query_img_vec.reshape(1, -1) if query_img_vec is not None else embed_text(query_text).reshape(1, -1)
    sims = (q @ cand_vecs.T).flatten()
    boosts = []
    qt = (query_text or "").lower()
    for o in objs:
        p = o.properties or {}
        b = 0.0
        for key in ("category", "color", "brand"):
            val = str(p.get(key, "")).lower()
            if val and val in qt:
                b += 0.02
        boosts.append(b)
    scores = sims + np.array(boosts, dtype=np.float32)
    order = np.argsort(-scores)
    return [objs[i] for i in order]

def fashionclip_rerank(objs: List[Any], query_text: str, query_img_bytes: Optional[bytes]) -> List[Any]:
    if not objs:
        return []
    model, processor, tokenizer, device = get_fashionclip()

    # Candidate TEXT embeddings (FashionCLIP)
    texts = [compose_rerank_text(o.properties or {}) for o in objs]
    text_inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        doc_embs = model.get_text_features(**text_inputs)
    doc_embs = doc_embs / (doc_embs.norm(dim=1, keepdim=True) + 1e-9)

    # Query embedding
    if query_img_bytes:
        try:
            q_img = Image.open(io.BytesIO(query_img_bytes)).convert("RGB")
            q_inputs = processor(images=q_img, return_tensors="pt").to(device)
            with torch.no_grad():
                q_emb = model.get_image_features(**q_inputs)
        except Exception:
            # fallback to text
            q_inputs = processor(text=[query_text or ""], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                q_emb = model.get_text_features(**q_inputs)
    else:
        q_inputs = processor(text=[query_text or ""], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            q_emb = model.get_text_features(**q_inputs)
    q_emb = q_emb / (q_emb.norm(dim=1, keepdim=True) + 1e-9)

    sims = (q_emb @ doc_embs.T).squeeze(0).cpu().numpy()
    order = np.argsort(-sims)
    ranked = [objs[i] for i in order]
    # annotate scores when possible
    for i, o in enumerate(ranked):
        try:
            if getattr(o, "metadata", None) is not None:
                setattr(o.metadata, "fclip_score", float(sims[order[i]]))
            else:
                if o.properties is None:
                    o.properties = {}
                o.properties["_fclip_score"] = float(sims[order[i]])
        except Exception:
            pass
    return ranked

# ===========
# Public APIs
# ===========
def search_single_items(params: SingleSearchParams) -> Dict[str, Any]:
    """
    Main single-item search for chatbot.

    Input: SingleSearchParams
    Output (JSON-safe):
      {
        "items": [ Product, ... ]  # dicts
        "limit": int,
        "offset": int,             # echo
        "next_offset": int,        # offset + limit
        "budget_tier_detected": str | null,
        "took_ms": float
      }
    """
    import time
    t0 = time.time()

    # Build effective price range
    price_min, price_max = None, None
    cats = params.categories or []
    # If a single category, clamp via tier table; else cap by user budget
    if len(cats) == 1 and cats[0] in BUDGET_TIERS and params.budget_tier:
        lo, hi = BUDGET_TIERS[cats[0]][params.budget_tier]
        price_min, price_max = lo, hi
        if params.user_budget is not None:
            price_max = min(price_max, params.user_budget) if price_max else params.user_budget
    else:
        if params.user_budget is not None:
            price_max = params.user_budget

    filters = build_filters(
        gender=(params.gender if params.gender and params.gender != "any" else None),
        categories=cats,
        price_min=price_min,
        price_max=price_max,
        brand_substr=(params.brand_substr.strip() if params.brand_substr else None),
        exclude_ids=params.exclude_ids,
    )

    # Prepare vectors
    text_vec = embed_text(params.text_query) if (params.text_query and not params.image_bytes) else None
    image_vec = embed_image(params.image_bytes) if params.image_bytes else None

    # Retrieve
    hard_limit = max(params.limit, params.topk_for_rerank)
    objs = do_hybrid_search(
        text_query=params.text_query,
        text_vec=text_vec,
        image_vec=image_vec,
        limit=hard_limit,
        offset=params.offset,
        filters=filters,
    )

    # Rerank
    if params.reranker == "fashionclip":
        ranked = fashionclip_rerank(objs[:params.topk_for_rerank], params.text_query, params.image_bytes)
    else:
        ranked = lightweight_rerank(
            objs[:params.topk_for_rerank],
            params.text_query,
            image_vec if params.image_bytes is not None else None
        )
    ranked = ranked[:params.limit]

    # Format output
    items: List[Product] = []
    for o in ranked:
        props = o.properties or {}
        items.append(Product(
            uuid=str(o.uuid),
            score=getattr(getattr(o, "metadata", None), "score", None),
            rerank_score=(getattr(getattr(o, "metadata", None), "fclip_score", None)
                          or (props.get("_fclip_score") if isinstance(props, dict) else None)),
            properties=props
        ))

    took_ms = (time.time() - t0) * 1000.0

    detected_tier = None
    if len(cats) == 1 and params.user_budget is not None:
        detected_tier, _, _ = get_tier_from_budget(cats[0], params.user_budget)

    return {
        "items": [asdict(p) for p in items],
        "limit": params.limit,
        "offset": params.offset,
        "next_offset": params.offset + params.limit,
        "budget_tier_detected": detected_tier,
        "took_ms": round(took_ms, 2),
    }

def compose_outfits(params: OutfitComposeParams) -> Dict[str, Any]:
    """
    Build outfits under a budget, with planner-optional composing.

    Output:
      {
        "outfits": [ { "items": [Product,...], "total_price": float, "budget_ok": bool }, ... ],
        "near_misses": [ same as outfits ],
        "took_ms": float
      }
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import itertools

    t0 = time.time()

    # Categories to include based on slider
    cats = OUTFIT_ORDER_BY_N[5 if params.articles >= 5 else params.articles]

    # --- Plan queries per category (LLM optional) ---
    plan = call_llm_plan_safe(
        event=params.event_text,
        categories=cats,
        budget=params.user_budget,
        style_prefs=params.brand_hint,
        num_outfits=params.num_outfits,
    ) or lm_fallback_plan(
        num_outfits=params.num_outfits,
        cats=cats,
        text_query=params.event_text,
        brand_hint=params.brand_hint or "",
    )

    valid_outfits: List[Outfit] = []
    near_misses: List[Outfit] = []
    used_ids: set[str] = set()

    # Helpers
    def build_category_filters(cat: str, extra_price_cap: Optional[float] = None) -> Optional[Filter]:
        lo, hi = price_bounds_for(cat, params.selected_tier)
        if extra_price_cap is not None:
            hi = min(hi, extra_price_cap) if hi is not None else extra_price_cap
        return build_filters(
            gender=(params.gender if params.gender and params.gender != "any" else None),
            categories=[cat],
            price_min=lo,
            price_max=hi,
            brand_substr=(params.brand_hint.strip() if params.brand_hint else None),
        )

    def rerank_for_query(objs: List[Any], q_text: str) -> List[Any]:
        # limit per-cat for speed before final combination
        pre = min(len(objs), max(8, params.per_cat_candidates * 2))
        if params.reranker == "fashionclip":
            return fashionclip_rerank(objs[:pre], q_text, params.image_bytes)
        return lightweight_rerank(objs[:pre], q_text, None)

    def obj_price(o: Any) -> float:
        return normalize_price((o.properties or {}).get("price", 0))

    def score_rank_order(index: int, total: int) -> float:
        if total <= 1: return 1.0
        return 1.0 - (index / (total - 1))

    # Compose outfits
    for outfit_idx in range(min(params.num_outfits, len(plan))):
        outfit_plan: Dict[str, str] = plan[outfit_idx]
        cats_in_this = list(outfit_plan.keys())
        caps = compute_category_caps(cats_in_this, params.user_budget)

        # Parallel retrieval per category
        cat_queries = [(c, (outfit_plan.get(c) or f"{params.event_text} {c} {params.brand_hint or ''}".strip()))
                       for c in cats_in_this]

        candidates_map: Dict[str, List[Any]] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(cat_queries))) as ex:
            futures = []
            for cat, q in cat_queries:
                f = build_category_filters(cat, extra_price_cap=caps.get(cat))
                futures.append(ex.submit(
                    do_hybrid_search,
                    text_query=q, text_vec=None, image_vec=None,
                    limit=max(6, params.per_cat_candidates * 2),
                    offset=0, filters=f
                ))
            for (cat, _), fut in zip(cat_queries, futures):
                objs = fut.result()
                ranked = rerank_for_query(objs, outfit_plan.get(cat, ""))
                # drop duplicates across categories
                filtered = [o for o in ranked if o.uuid not in used_ids][:params.per_cat_candidates]
                candidates_map[cat] = filtered

        # skip if any category failed
        if any(len(candidates_map.get(c, [])) == 0 for c in cats_in_this):
            continue

        # Safety guard: cap combinatorics
        lengths = [len(candidates_map[c]) for c in cats_in_this]
        max_combos = 60000
        prod_len = math.prod(lengths)
        if prod_len > max_combos:
            # Reduce each list proportionally to fit the cap
            k = int(max_combos ** (1.0 / len(lengths)))
            for c in cats_in_this:
                candidates_map[c] = candidates_map[c][:max(k, 2)]

        cand_lists = [candidates_map[c] for c in cats_in_this]
        rank_scores: List[Dict[str, float]] = []
        for lst in cand_lists:
            n = max(1, len(lst))
            rank_scores.append({o.uuid: score_rank_order(i, n) for i, o in enumerate(lst)})

        best_combo = None
        best_score = -1e9
        best_price = 1e12

        import itertools
        for combo in itertools.product(*cand_lists):
            uuids = {o.uuid for o in combo}
            if len(uuids) < len(combo):  # unique across categories
                continue
            total_price = float(sum(obj_price(o) for o in combo))
            score = 0.0
            for idx, o in enumerate(combo):
                cat = cats_in_this[idx]
                s_rank = rank_scores[idx][o.uuid]
                cap = caps.get(cat) or (params.user_budget / len(cats_in_this))
                s_price_penalty = 0.02 * (obj_price(o) / max(1.0, cap))
                score += (s_rank - s_price_penalty)
            if total_price <= params.user_budget:
                if (score > best_score) or (score == best_score and total_price < best_price):
                    best_score, best_combo, best_price = score, combo, total_price

        if best_combo is None:
            # choose closest over-budget combo
            best_over = None
            best_gap = 1e12
            best_over_score = -1e9
            for combo in itertools.product(*cand_lists):
                uuids = {o.uuid for o in combo}
                if len(uuids) < len(combo):  # unique
                    continue
                total_price = float(sum(obj_price(o) for o in combo))
                gap = total_price - params.user_budget
                score = 0.0
                for idx, o in enumerate(combo):
                    cat = cats_in_this[idx]
                    s_rank = rank_scores[idx][o.uuid]
                    cap = caps.get(cat) or (params.user_budget / len(cats_in_this))
                    s_price_penalty = 0.02 * (obj_price(o) / max(1.0, cap))
                    score += (s_rank - s_price_penalty)
                if (gap < best_gap) or (gap == best_gap and score > best_over_score):
                    best_gap, best_over_score, best_over = gap, score, (combo, total_price)
            if best_over:
                combo, total_price = best_over
                items = [weaviate_obj_to_product(o) for o in combo]
                # mark used to diversify subsequent outfits
                for o in combo: used_ids.add(o.uuid)
                near_misses.append(Outfit(items=items, total_price=total_price, budget_ok=False))
                # Still add to overall set (caller can decide what to show)
                continue

        if best_combo:
            items = [weaviate_obj_to_product(o) for o in best_combo]
            for o in best_combo: used_ids.add(o.uuid)
            valid_outfits.append(Outfit(items=items, total_price=best_price, budget_ok=True))

    took_ms = (time.time() - t0) * 1000.0
    return {
        "outfits": [asdict(o) for o in valid_outfits],
        "near_misses": [asdict(o) for o in near_misses],
        "took_ms": round(took_ms, 2),
    }

def weaviate_obj_to_product(o: Any) -> Product:
    props = o.properties or {}
    return Product(
        uuid=str(o.uuid),
        score=getattr(getattr(o, "metadata", None), "score", None),
        rerank_score=(getattr(getattr(o, "metadata", None), "fclip_score", None)
                      or (props.get("_fclip_score") if isinstance(props, dict) else None)),
        properties=props
    )

# ===========
# Planner (LLM optional & safe)
# ===========
def lm_fallback_plan(num_outfits: int, cats: Sequence[str], text_query: str, brand_hint: str) -> List[Dict[str, str]]:
    base = (text_query or "").strip()
    plans = []
    for _ in range(num_outfits):
        plan = {}
        for c in cats:
            q = " ".join([w for w in [base, c, (brand_hint or "").strip()] if w])
            plan[c] = q
        plans.append(plan)
    return plans

def call_llm_plan_safe(event: str, categories: Sequence[str], budget: float, style_prefs: Optional[str], num_outfits: int = 1) -> Optional[List[Dict[str, str]]]:
    """
    Try OpenAI planner; return None if not configured or any failure.
    Returns a list of dicts keyed by provided categories.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You output only valid JSON: a LIST of outfit objects keyed by the provided categories."},
                {"role": "user", "content": f"""
Create {num_outfits} outfit plans as a JSON LIST. Use exactly these categories:
{list(categories)}
Constraints:
- Event: {event}
- Total budget (EUR): {budget}
- Style/brand hints: {style_prefs}

Output example:
[
  {{"Tops": "white oxford shirt, slim fit", "Bottoms": "navy tailored trousers", "Shoes": "black leather oxford shoes"}}
]
"""},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        if isinstance(data, dict) and "outfits" in data:
            data = data["outfits"]
        if isinstance(data, dict):
            data = [data]
        plans: List[Dict[str, str]] = []
        for outfit in (data or []):
            if not isinstance(outfit, dict): continue
            plan: Dict[str, str] = {}
            for k, v in outfit.items():
                cat = canonicalize_category(k) or k
                if cat in categories:
                    plan[cat] = v if isinstance(v, str) else " ".join(v) if isinstance(v, list) else str(v)
            # Ensure all categories exist
            plan = {c: plan.get(c, f"{event} {c} {style_prefs or ''}".strip()) for c in categories}
            plans.append(plan)
        return plans or None
    except Exception:
        return None

def close_weaviate_client():
    client = get_weaviate_client()
    try:
        client.close()
    except Exception:
        pass
    get_weaviate_client.cache_clear()
    
# ===========
# High-level Chatbot Entry
# ===========
def handle_chatbot_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic entry point for your chatbot.
    Payload examples:

    Single search:
    {
      "mode": "single",
      "text_query": "blue tee and white sneakers",
      "image_bytes": null,
      "gender": "female",
      "categories": ["Tops","Shoes"],
      "brand_substr": "adidas",
      "budget_tier": "Mid",
      "user_budget": 250,
      "limit": 12,
      "topk_for_rerank": 40,
      "offset": 0,
      "exclude_ids": [],
      "reranker": "fashionclip"
    }

    Outfit build:
    {
      "mode": "outfit",
      "event_text": "smart casual dinner",
      "image_bytes": null,
      "gender": "female",
      "brand_hint": "COS, Arket",
      "selected_tier": "Mid",
      "user_budget": 400,
      "num_outfits": 2,
      "articles": 3,
      "per_cat_candidates": 5,
      "reranker": "fashionclip"
    }
    """
    mode = (payload.get("mode") or "").lower()
    if mode == "single":
        params = SingleSearchParams(
            text_query=payload.get("text_query", ""),
            image_bytes=payload.get("image_bytes"),
            gender=payload.get("gender"),
            categories=payload.get("categories"),
            brand_substr=payload.get("brand_substr"),
            budget_tier=payload.get("budget_tier"),
            user_budget=payload.get("user_budget"),
            limit=int(payload.get("limit", 12)),
            topk_for_rerank=int(payload.get("topk_for_rerank", 40)),
            offset=int(payload.get("offset", 0)),
            exclude_ids=payload.get("exclude_ids") or [],
            reranker=payload.get("reranker", "fashionclip"),
        )
        return search_single_items(params)
    elif mode == "outfit":
        params = OutfitComposeParams(
            event_text=payload.get("event_text", ""),
            image_bytes=payload.get("image_bytes"),
            gender=payload.get("gender"),
            brand_hint=payload.get("brand_hint"),
            selected_tier=payload.get("selected_tier", "Mid"),
            user_budget=float(payload.get("user_budget", 350.0)),
            num_outfits=int(payload.get("num_outfits", 3)),
            articles=max(2, min(5, int(payload.get("articles", 3)))),
            per_cat_candidates=max(1, min(10, int(payload.get("per_cat_candidates", 5)))),
            reranker=payload.get("reranker", "fashionclip"),
        )
        return compose_outfits(params)
    else:
        return {"error": "Invalid mode. Use 'single' or 'outfit'."}
