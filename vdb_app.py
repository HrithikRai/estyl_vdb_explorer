import os
import io
import time
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import streamlit as st
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from weaviate.classes.query import MetadataQuery
import openai
import json
# =============================================================
# Config & Caching
# =============================================================
load_dotenv()

st.set_page_config(
    page_title="Estyl Multimodal Search",
    page_icon="👗",
    layout="wide",
)

# --- Environment ---
WEAVIATE_URL = os.getenv("WEAVIATE_HOST", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
COLLECTION = os.getenv("WEAVIATE_COLLECTION", "Estyl_articles")

# --- Session init ---
if "seen_ids" not in st.session_state:
    st.session_state.seen_ids: set[str] = set()
if "last_query_sig" not in st.session_state:
    st.session_state.last_query_sig = None
if "offset" not in st.session_state:
    st.session_state.offset = 0

# --- Cache Weaviate client and CLIP ---
@st.cache_resource(show_spinner=False)
def get_client() -> weaviate.WeaviateClient:
    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        raise RuntimeError("Missing WEAVIATE_HOST / WEAVIATE_API_KEY env vars.")
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )

@st.cache_resource(show_spinner=False)
def get_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

@st.cache_resource(show_spinner=False)
def get_fashionclip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load FashionCLIP model + processor directly from Hugging Face
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    return model, processor, device

fclip_model, fclip_processor, fclip_device = get_fashionclip()

@st.cache_data(show_spinner=False, ttl=60*30)
def embed_text_cached(text: str) -> np.ndarray:
    model, processor, device = get_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        vec = model.get_text_features(**inputs).cpu().numpy().flatten().astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-9
    return (vec / norm).astype(np.float32)

@st.cache_data(show_spinner=False, ttl=60*10)
def embed_image_cached(image_bytes: bytes) -> np.ndarray:
    model, processor, device = get_clip()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vec = model.get_image_features(**inputs).cpu().numpy().flatten().astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-9
    return (vec / norm).astype(np.float32)

@st.cache_data(show_spinner=False, ttl=60*10)
def embed_texts_batch(texts: List[str]) -> np.ndarray:
    model, processor, device = get_clip()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        vecs = model.get_text_features(**inputs).cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return (vecs / norms).astype(np.float32)

# =============================================================
# Budget tiers (EUR) from your chart
# Each entry: (min, max) where None means unbounded on that side
# =============================================================
BUDGET_TIERS = {
    "Tops": {
        "Budget": (0, 30), "Mid": (30, 70), "Premium": (70, 150), "Luxury": (150, None)
    },
    "Dresses & One-Pieces": {
        "Budget": (0, 100), "Mid": (100, 250), "Premium": (250, 600), "Luxury": (600, None)
    },
    "Bottoms": {
        "Budget": (0, 60), "Mid": (60, 130), "Premium": (130, 250), "Luxury": (250, None)
    },
    "Outerwear": {
        "Budget": (0, 120), "Mid": (120, 300), "Premium": (300, 700), "Luxury": (700, None)
    },
    "Suits & Tailoring": {
        "Budget": (0, 150), "Mid": (150, 350), "Premium": (350, 800), "Luxury": (800, None)
    },
    "Lingerie/Underwear": {
        "Budget": (0, 20), "Mid": (20, 50), "Premium": (50, 100), "Luxury": (100, None)
    },
    "Sleep & Lounge": {
        "Budget": (0, 40), "Mid": (40, 100), "Premium": (100, 200), "Luxury": (200, None)
    },
    "Activewear": {
        "Budget": (0, 40), "Mid": (40, 90), "Premium": (90, 180), "Luxury": (180, None)
    },
    "Swimwear": {
        "Budget": (0, 40), "Mid": (40, 100), "Premium": (100, 200), "Luxury": (200, None)
    },
    "Shoes": {
        "Budget": (0, 80), "Mid": (80, 150), "Premium": (150, 300), "Luxury": (300, None)
    },
    "Bags": {
        "Budget": (0, 80), "Mid": (80, 200), "Premium": (200, 500), "Luxury": (500, None)
    },
    "Small Leather Goods": {
        "Budget": (0, 40), "Mid": (40, 100), "Premium": (100, 200), "Luxury": (200, None)
    },
    "Accessories": {
        "Budget": (0, 30), "Mid": (30, 80), "Premium": (80, 150), "Luxury": (150, None)
    },
    "Jewelry & Watches": {
        "Budget": (0, 80), "Mid": (80, 200), "Premium": (200, 500), "Luxury": (500, None)
    },
}

# Default proportion of outfit budget per category (used to set upper caps)
OUTFIT_WEIGHTS = {
    "Tops": 0.22,
    "Bottoms": 0.25,
    "Shoes": 0.30,
    "Outerwear": 0.35,
    "Accessories": 0.08,
}

# =============================================================
# Helpers
# =============================================================

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


def signature_for_query(mode: str, text: str, has_image: bool, knobs: Dict) -> Tuple:
    return (
        mode,
        text.strip().lower(),
        bool(has_image),
        tuple(sorted(knobs.items())),
    )


def build_filters(
    gender: Optional[str],
    categories: List[str],
    price_min: Optional[float],
    price_max: Optional[float],
    brand_substr: Optional[str],
    exclude_ids: Optional[List[str]] = None,
):
    parts = []
    if gender and gender.lower() != "any":
        parts.append(Filter.by_property("gender").equal(gender))
    if categories:
        parts.append(Filter.any_of([Filter.by_property("category").equal(c) for c in categories]))
    if price_min is not None:
        parts.append(Filter.by_property("price").greater_or_equal(price_min))
    if price_max is not None:
        parts.append(Filter.by_property("price").less_or_equal(price_max))
    if brand_substr:
        parts.append(Filter.by_property("brand").like(f"*{brand_substr}*"))

    if exclude_ids:
        try:
            parts.append(Filter.by_id().not_in(exclude_ids))
        except Exception:
            pass

    if not parts:
        return None
    return Filter.all_of(parts)


def compose_rerank_text(p: Dict) -> str:
    # Compact description string for reranking
    return " | ".join([
        str(p.get("title", "")),
        str(p.get("brand", "")),
        str(p.get("category", "")),
        str(p.get("color", "")),
        str(p.get("image_caption", "")),
        str(p.get("unique_features", "")),
        str(p.get("description", ""))[:400],
    ])


def lightweight_rerank(objs, query_text: str, query_img_vec: Optional[np.ndarray]):
    if not objs:
        return []

    # Prepare candidate texts
    texts = [compose_rerank_text(o.properties or {}) for o in objs]
    cand_vecs = embed_texts_batch(texts)  # (n, d)

    # Query embedding
    if query_img_vec is not None:
        # image↔text similarity in CLIP space
        q = query_img_vec.reshape(1, -1)
    else:
        q = embed_text_cached(query_text).reshape(1, -1)

    sims = (q @ cand_vecs.T).flatten()  # cosine-like because both normalized

    # Add tiny metadata boosts
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

def fashionclip_rerank(objs, query_text: str, query_img_bytes: Optional[bytes]):
    """Rerank `objs` using the transformers-based FashionCLIP weights loaded by get_fashionclip().
    - If query_img_bytes is present: compare query image embedding -> candidate TEXT embeddings.
    - Else: compare query TEXT embedding -> candidate TEXT embeddings.
    Returns a reordered list of objs (highest FashionCLIP score first).
    """
    if not objs:
        return []

    # use the cached HF model + processor you load earlier
    model, processor, device = fclip_model, fclip_processor, fclip_device

    # Build compact text for each candidate
    texts = [compose_rerank_text(o.properties or {}) for o in objs]

    # Encode candidate texts as a batch (FashionCLIP text features)
    tokenizer = CLIPTokenizer.from_pretrained("patrickjohncyh/fashion-clip")
    text_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,   # ✅ ensures it won’t exceed 77 tokens
        max_length=77,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        doc_embs = model.get_text_features(**text_inputs)  # (n, dim)
    # normalize
    doc_embs = doc_embs / (doc_embs.norm(dim=1, keepdim=True) + 1e-9)  # (n, dim)

    # Encode query (image OR text)
    if query_img_bytes is not None:
        # Query is an image: encode it
        try:
            query_img = Image.open(io.BytesIO(query_img_bytes)).convert("RGB")
            img_inputs = processor(images=query_img, return_tensors="pt").to(device)
            with torch.no_grad():
                q_emb = model.get_image_features(**img_inputs)  # (1, dim)
        except Exception:
            # fallback to text if image decoding fails
            q_inputs = processor(text=[query_text or ""], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                q_emb = model.get_text_features(**q_inputs)
    else:
        # Query is text
        q_inputs = processor(text=[query_text or ""], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            q_emb = model.get_text_features(**q_inputs)  # (1, dim)

    q_emb = q_emb / (q_emb.norm(dim=1, keepdim=True) + 1e-9)  # (1, dim)

    # similarity: (1,dim) @ (dim,n) -> (1,n)
    sims = (q_emb @ doc_embs.T).squeeze(0)  # torch tensor (n,)

    sims_np = sims.cpu().numpy()

    # attach scores (safe; some weaviate objects have metadata attr; use setattr fallback)
    for o, s in zip(objs, sims_np):
        try:
            o.metadata.fclip_score = float(s)
        except Exception:
            try:
                # try placing in properties
                if o.properties is None:
                    o.properties = {}
                o.properties["_fclip_score"] = float(s)
            except Exception:
                setattr(o, "fclip_score", float(s))

    # order descending
    order = np.argsort(-sims_np)
    return [objs[i] for i in order]

def do_hybrid_search(
    client: weaviate.WeaviateClient,
    text_query: str,
    text_vec: Optional[np.ndarray],
    image_vec: Optional[np.ndarray],
    alpha: float,
    limit: int,
    offset: int,
    filters: Optional[Filter],
):
    # Decide the effective alpha per the spec
    if image_vec is not None:
        effective_alpha = 0.60
        primary_vec = image_vec
        target_vector = "image_vector"
    else:
        # Text only → keyword-heavy; keep closer to 0 while respecting user slider
        effective_alpha = 0
        primary_vec = text_vec
        target_vector = "text_vector"

    q = client.collections.get(COLLECTION).query.hybrid(
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
    return q.objects


def result_cards(objs, cols=4, add_to_seen=True):
    grid_cols = st.columns(cols)
    for i, obj in enumerate(objs):
        col = grid_cols[i % cols]
        with col:
            props = obj.properties or {}
            title = props.get("title", "(no title)")
            brand = props.get("brand", "")
            price = props.get("price", "")
            img = props.get("gcs_image_path")
            url = props.get("product_url")
            cat = props.get("category")
            subcat = props.get("subcategory")
            subsubcat = props.get("subsubcategory")
            description = props.get("description")
            caption = props.get("image_caption")

            if img:
                st.image(img, use_container_width=True)
            st.markdown(f"**{title}**")
            meta = " ".join([x for x in [brand, f"€{price}" if price != "" else None] if x])
            if meta:
                st.caption(meta)
            if hasattr(obj.metadata, "score") and obj.metadata.score is not None:
                st.caption(f"Weaviate score: {obj.metadata.score:.3f}")
            if url:
                st.link_button("View", url, use_container_width=True)
            if cat:
                st.caption(f"Category: {cat}>{subcat}>{subsubcat}")
            if description:
                st.caption(f"Description: {description}")
            if caption:
                st.caption(f"Image caption: {caption}")

        if add_to_seen:
            st.session_state.seen_ids.add(obj.uuid)


# =============================================================
# Sidebar Controls
# =============================================================

st.sidebar.header("Search Controls")
mode = st.sidebar.radio("Mode", ["Single Items", "Outfit Builder"], horizontal=False)

search_type = st.sidebar.selectbox(
    "Search with",
    ["Text", "Image", "Text + Image"],
)

gender = st.sidebar.selectbox("Gender", ["any", "male", "female", "unisex"], index=0)
selected_categories = st.sidebar.multiselect("Categories (Single Items)", [
    "Tops","Bottoms","Shoes","Outerwear","Jewelry & Watches","Accessories",
    "Activewear","Suits & Tailoring","Underwear","Bags","Dresses & One-Pieces","Lingerie/Underwear",
    "Small Leather Goods", "Swimwear"
], default=[])
brand_filter = st.sidebar.text_input("Brand contains")

# Reranker knob
topk_for_rerank = st.sidebar.slider("Rerank top-k", 10, 100, 40, 5)

# Outfit budget controls
st.sidebar.markdown("---")
selected_tier = st.sidebar.selectbox("Budget tier", ["Budget", "Mid", "Premium", "Luxury"], index=1)
user_budget = st.sidebar.number_input("Budget (EUR)", min_value=10, max_value=5000, value=350)

# For Single Items only (kept for backwards-compat debug)
limit = st.sidebar.slider("Results per batch", 4, 24, 12, 1)

# =============================================================
# Main Area
# =============================================================

st.title("⚡ Estyl Multimodal Fashion Search")
st.caption("Hybrid Weaviate search + CLIP reranking. Compose full looks under a budget.")

text_query = st.text_input(
    "Describe what you want (style, color, event, vibe)",
    placeholder="e.g., blue tee, off-white pants, white sneakers, glasses",
    value="",
)

uploaded_image = None
if search_type in ("Image", "Text + Image"):
    uploaded_file = st.file_uploader("Or drop an image for visual search", type=["jpg","jpeg","png","webp"]) 
    if uploaded_file:
        uploaded_image = uploaded_file.read()
        st.image(uploaded_image, caption="Query image", use_container_width=False)

# Make vectors
text_vec = embed_text_cached(text_query) if text_query.strip() and search_type in ("Text", "Text + Image") else None
image_vec = embed_image_cached(uploaded_image) if uploaded_image and search_type in ("Image", "Text + Image") else None

# Build filters for single items (category filter comes from sidebar)
price_min, price_max = None, None
if selected_categories and len(selected_categories) == 1 and selected_categories[0] in BUDGET_TIERS:
    cat = selected_categories[0]
    price_min, price_max = BUDGET_TIERS[cat][selected_tier]
    if user_budget:
        price_max = min(price_max, user_budget) if price_max else user_budget
else:
    # no category or multiple categories → just cap by user budget
    if user_budget:
        price_max = user_budget

filters_single = build_filters(
    gender=None if gender == "any" else gender,
    categories=selected_categories,
    price_min=price_min,
    price_max=price_max,
    brand_substr=brand_filter.strip() or None,
    exclude_ids=list(st.session_state.seen_ids) if st.session_state.seen_ids else None,
)

# Signature to reset seen/offset when the query changes materially
sig = signature_for_query(mode, text_query, uploaded_image is not None, {
    "gender": gender,
    "cats": tuple(selected_categories),
    "brand": brand_filter,
    "tier": selected_tier,
    "outfit_budget": user_budget,
})
if sig != st.session_state.last_query_sig:
    st.session_state.last_query_sig = sig
    st.session_state.seen_ids = set()
    st.session_state.offset = 0

client = get_client()

# =============================================================
# Mode: Single Items
# =============================================================
def get_tier_from_budget(category, budget):
    if category not in BUDGET_TIERS:
        return None, None, None
    for tier, (lo, hi) in BUDGET_TIERS[category].items():
        if (budget >= lo) and (hi is None or budget <= hi):
            return tier, lo, hi
    return None, None, None

if mode == "Single Items":
    c1, c2 = st.columns([3,1])
    with c1:
        run = st.button("Search", type="primary")
    with c2:
        refine = st.button("Refine (no repeats)")

    if run or refine:
        # --- Budget sanity check vs tier ---
        if selected_categories and len(selected_categories) == 1:
            cat = selected_categories[0]
            tier, lo, hi = get_tier_from_budget(cat, user_budget)
            if tier is None:
                st.warning(f"⚠️ No results under €{user_budget} for {cat}. Try raising budget.")
            else:
                st.caption(f"🎯 Your budget €{user_budget} falls in the {tier} tier for {cat}.")
        if refine:
            st.session_state.offset += limit
        t0 = time.time()
        with st.spinner("Searching…"):
            objs = do_hybrid_search(
                client=client,
                text_query=text_query,
                text_vec=text_vec,
                image_vec=image_vec,
                alpha=0.0,  # ignored internally if image present / text-only
                limit=max(limit, topk_for_rerank),
                offset=st.session_state.offset,
                filters=filters_single,
            )
            # Rerank the top-k client-side
            objs = fashionclip_rerank(objs[:topk_for_rerank], text_query, uploaded_image)
            objs = objs[:limit]
        took = (time.time() - t0) * 1000
        st.caption(f"Returned {len(objs)} results in {took:.0f} ms (post-rerank)")
        if not objs:
            st.info("No results. Try relaxing filters.")
        else:
            result_cards(objs, cols=4, add_to_seen=True)

# =============================================================
# Mode: Outfit Builder  (fixed: robust planner + fast retrieval + budgeted assembly)
# =============================================================
else:
    st.write("Build complete looks that obey your budget tier + total budget. (Planner-optional composing)")

    num_outfits = st.slider("How many outfits?", 1, 6, 3, 1)
    articles = st.slider(
        "Articles per outfit", 2, 5, 3, 1,
        help="2→Tops+Bottoms; 3→+Shoes; 4→+Outerwear; ≥5→+Accessory"
    )
    per_cat_candidates = st.slider(
        "Per-category candidates", 1, 10, 5, 1,
        help="How many results to fetch per category before picking"
    )
    build_btn = st.button("Compose Outfits", type="primary")

    # --- Helpers specific to Outfit Builder ---
    import itertools
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def price_bounds_for(cat: str, tier: str) -> Tuple[Optional[float], Optional[float]]:
        if cat not in BUDGET_TIERS:
            return None, None
        return BUDGET_TIERS[cat][tier]

    def build_category_filters(cat: str, extra_price_cap: Optional[float] = None) -> Filter:
        lo, hi = price_bounds_for(cat, selected_tier)
        # Optional tighter cap per category (from budget split)
        if extra_price_cap is not None:
            hi = min(hi, extra_price_cap) if hi is not None else extra_price_cap
        return build_filters(
            gender=None if gender == "any" else gender,
            categories=[cat],
            price_min=lo,
            price_max=hi,
            brand_substr=(brand_filter.strip() or None),
        )

    # Map free-form keys to canonical category names used in Weaviate
    _CAT_SYNONYMS = {
        "tops": "Tops", "shirt": "Tops", "t-shirt": "Tops", "tee": "Tops", "blouse": "Tops",
        "bottoms": "Bottoms", "pants": "Bottoms", "trousers": "Bottoms", "jeans": "Bottoms", "skirt": "Bottoms", "shorts": "Bottoms",
        "shoes": "Shoes", "sneakers": "Shoes", "boots": "Shoes", "heels": "Shoes", "sandals": "Shoes", "loafers": "Shoes",
        "outerwear": "Outerwear", "jacket": "Outerwear", "coat": "Outerwear", "blazer": "Outerwear", "hoodie": "Outerwear",
        "accessories": "Accessories", "watch": "Accessories", "belt": "Accessories", "hat": "Accessories",
        "scarf": "Accessories", "sunglasses": "Accessories", "glasses": "Accessories", "tie": "Accessories",
    }

    def canonicalize_category(s: str) -> Optional[str]:
        if not s:
            return None
        k = str(s).strip().lower()
        return _CAT_SYNONYMS.get(k, s) if s in CATEGORY_OPTIONS else _CAT_SYNONYMS.get(k, None)

    def lm_fallback_plan(num_outfits, cats, text_query, brand_hint):
        """Deterministic local plan that guarantees canonical categories."""
        plans = []
        base = (text_query or "").strip()
        for _ in range(num_outfits):
            plan = {}
            for c in cats:
                q = " ".join([w for w in [base, c, brand_hint.strip() if brand_hint else None] if w])
                plan[c] = q
            plans.append(plan)
        return plans

    def call_llm_plan_safe(event, categories, budget, style_prefs, num_outfits=1):
        """Try OpenAI planner; fall back to None if not configured/usable."""
        try:
            # Prefer OpenAI v1 client if available
            try:
                from openai import OpenAI
                _client = OpenAI()
                resp = _client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You output only valid JSON: a LIST of outfit objects keyed by your provided categories."},
                        {"role": "user", "content": f"""
Create {num_outfits} outfit plans as a JSON LIST. Use these exact categories only:
{categories}

Constraints:
- Event: {event}
- Total budget (EUR): {budget}
- Style/brand hints: {style_prefs}

Output example (LIST):
[
  {{"Tops": "white oxford shirt, slim fit", "Bottoms": "navy tailored trousers", "Shoes": "black leather oxford shoes"}}
]
"""}
                    ],
                    temperature=0.5,
                )
                raw = resp.choices[0].message.content.strip()
            except Exception:
                # Older SDKs: try module-level call; still fully optional
                raw = None

            if not raw:
                return None

            # Parse + canonicalize
            data = json.loads(raw)
            if isinstance(data, dict) and "outfits" in data:
                data = data["outfits"]
            if isinstance(data, dict):
                data = [data]

            plans = []
            for outfit in (data or []):
                if not isinstance(outfit, dict):
                    continue
                plan = {}
                for k, v in outfit.items():
                    cat = canonicalize_category(k) or k
                    if cat in categories:
                        plan[cat] = v if isinstance(v, str) else " ".join(v) if isinstance(v, list) else str(v)
                # Ensure we only keep requested categories in order
                plan = {c: plan.get(c, f"{event} {c} {style_prefs or ''}".strip()) for c in categories}
                plans.append(plan)
            return plans or None
        except Exception:
            return None

    def compute_category_caps(cats_in_outfit, total_budget):
        weights = []
        for c in cats_in_outfit:
            w = OUTFIT_WEIGHTS.get(c)
            weights.append(w if w is not None else 1.0)
        s = sum(weights) or 1.0
        return {c: float(total_budget * (w / s)) for c, w in zip(cats_in_outfit, weights)}

    def normalize_price(x) -> float:
        try:
            if x is None:
                return 0.0
            if isinstance(x, (int, float)):
                return float(x)
            # strip currency symbols if any
            return float(str(x).replace("€", "").replace("$", "").replace(",", "").strip())
        except Exception:
            return 0.0

    def score_rank_order(index: int, total: int) -> float:
        # Simple rank-based score: top=1.0 → 0.0
        if total <= 1:
            return 1.0
        return 1.0 - (index / (total - 1))

    # Main composition flow
    if build_btn:
        # Categories to include based on slider (stable & canonical)
        cats = OUTFIT_ORDER_BY_N[5 if articles >= 5 else articles]

        # Try LLM; fall back to deterministic local planner that guarantees canonical keys
        plan = call_llm_plan_safe(
            event=text_query,
            categories=cats,
            budget=user_budget,
            style_prefs=brand_filter.strip(),
            num_outfits=num_outfits,
        ) or lm_fallback_plan(num_outfits, cats, text_query, brand_filter)

        all_outfits = []
        valid_outfits = []
        near_misses = []  # best over-budget picks to show if nothing fits
        used_ids = set()

        t0 = time.time()
        with st.spinner("Composing outfits…"):
            for outfit_idx in range(min(num_outfits, len(plan))):
                outfit_plan = plan[outfit_idx]  # dict: { "Tops": "...", ... }
                cats_in_this = list(outfit_plan.keys())
                caps = compute_category_caps(cats_in_this, user_budget)

                # --- parallel retrieval per category ---
                def _retrieve_for_cat(cat_and_query):
                    cat, query_text_local = cat_and_query
                    # Tighten filter with per-category cap
                    f = build_category_filters(cat, extra_price_cap=caps.get(cat))
                    # Important: pass text_vec=None so BM25 matches THIS query string
                    objs = do_hybrid_search(
                        client=client,
                        text_query=query_text_local,
                        text_vec=None,
                        image_vec=None,
                        alpha=0.0,
                        limit=max(6, per_cat_candidates * 2),
                        offset=0,
                        filters=f,
                    )
                    if not objs:
                        return cat, []
                    # Rerank: image-aware if provided; else lightweight text rerank
                    if uploaded_image:
                        reranked = fashionclip_rerank(
                            objs[:min(len(objs), max(8, per_cat_candidates * 2))],
                            query_text_local,
                            uploaded_image,
                        )
                    else:
                        reranked = lightweight_rerank(
                            objs[:min(len(objs), max(8, per_cat_candidates * 2))],
                            query_text_local,
                            None,
                        )
                    # Keep top per_cat_candidates and drop used IDs
                    filtered = [o for o in reranked if o.uuid not in used_ids][:per_cat_candidates]
                    return cat, filtered

                cat_queries = [(c, (outfit_plan.get(c) or f"{text_query} {c} {brand_filter}".strip())) for c in cats_in_this]

                candidates_map = {}
                with ThreadPoolExecutor(max_workers=min(8, len(cat_queries))) as ex:
                    futures = [ex.submit(_retrieve_for_cat, cq) for cq in cat_queries]
                    for fut in as_completed(futures):
                        cat, cand = fut.result()
                        candidates_map[cat] = cand

                # If any category completely failed, skip this outfit attempt
                if any(len(candidates_map.get(c, [])) == 0 for c in cats_in_this):
                    continue

                # --- choose best combo under the global budget ---
                # Build lists in the fixed category order to keep consistency
                cand_lists = [candidates_map[c] for c in cats_in_this]
                sizes = [len(lst) for lst in cand_lists]
                # Bounded product: at worst 10^5=100k (but UI slider caps at 10)
                best_combo = None
                best_score = -1e9
                best_price = 1e12

                def item_price(o):
                    return normalize_price((o.properties or {}).get("price", 0))

                # Pre-compute rank-based scores per category list
                rank_scores = []
                for lst in cand_lists:
                    n = max(1, len(lst))
                    rank_scores.append({o.uuid: score_rank_order(i, n) for i, o in enumerate(lst)})

                for combo in itertools.product(*cand_lists):
                    # Ensure no duplicates across categories (paranoid; usually distinct)
                    uuids = {o.uuid for o in combo}
                    if len(uuids) < len(combo):
                        continue
                    total_price = sum(item_price(o) for o in combo)

                    # Score: sum rank-based, lightly penalize price pressure vs cat cap
                    score = 0.0
                    for cat, o in zip(cats_in_this, combo):
                        s_rank = rank_scores[cats_in_this.index(cat)][o.uuid]
                        cap = caps.get(cat) or (user_budget / len(cats_in_this))
                        s_price_penalty = 0.02 * (item_price(o) / max(1.0, cap))
                        score += (s_rank - s_price_penalty)

                    if total_price <= user_budget:
                        # Prefer higher score; tie-break on lower price
                        if (score > best_score) or (score == best_score and total_price < best_price):
                            best_score = score
                            best_combo = combo
                            best_price = total_price

                if best_combo is None:
                    # No combo under budget; pick the best overall (min overage)
                    best_over = None
                    best_over_gap = 1e12
                    best_over_score = -1e9
                    for combo in itertools.product(*cand_lists):
                        uuids = {o.uuid for o in combo}
                        if len(uuids) < len(combo):
                            continue
                        total_price = sum(item_price(o) for o in combo)
                        gap = total_price - user_budget
                        # reuse score calc
                        score = 0.0
                        for cat, o in zip(cats_in_this, combo):
                            s_rank = rank_scores[cats_in_this.index(cat)][o.uuid]
                            cap = caps.get(cat) or (user_budget / len(cats_in_this))
                            s_price_penalty = 0.02 * (item_price(o) / max(1.0, cap))
                            score += (s_rank - s_price_penalty)
                        if gap < best_over_gap or (gap == best_over_gap and score > best_over_score):
                            best_over_gap = gap
                            best_over_score = score
                            best_over = (combo, total_price)

                    if best_over:
                        combo, total_price = best_over
                        near_misses.append((list(combo), total_price))
                        # Still reserve IDs to reduce duplication in subsequent outfits
                        for o in combo:
                            used_ids.add(o.uuid)
                        all_outfits.append((list(combo), total_price))
                    continue

                # Record best valid outfit, mark items as used
                final_list = list(best_combo)
                for o in final_list:
                    used_ids.add(o.uuid)
                valid_outfits.append((final_list, best_price))
                all_outfits.append((final_list, best_price))

        took = (time.time() - t0) * 1000

        st.caption(f"Composed {len(all_outfits)} outfit(s) in {took:.0f} ms")
        if not valid_outfits and near_misses:
            st.warning("No outfit fit the budget exactly. Showing the closest matches over budget.")
            # Show up to 2 near-misses
            for idx, (pieces, tot_price) in enumerate(sorted(near_misses, key=lambda x: x[1])[:2], start=1):
                st.subheader(f"Near-match {idx} — Total: €{tot_price:.0f} (budget €{user_budget})")
                cols = st.columns(max(2, len(pieces)))
                for i, obj in enumerate(pieces):
                    with cols[i % len(cols)]:
                        result_cards([obj], cols=1, add_to_seen=False)
                st.divider()
        elif not valid_outfits:
            st.info("Couldn’t compose outfits. Try increasing the budget, switching tier, or widening brand hints.")
        else:
            for idx, (pieces, tot_price) in enumerate(valid_outfits, start=1):
                st.subheader(f"Outfit {idx} — Total: €{tot_price:.0f} (budget ≤ €{user_budget})")
                cols = st.columns(max(2, len(pieces)))
                for i, obj in enumerate(pieces):
                    with cols[i % len(cols)]:
                        result_cards([obj], cols=1, add_to_seen=False)
                st.divider()


# =============================================================
# Footer & Debug
# =============================================================

with st.expander("Debug (session)"):
    st.write({
        "seen_ids_count": len(st.session_state.seen_ids),
        "offset": st.session_state.offset,
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "collection": COLLECTION,
    })

st.caption("Tips: Image query forces alpha=0.6 (visual). Text-only clamps alpha≈0.15 for keyword-heavy retrieval. Reranker is CLIP-based.")
