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
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
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
        effective_alpha = min(alpha, 0.15)
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
max_outfit_budget = st.sidebar.number_input("Total outfit budget (EUR)", min_value=30, max_value=5000, value=350)

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
filters_single = build_filters(
    gender=None if gender == "any" else gender,
    categories=selected_categories,
    price_min=None,
    price_max=None,
    brand_substr=brand_filter.strip() or None,
    exclude_ids=list(st.session_state.seen_ids) if st.session_state.seen_ids else None,
)

# Signature to reset seen/offset when the query changes materially
sig = signature_for_query(mode, text_query, uploaded_image is not None, {
    "gender": gender,
    "cats": tuple(selected_categories),
    "brand": brand_filter,
    "tier": selected_tier,
    "outfit_budget": max_outfit_budget,
})
if sig != st.session_state.last_query_sig:
    st.session_state.last_query_sig = sig
    st.session_state.seen_ids = set()
    st.session_state.offset = 0

client = get_client()

# =============================================================
# Mode: Single Items
# =============================================================

if mode == "Single Items":
    c1, c2 = st.columns([3,1])
    with c1:
        run = st.button("Search", type="primary")
    with c2:
        refine = st.button("Refine (no repeats)")

    if run or refine:
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
# Mode: Outfit Builder
# =============================================================
    st.write("Build complete looks that obey your budget tier + total budget.")
    num_outfits = st.slider("How many outfits?", 1, 6, 3, 1)
    articles = st.slider("Articles per outfit", 2, 5, 3, 1,
                         help="2→Tops+Bottoms; 3→+Shoes; 4→+Outerwear; ≥5→+Accessory")
    per_cat_candidates = st.slider("Per-category candidates", 1, 10, 5, 1,
                                   help="How many results to fetch per category before picking")
    build_btn = st.button("Compose Outfits", type="primary")

    def price_bounds_for(cat: str, tier: str) -> Tuple[Optional[float], Optional[float]]:
        if cat not in BUDGET_TIERS:
            return None, None
        return BUDGET_TIERS[cat][tier]

    def build_category_filters(cat: str) -> Filter:
        lo, hi = price_bounds_for(cat, selected_tier)
        return build_filters(
            gender=None if gender == "any" else gender,
            categories=[cat],
            price_min=lo,
            price_max=hi,
            brand_substr=brand_filter.strip() or None,
        )

    def pick_piece(cat: str, used_ids: set) -> Optional[object]:
        f = build_category_filters(cat)
        objs = do_hybrid_search(
            client=client,
            text_query=text_query,
            text_vec=text_vec,
            image_vec=image_vec,
            alpha=0.0,
            limit=per_cat_candidates,
            offset=0,
            filters=f,
        )
        # remove already used
        objs = [o for o in objs if o.uuid not in used_ids]
        if not objs:
            return None
        # lightweight rerank on this small pool
        objs = fashionclip_rerank(objs, text_query, uploaded_image)
        # randomize a bit so results differ across outfits
        choice = np.random.choice(objs[:min(3, len(objs))])
        used_ids.add(choice.uuid)
        return choice

    if build_btn:
        all_outfits = []
        used_ids = set()  # global dedup across all outfits
        cats = OUTFIT_ORDER_BY_N[5 if articles >= 5 else articles]

        t0 = time.time()
        with st.spinner("Composing outfits…"):
            for _ in range(num_outfits):
                chosen = []
                total_price = 0.0
                for cat in cats:
                    piece = pick_piece(cat, used_ids)
                    if piece:
                        chosen.append(piece)
                        price = float((piece.properties or {}).get("price", 0) or 0)
                        total_price += price
                all_outfits.append((chosen, total_price))

        took = (time.time() - t0) * 1000
        st.caption(f"Generated {len(all_outfits)} outfit(s) in {took:.0f} ms")

        if not any(len(pieces) for pieces, _ in all_outfits):
            st.info("Couldn’t compose outfits. Try a different tier or increase the budget.")
        else:
            for idx, (pieces, tot_price) in enumerate(all_outfits, start=1):
                st.subheader(f"Outfit {idx} — Total: €{tot_price:.0f} (≤ €{max_outfit_budget})")
                cols = st.columns(max(2, len(pieces)))
                for i, obj in enumerate(pieces):
                    with cols[i % len(cols)]:
                        result_cards([obj], cols=1, add_to_seen=False)
                st.divider()
else:
    st.write("Build complete looks that obey your budget tier + total budget. (LLM-assisted composing)")

    num_outfits = st.slider("How many outfits?", 1, 6, 3, 1)
    articles = st.slider("Articles per outfit", 2, 5, 3, 1,
                         help="2→Tops+Bottoms; 3→+Shoes; 4→+Outerwear; 5→+Accessory")
    per_cat_candidates = st.slider("Per-category candidates", 1, 10, 5, 1,
                                   help="How many results to fetch per category before picking")
    build_btn = st.button("Compose Outfits (LLM)", type="primary")

    # Helpers (local redefinitions kept compact so we only change this block)
    def price_bounds_for(cat: str, tier: str) -> Tuple[Optional[float], Optional[float]]:
        if cat not in BUDGET_TIERS:
            return None, None
        return BUDGET_TIERS[cat][tier]

    def build_category_filters(cat: str) -> Filter:
        lo, hi = price_bounds_for(cat, selected_tier)
        return build_filters(
            gender=None if gender == "any" else gender,
            categories=[cat],
            price_min=lo,
            price_max=hi,
            brand_substr=brand_filter.strip() or None,
        )

    import json, re, textwrap, time

    def try_parse_json(s: str):
        # try to find a JSON object/array in a string robustly
        s = s.strip()
        # common case: pure JSON
        try:
            return json.loads(s)
        except Exception:
            pass
        # try to extract first { ... } or [ ... ]
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return None

    def call_llm_plan(num_outfits, articles, cats, text_query, tier, budget, brand_hint):
        """Call an LLM (if OPENAI_API_KEY present) to propose concise retrieval queries per outfit.
           Returns a dict with {"outfits": [ [ { "category": "Tops", "query":"..." }, ... ], ... ] }
           If LLM is unavailable or fails, returns None and caller should fallback to heuristic.
        """
        system = (
            "You are a compact outfit-planning assistant. Output ONLY valid JSON. "
            "Structure: {\"outfits\": [ [ {\"category\":\"Tops\",\"query\":\"short retrieval query\"}, ... ], ... ] }. "
            "Each outfit must contain exactly the provided categories in the given order. "
        )
        user = textwrap.dedent(f"""
            num_outfits: {num_outfits}
            articles_per_outfit: {articles}
            categories_order: {cats}
            user_text_query: \"{text_query}\"
            budget_tier: {tier}
            total_budget_eur: {budget}
            brand_hint: \"{brand_hint}\"

            For each outfit produce concise retrieval queries (1-6 words preferred) that combine the user's text_query + category + any style/color hints.
            Keep outputs short and JSON-valid.
        """)
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
            if not openai.api_key:
                return None
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.15,
                max_tokens=400,
                n=1,
                timeout=10,
            )
            text = resp["choices"][0]["message"]["content"]
            parsed = try_parse_json(text)
            return parsed
        except Exception:
            return None

    def lm_fallback_plan(num_outfits, articles, cats, text_query, tier, budget, brand_hint):
        # Simple deterministic fallback: for each outfit, create queries per category by combining user text + category
        outfits = []
        for _ in range(num_outfits):
            items = []
            for c in cats:
                q = f"{text_query} {c}".strip()
                if brand_hint:
                    q += f" {brand_hint}"
                items.append({"category": c, "query": q})
            outfits.append(items)
        return {"outfits": outfits}

    def compute_category_caps(cats_in_outfit, total_budget):
        # derive normalized weights for only the categories we will pick
        weights = []
        for c in cats_in_outfit:
            w = OUTFIT_WEIGHTS.get(c, None)
            weights.append(w if w is not None else 1.0)
        s = sum(weights) or 1.0
        caps = {c: float(total_budget * (w / s)) for c, w in zip(cats_in_outfit, weights)}
        return caps

    def pick_best_from_candidates(objs, cap_price):
        """Choose the best candidate (reranked order assumed). Prefer <=cap_price; otherwise pick cheapest."""
        if not objs:
            return None
        # filter by price <= cap
        valid = []
        for o in objs:
            try:
                p = float((o.properties or {}).get("price", 0) or 0)
            except Exception:
                p = 0.0
            valid.append((o, p))
        # prefer those under cap, sorted by rerank order already (objs order preserved)
        under = [o for o,pr in valid if pr <= (cap_price if cap_price is not None else 1e12)]
        if under:
            return under[0]
        # else return cheapest
        valid.sort(key=lambda x: x[1])
        return valid[0][0]

    # Main composition flow (LLM plan + retrieval + rerank)
    if build_btn:
        cats = OUTFIT_ORDER_BY_N[5 if articles >= 5 else articles]
        # ask LLM for retrieval-suitable queries
        plan = call_llm_plan(num_outfits, articles, cats, text_query, selected_tier, max_outfit_budget, brand_filter.strip())
        if not plan or "outfits" not in plan:
            plan = lm_fallback_plan(num_outfits, articles, cats, text_query, selected_tier, max_outfit_budget, brand_filter.strip())

        all_outfits = []
        used_ids = set()
        t0 = time.time()
        with st.spinner("LLM planning + retrieval + reranking…"):
            for outfit_idx, outfit_spec in enumerate(plan["outfits"][:num_outfits]):
                chosen = []
                total_price = 0.0
                cats_in_this = [it.get("category") for it in outfit_spec]
                caps = compute_category_caps(cats_in_this, max_outfit_budget)

                for item in outfit_spec:
                    cat = item.get("category")
                    q = item.get("query") or f"{text_query} {cat}"
                    # Build filters: category + tier-based price bounds + brand hint
                    f = build_category_filters(cat)

                    # retrieve a small pool
                    objs = do_hybrid_search(
                        client=client,
                        text_query=q,
                        text_vec=text_vec,
                        image_vec=image_vec,
                        alpha=0.0,
                        limit=max(3, per_cat_candidates),
                        offset=0,
                        filters=f,
                    )
                    # dedupe
                    objs = [o for o in objs if o.uuid not in used_ids]
                    if not objs:
                        # no candidates; skip
                        continue

                    # rerank: prefer FashionCLIP when image query exists, else lightweight text-space rerank
                    if uploaded_image:
                        reranked = fashionclip_rerank(objs[:min(len(objs), max(8, topk_for_rerank))], q, uploaded_image)
                    else:
                        reranked = lightweight_rerank(objs[:min(len(objs), max(8, topk_for_rerank))], q, None)

                    # choose respecting per-category cap
                    cap = caps.get(cat)
                    pick = pick_best_from_candidates(reranked, cap)
                    if pick:
                        chosen.append(pick)
                        used_ids.add(pick.uuid)
                        try:
                            price = float((pick.properties or {}).get("price", 0) or 0)
                        except Exception:
                            price = 0.0
                        total_price += price

                # finalize this outfit
                all_outfits.append((chosen, total_price))

        took = (time.time() - t0) * 1000
        st.caption(f"Generated {len(all_outfits)} outfit(s) in {took:.0f} ms (LLM-assisted)")

        if not any(len(pieces) for pieces, _ in all_outfits):
            st.info("Couldn’t compose outfits. Try a different tier, increase the budget, or relax filters.")
        else:
            for idx, (pieces, tot_price) in enumerate(all_outfits, start=1):
                st.subheader(f"Outfit {idx} — Total: €{tot_price:.0f} (budget ≤ €{max_outfit_budget})")
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
