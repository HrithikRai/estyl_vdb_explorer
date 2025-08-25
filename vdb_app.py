# streamlit_outfit_search_app.py
# -------------------------------------------------------------
# Fast, reactive Streamlit app for multimodal search on Weaviate
# - Two modes: Single Items vs Outfit Builder
# - Text / Image / Text+Image queries
# - Dialable knobs (alpha, limits, price, gender, category, etc.)
# - "Refine" to get fresh results (no repeats in the session)
# - Optimized with caching and lightweight rendering
# -------------------------------------------------------------

import os
import io
import time
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import streamlit as st
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from weaviate.classes.query import Rerank, MetadataQuery


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

@st.cache_data(show_spinner=False, ttl=60*30)
def embed_text_cached(text: str) -> np.ndarray:
    model, processor, device = get_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        vec = model.get_text_features(**inputs).cpu().numpy().flatten().astype(np.float32)
    # Normalize for cosine-like similarity stability
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
    "Activewear","Suits & Tailoring","Underwear","Bags","Dresses & One-Pieces","Lingerie, Underwear & Sleep",
    "Small Leather Goods", "Swimwear"
]

OUTFIT_CATEGORIES = [
    ("Top", "Tops"),
    ("Bottom", "Bottoms"),
    ("Shoes", "Shoes"),
    ("Outerwear", "Outerwear"),
    ("Accessory", "Jewelry & Watches"),
]


def signature_for_query(mode: str, text: str, has_image: bool, knobs: Dict) -> Tuple:
    """Make a signature tuple for the current search to track repeats per distinct query."""
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
        # weaviate v4 supports Like for wildcard matching
        parts.append(Filter.by_property("brand").like(f"*{brand_substr}*"))

    # Exclude already seen items if the client supports by_id().not_in([...])
    if exclude_ids:
        try:
            parts.append(Filter.by_id().not_in(exclude_ids))
        except Exception:
            # Fallback: no id-filter support; rely on offset handled by "Refine" as a backup
            pass

    if not parts:
        return None
    return Filter.all_of(parts)


def do_hybrid_search(
    client: weaviate.WeaviateClient,
    text_query: str,
    text_vec: Optional[np.ndarray],
    image_vec: Optional[np.ndarray],
    alpha: float,
    limit: int,
    offset: int,
    filters: Optional[Filter],
    rerank_query: Optional[str] = None,
    rerank_prop: Optional[str] = None,
):
    primary_vec = text_vec if text_vec is not None else image_vec
    target_vector = "image_vector" if image_vec is not None else "text_vector"

    q = client.collections.get(COLLECTION).query.hybrid(
        query=text_query or "",
        vector=primary_vec.tolist() if primary_vec is not None else None,
        alpha=float(alpha),
        target_vector=target_vector,
        query_properties=QUERY_PROPS,
        limit=int(limit),
        offset=int(offset),
        filters=filters,
        return_properties=RETURN_PROPS,
        return_metadata=MetadataQuery(score=True) if rerank_query else None,
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
            # metadata
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
                st.caption(f"Rerank score: {obj.metadata.score:.3f}")
            if url:
                st.link_button("View", url, use_container_width=True)
            if cat:
                st.caption(f"Category: {cat}>{subcat}>{subsubcat}")
            if description:
                st.caption(f"Description: {description}")
            if caption:
                st.caption(f"Description: {caption}")


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
selected_categories = st.sidebar.multiselect("Categories", CATEGORY_OPTIONS, default=[])
price_min, price_max = st.sidebar.slider("Price Range", 0, 1000, (0, 1000))
brand_filter = st.sidebar.text_input("Brand contains")

alpha = st.sidebar.slider("Hybrid alpha (0=keyword, 1=vector)", 0.0, 1.0, 0.5, 0.05)
limit = st.sidebar.slider("Results per batch", 4, 24, 12, 1)


# Advanced: pagination / refine offset
if "offset" not in st.session_state:
    st.session_state.offset = 0

# =============================================================
# Main Area
# =============================================================

st.title("⚡ Estyl Multimodal Fashion Search")
st.caption("Blazing-fast hybrid search over your Weaviate fashion catalog. Find single pieces or compose full outfits.")

text_query = ""
text_query = st.text_input("Describe what you want (style, color, event, vibe)", placeholder="e.g., blue tee, off-white pants, white sneakers, glasses", value=text_query)

uploaded_image = None
if search_type in ("Image", "Text + Image"):
    uploaded_file = st.file_uploader("Or drop an image for visual search", type=["jpg","jpeg","png","webp"]) 
    if uploaded_file:
        uploaded_image = uploaded_file.read()
        st.image(uploaded_image, caption="Query image", use_container_width=False)

# Make vectors
text_vec = embed_text_cached(text_query) if text_query.strip() and search_type in ("Text", "Text + Image") else None
image_vec = embed_image_cached(uploaded_image) if uploaded_image and search_type in ("Image", "Text + Image") else None

# Build filters
filters = build_filters(
    gender=None if gender == "any" else gender,
    categories=selected_categories,
    price_min=float(price_min) if price_min is not None else None,
    price_max=float(price_max) if price_max is not None else None,
    brand_substr=brand_filter.strip() or None,
    exclude_ids=list(st.session_state.seen_ids) if st.session_state.seen_ids else None,
)

# Signature to reset seen/offset when the query changes materially
sig = signature_for_query(mode, text_query, uploaded_image is not None, {
    "gender": gender,
    "cats": tuple(selected_categories),
    "price_min": price_min,
    "price_max": price_max,
    "brand": brand_filter,
    "alpha": alpha,
    "search_type": search_type,
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
                alpha=alpha,
                limit=limit,
                offset=st.session_state.offset,
                filters=filters,
            )
        took = (time.time() - t0) * 1000
        st.caption(f"Returned {len(objs)} results in {took:.0f} ms")
        if not objs:
            st.info("No results. Try relaxing filters or changing alpha.")
        else:
            result_cards(objs, cols=4, add_to_seen=True)

# =============================================================
# Mode: Outfit Builder
# =============================================================
else:
    st.write("Build a complete look across categories.")
    per_cat_limit = st.slider("Per-piece candidates", 1, 8, 3, 1)
    build_btn = st.button("Compose Outfit", type="primary")

    if build_btn:
        chosen = []
        t0 = time.time()
        with st.spinner("Composing outfit…"):
            for label, cat in OUTFIT_CATEGORIES:
                f = build_filters(
                    gender=None if gender == "any" else gender,
                    categories=[cat],
                    price_min=float(price_min) if price_min is not None else None,
                    price_max=float(price_max) if price_max is not None else None,
                    brand_substr=brand_filter.strip() or None,
                    exclude_ids=list(st.session_state.seen_ids) if st.session_state.seen_ids else None,
                )
                objs = do_hybrid_search(
                client=client,
                text_query=text_query,
                text_vec=text_vec,
                image_vec=image_vec,
                alpha=alpha,
                limit=limit,
                offset=st.session_state.offset,
                filters=filters,

            )
                if objs:
                    # Pick the first unseen
                    for o in objs:
                        if o.uuid not in st.session_state.seen_ids:
                            chosen.append((label, o))
                            st.session_state.seen_ids.add(o.uuid)
                            break

        took = (time.time() - t0) * 1000
        st.caption(f"Composed {len(chosen)} pieces in {took:.0f} ms")

        if not chosen:
            st.info("Couldn’t compose an outfit. Try different filters or query.")
        else:
            # Render outfit in columns
            cols = st.columns(len(chosen))
            for i, (label, obj) in enumerate(chosen):
                with cols[i]:
                    st.subheader(label)
                    result_cards([obj], cols=1, add_to_seen=False)

            st.divider()
            st.button("More like this (refine)")

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

st.caption("\nTip: tune alpha — lower values lean on keywords; higher values lean on embeddings. Use Refine to avoid repeats.")
