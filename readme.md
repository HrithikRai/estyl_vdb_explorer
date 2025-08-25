# ⚡ Estyl Multimodal Fashion Search

A **fast, reactive, multimodal Streamlit app** for searching fashion items and composing outfits using **Weaviate + CLIP embeddings**.  
Supports **text, image, or hybrid (text+image) queries**, with refinements, filters, and an **Outfit Builder** mode.

---

## 🚀 Features

- 🔍 **Single Item Search** — find fashion pieces using text, images, or both.
- 🧑‍🤝‍🧑 **Outfit Builder** — compose full looks across categories (tops, bottoms, shoes, outerwear, accessories).
- ⚡ **Refine mode** — ensures new results, avoids repetition.
- 🎚️ **Dialable knobs** — control hybrid alpha, result count, filters (price, gender, category, brand).
- 🖼️ **Image search** — upload reference images for visual similarity.
- 🛠️ **Optimized** — caching, lightweight rendering, fast UX.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – UI framework
- [Weaviate](https://weaviate.io/) – vector database
- [Hugging Face Transformers](https://huggingface.co/) – CLIP embeddings
- [Torch](https://pytorch.org/) – model backend
- [PIL](https://pillow.readthedocs.io/) – image preprocessing
- [dotenv](https://pypi.org/project/python-dotenv/) – environment config

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/estyl-multimodal-search.git
cd estyl-multimodal-search
pip install -r requirements.txt

Env variables ->
WEAVIATE_HOST="your-weaviate-endpoint"
WEAVIATE_API_KEY="your-api-key"
WEAVIATE_COLLECTION="Estyl_articles"