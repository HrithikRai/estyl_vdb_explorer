from retriever_core import handle_chatbot_request, close_weaviate_client

single_payload = {
      "mode": "single",
      "text_query": "blue tee and white sneakers",
      "image_bytes": None,
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

outfit_payload = {
      "mode": "outfit",
      "event_text": "smart casual dinner",
      "image_bytes": None,
      "gender": "female",
      "brand_hint": "COS, Arket",
      "selected_tier": "Mid",
      "user_budget": 400,
      "num_outfits": 2,
      "articles": 3,
      "per_cat_candidates": 5,
      "reranker": "fashionclip"
    }

results = handle_chatbot_request(outfit_payload)
print(results)
close_weaviate_client()