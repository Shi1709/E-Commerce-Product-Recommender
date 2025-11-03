# force rebuild
import os
import random
import textwrap
from typing import List, Dict, Set, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---------------- Category images ----------------
CATEGORY_IMAGES = {
    "Electronics": [
        "https://plus.unsplash.com/premium_photo-1679079456083-9f288e224e96?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8RWxlY3Ryb25pY3N8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1603732551658-5fabbafa84eb?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8RWxlY3Ryb25pY3N8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1611186871348-b1ce696e52c9?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTR8fEVsZWN0cm9uaWNzfGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1591799264318-7e6ef8ddb7ea?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjR8fEVsZWN0cm9uaWNzfGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1620783770629-122b7f187703?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjN8fEVsZWN0cm9uaWNzfGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1586062129117-08db958ba215?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MzJ8fEVsZWN0cm9uaWNzfGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1649859398731-d3c4ebca53fc?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MzV8fEVsZWN0cm9uaWNzfGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600"
    ],
    "Fashion": [
        "https://images.unsplash.com/photo-1571513800374-df1bbe650e56?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8RmFzaGlvbnxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
        "https://plus.unsplash.com/premium_photo-1675186049419-d48f4b28fe7c?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OXx8RmFzaGlvbnxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1603400521630-9f2de124b33b?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mzl8fEZhc2hpb258ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://plus.unsplash.com/premium_photo-1664202526047-405824c633e7?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Njl8fEZhc2hpb258ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1558303522-d7a2bdfdbd82?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTE0fHxGYXNoaW9ufGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1551232864-3f0890e580d9?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTE4fHxGYXNoaW9ufGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1523398002811-999ca8dec234?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTMyfHxGYXNoaW9ufGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600"
    ],
    "Home": [
        "https://images.unsplash.com/photo-1583847268964-b28dc8f51f92?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8SG9tZXxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
        "https://plus.unsplash.com/premium_photo-1661964014750-963a28aeddea?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8SG9tZXxlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTR8fEhvbWV8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1484154218962-a197022b5858?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fEhvbWV8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1532372576444-dda954194ad0?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjB8fEhvbWV8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1556020685-ae41abfc9365?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mjd8fEhvbWV8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1519643381401-22c77e60520e?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NDh8fEhvbWV8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600"
    ],
    "Sports": [
        "https://plus.unsplash.com/premium_photo-1685303469251-4ee0ea014bb3?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8U3BvcnRzfGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600",
        "https://plus.unsplash.com/premium_photo-1676634832558-6654a134e920?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTN8fFNwb3J0c3xlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1535131749006-b7f58c99034b?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjJ8fFNwb3J0c3xlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
        "https://plus.unsplash.com/premium_photo-1666913667023-4bfd0f6cff0a?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NDV8fFNwb3J0c3xlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1560089000-7433a4ebbd64?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NTZ8fFNwb3J0c3xlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1516820612845-a13894592046?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Nzl8fFNwb3J0c3xlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1512746804203-e53e69406f93?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTAwfHxTcG9ydHN8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600"
    ],
    "Beauty": [
        "https://images.unsplash.com/photo-1598528738936-c50861cc75a9?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8YmVhdXR5JTIwcHJvZHVjdHN8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1612817288484-6f916006741a?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8YmVhdXR5JTIwcHJvZHVjdHN8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://plus.unsplash.com/premium_photo-1679046948726-72f7d53296c5?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OXx8YmVhdXR5JTIwcHJvZHVjdHN8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTF8fGJlYXV0eSUyMHByb2R1Y3RzfGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600",
        "https://plus.unsplash.com/premium_photo-1684407616442-8d5a1b7c978e?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjF8fGJlYXV0eSUyMHByb2R1Y3RzfGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1598460880248-71ec6d2d582b?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjZ8fGJlYXV0eSUyMHByb2R1Y3RzfGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1556229010-aa3f7ff66b24?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MzB8fGJlYXV0eSUyMHByb2R1Y3RzfGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600"
    ],
    "Books": [
        "https://images.unsplash.com/photo-1512820790803-83ca734da794",
        "https://images.unsplash.com/photo-1519791883288-dc8bd696e667?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8ODh8fGJvb2t8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600",
        "https://images.unsplash.com/photo-1577627444534-b38e16c9d796?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=736",
        "https://images.unsplash.com/photo-1506880018603-83d5b814b5a6?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=1074",
        "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=1074",
        "https://images.unsplash.com/photo-1543002588-bfa74002ed7e?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=687",
        "https://images.unsplash.com/photo-1544947950-fa07a98d237f?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=687"
    ],
"Groceries": [
    "https://images.unsplash.com/photo-1542838132-92c53300491e?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8Z3JvY2VyaWVzfGVufDB8fDB8fHww&auto=format&fit=crop&q=60&w=600",
    "https://media.istockphoto.com/id/613121170/photo/hot-7-grain-breakfast-cereal-with-yogurt-and-fresh-fruit.webp?a=1&b=1&s=612x612&w=0&k=20&c=sJ_1bIMQs9-hGiR0EzJ7huNQQ8KOd-A11xRUZg5hDvg=",
    "https://images.unsplash.com/photo-1614735241165-6756e1df61ab?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mjd8fGdyb2Nlcmllc3xlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
    "https://images.unsplash.com/photo-1614907634002-65ac4cb74acb?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NDd8fGdyb2Nlcmllc3xlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
    "https://images.unsplash.com/photo-1585735633320-d24595a213a1?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NTB8fGdyb2Nlcmllc3xlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
    "https://images.unsplash.com/photo-1691476093794-d923c08df935?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Nzh8fGdyb2Nlcmllc3xlbnwwfHwwfHx8MA%3D%3D&auto=format&fit=crop&q=60&w=600",
    "https://images.unsplash.com/photo-1566454825481-4e48f80aa4d7?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTA4fHxncm9jZXJpZXN8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&q=60&w=600"
     ]
}
FALLBACK_IMAGES = [u for imgs in CATEGORY_IMAGES.values() for u in imgs]

# ---------------- Gemini helper (optional) ----------------
try:
    from google import genai
except Exception:
    genai = None

def init_genai(api_key: str = None):
    if genai is None:
        raise RuntimeError("google.genai SDK not found. Install google-generativeai.")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    return genai.Client()

def generate_explanation(client, model: str, prompt: str) -> str:
    resp = client.models.generate_content(model=model, contents=prompt)
    text = getattr(resp, "text", None)
    if not text:
        try:
            text = resp.output[0].content[0].text
        except Exception:
            text = str(resp)
    return text

def make_prompt_for_explanation(user_summary: str, product_row: pd.Series) -> str:
    return textwrap.dedent(f"""
        You are an assistant that writes concise, user-focused explanations for product recommendations.
        User summary: {user_summary}

        Product:
        - Name: {product_row.get('name', '')}
        - Category: {product_row.get('category', '')}
        - Price: {product_row.get('price', '')}
        - Description: {product_row.get('description', '')}

        Write a short paragraph (3–5 sentences) titled "Why we recommend this" that:
        - Refers to the user's preferences,
        - Highlights key product benefits,
        - Uses friendly, natural tone.
    """).strip()

# ---------------- Utilities ----------------
def normalize_category(cat: str) -> str:
    if not isinstance(cat, str):
        return "Home"
    c = cat.strip().lower()
    if any(k in c for k in ("shoe","foot","sneaker","footwear","running","run","trail","sportswear")):
        return "Sports"
    if any(k in c for k in ("cloth","fashion","apparel","jacket","shorts","tshirt","apparel")):
        return "Fashion"
    if any(k in c for k in ("phone","elect","camera","ssd","tv","watch","tracker","gps","headphone","console","appliance")):
        return "Electronics"
    if any(k in c for k in ("airfry","oven","kitchen","home","furnitur","appliance")):
        return "Home"
    if any(k in c for k in ("beauty","skincare","care","tooth")):
        return "Beauty"
    if any(k in c for k in ("book","novel","paperback","kindle")):
        return "Books"
    if any(k in c for k in ("grocery","food","snack","tea","coffee")):
        return "Groceries"
    return "Home"

def assign_category_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Always overwrite image_url with a category-based image (silent autofill).
    """
    df = df.copy()
    if "image_url" not in df.columns:
        df["image_url"] = ""
    for idx, row in df.iterrows():
        cat = normalize_category(row.get("category", ""))
        imgs = CATEGORY_IMAGES.get(cat, FALLBACK_IMAGES)
        choice = random.choice(imgs) if imgs else random.choice(FALLBACK_IMAGES)
        # always overwrite regardless of existing value
        df.at[idx, "image_url"] = choice
    return df

# ---------------- Recommender core ----------------
def build_tfidf_matrix(df: pd.DataFrame) -> Tuple[TfidfVectorizer, any]:
    docs = df["text_blob"].fillna("").astype(str).tolist()
    vect = TfidfVectorizer(stop_words="english", max_features=6000)
    tfidf = vect.fit_transform(docs)
    return vect, tfidf

def top_similar_indices(tfidf_matrix, target_index:int, top_n:int=10):
    sims = linear_kernel(tfidf_matrix[target_index:target_index+1], tfidf_matrix).flatten()
    sims[target_index] = -1
    order = sims.argsort()[::-1]
    return [(int(i), float(sims[i])) for i in order[:top_n]]

def aggregate_user_profile(behaviour: pd.DataFrame, products: pd.DataFrame, user_id: str) -> Dict:
    df = behaviour[behaviour["user_id"] == user_id].copy()
    category_scores: Dict[str, float] = {}
    product_scores: Dict[str, float] = {}
    if df.empty:
        return {"category_scores": {}, "product_scores": {}, "user_summary": ""}
    weight = {"view": 1.0, "purchase": 3.0}
    for _, row in df.iterrows():
        pid = str(row["product_id"])
        evt = str(row.get("event", "view")).lower()
        w = weight.get(evt, 1.0)
        product_scores[pid] = product_scores.get(pid, 0.0) + w
        matched = products[products["id"].astype(str) == pid]
        if not matched.empty:
            cat = normalize_category(matched.iloc[0].get("category", ""))
            category_scores[cat] = category_scores.get(cat, 0.0) + w
    top_cats = sorted(category_scores.items(), key=lambda x: -x[1])[:3]
    cat_text = ", ".join([f"{c} ({int(s)})" for c, s in top_cats])
    top_products = sorted(product_scores.items(), key=lambda x: -x[1])[:5]
    prod_ids = ", ".join([p for p, _ in top_products])
    user_summary = f"User has interacted mainly with categories: {cat_text}. Recent products: {prod_ids}."
    return {"category_scores": category_scores, "product_scores": product_scores, "user_summary": user_summary}

def recommend_for_user(user_id: str, behaviour: pd.DataFrame, products: pd.DataFrame, tfidf, num_recs:int=5, strict:bool=False):
    profile = aggregate_user_profile(behaviour, products, user_id)
    preferred_cats: Set[str] = set(k for k, v in profile["category_scores"].items() if v > 0)
    product_scores = profile["product_scores"]

    seed_product_ids = sorted(product_scores.items(), key=lambda x: -x[1])[:3]
    seed_indices: List[int] = []
    for pid, _ in seed_product_ids:
        matches = products.index[products["id"].astype(str) == pid].tolist()
        if matches:
            seed_indices.append(matches[0])
    if not seed_indices:
        if not behaviour.empty:
            last_pid = str(behaviour.iloc[-1]["product_id"])
            matches = products.index[products["id"].astype(str) == last_pid].tolist()
            if matches:
                seed_indices = [matches[0]]
        if not seed_indices:
            seed_indices = [0]

    all_candidates: Dict[int, float] = {}
    for seed_idx in seed_indices:
        sim_pairs = top_similar_indices(tfidf, seed_idx, top_n=40)
        for idx, sim_score in sim_pairs:
            pid = str(products.iloc[idx]["id"])
            cat = normalize_category(products.iloc[idx].get("category", ""))
            behavior_bonus = product_scores.get(pid, 0.0)
            cat_bonus = 1.5 if (cat in preferred_cats and preferred_cats) else 0.0
            final_score = sim_score + 0.6 * cat_bonus + 0.4 * behavior_bonus
            if strict and cat not in preferred_cats:
                continue
            prev = all_candidates.get(idx, -1.0)
            if final_score > prev:
                all_candidates[idx] = final_score

    if not all_candidates:
        return [], profile
    cand_df = pd.DataFrame([
        {"index": idx, "score": sc, "id": str(products.iloc[idx]["id"]), "name": products.iloc[idx]["name"],
         "category": products.iloc[idx].get("category", ""), "price": products.iloc[idx].get("price",""),
         "description": products.iloc[idx].get("description",""), "image_url": products.iloc[idx].get("image_url","")}
        for idx, sc in all_candidates.items()
    ])
    cand_df = cand_df.sort_values("score", ascending=False)
    results = cand_df.head(num_recs).to_dict(orient="records")
    return results, profile

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="E-commerce Product Recommender", layout="wide")
st.title("E-commerce Product Recommender")

st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input("Gemini API KEY (optional)", type="password")
model_choice = st.sidebar.selectbox("Gemini model (optional)", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"])
strict_mode = st.sidebar.checkbox("Strictly prefer user's categories", value=False)
num_recs = st.sidebar.number_input("Number of recommendations", min_value=1, max_value=10, value=5)

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_products = st.file_uploader("Upload products CSV", type=["csv"])
    if uploaded_products:
        products = pd.read_csv(uploaded_products)
        products["id"] = products["id"].astype(str)
        products["text_blob"] = (products.get("name","").fillna("").astype(str) + " . " +
                                 products.get("category","").fillna("").astype(str) + " . " +
                                 products.get("description","").fillna("").astype(str))
        # silent auto-fill images and ALWAYS overwrite image_url
        products = assign_category_images(products)
        st.success(f"Loaded {len(products)} products.")
        st.dataframe(products.head())
    else:
        st.info("No product CSV uploaded. Click below to load a small demo set.")
        if st.button("Load sample products"):
            data = [
                {"id":"p1","name":"Comfort Running Shoes","category":"Footwear","description":"Lightweight, breathable running shoes.","price":59.99},
                {"id":"p2","name":"Trail Running Shoes","category":"Footwear","description":"Durable shoes with grip for rough terrain.","price":79.99},
                {"id":"p3","name":"Bluetooth Earbuds","category":"Electronics","description":"Wireless earbuds with long battery life.","price":99.99},
                {"id":"p4","name":"Air Purifier","category":"Home Appliance","description":"Cleans indoor air efficiently.","price":149.99},
            ]
            products = pd.DataFrame(data)
            products["id"] = products["id"].astype(str)
            products["text_blob"] = products["name"] + " . " + products["category"] + " . " + products["description"]
            products = assign_category_images(products)
            st.success("Sample products loaded.")
            st.dataframe(products)

with col2:
    uploaded_behaviour = st.file_uploader("Upload user behaviour CSV", type=["csv"])
    if uploaded_behaviour:
        behaviour = pd.read_csv(uploaded_behaviour)
        behaviour.columns = [c.strip() for c in behaviour.columns]
        if "user_id" not in behaviour.columns or "product_id" not in behaviour.columns:
            st.error("Behaviour CSV must contain columns: user_id, product_id, event")
            behaviour = pd.DataFrame(columns=["user_id","product_id","event"])
        else:
            behaviour["user_id"] = behaviour["user_id"].astype(str)
            behaviour["product_id"] = behaviour["product_id"].astype(str)
            behaviour["event"] = behaviour.get("event","view").astype(str).str.lower()
            st.success(f"Loaded {len(behaviour)} behaviour records.")
            st.dataframe(behaviour.tail(10))
    else:
        behaviour = pd.DataFrame(columns=["user_id","product_id","event"])

if "products" not in locals():
    st.stop()

# Build TF-IDF matrix
try:
    vect, tfidf = build_tfidf_matrix(products)
except Exception as e:
    st.error(f"TF-IDF build failed: {e}")
    st.stop()

user_list = behaviour["user_id"].unique().tolist() if not behaviour.empty else []
selected_user = st.selectbox("Select user", options=user_list or ["user_1"])

if st.button("Generate recommendations"):
    client = None
    try:
        if api_key_input:
            client = init_genai(api_key_input)
    except Exception as e:
        st.warning(f"Gemini init failed: {e}")

    recs, profile = recommend_for_user(selected_user, behaviour, products, tfidf, num_recs=int(num_recs), strict=bool(strict_mode))

    if not recs:
        st.warning("No recommendations found.")
    else:
        st.success(f"Top {len(recs)} recommendations for user {selected_user}:")
        for item in recs:
            cols = st.columns([1, 3])
            with cols[0]:
                try:
                    st.image(item.get("image_url", random.choice(FALLBACK_IMAGES)), width=140)
                except Exception:
                    st.write("[image]")
            with cols[1]:
                st.write(f"{item['name']} — {item['category']} — ${item.get('price','')}")
                st.write(item.get("description",""))
                explanation = "(No Gemini — explanation skipped)"
                if client:
                    try:
                        prompt = make_prompt_for_explanation(profile.get("user_summary",""), pd.Series(item))
                        explanation = generate_explanation(client, model_choice, prompt)
                    except Exception as e:
                        explanation = f"(Gemini error: {e})"
                st.markdown("Why we recommend this:")
                st.write(explanation)

    out_df = pd.DataFrame(recs)
    if not out_df.empty:
        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download recommendations CSV", data=csv, file_name=f"recs_{selected_user}.csv", mime="text/csv")

st.caption("Recommender = TF-IDF similarity + behavior-derived category & product boosts. Tune weights in recommend_for_user() to adjust personalization intensity.")



