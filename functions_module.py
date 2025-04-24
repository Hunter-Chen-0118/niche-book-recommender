import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from scipy.spatial.distance import cosine
import json

# --- Constants ---
SIZE_BUCKETS = {
    'short':      (0,   250),
    'medium':     (250, 500),
    'long':       (500, 700),
    'extra_long': (700, np.inf)
}

WEIGHTS = {
    'genre':     0.5,
    'length':    0.25,
    'reference': 0.25
}

# --- 1. Data loading ---
def load_data(full_path: str, niche_path: str) -> (pd.DataFrame, pd.DataFrame):
    full_df  = pd.read_excel(full_path)
    niche_df = pd.read_excel(niche_path)
    return full_df, niche_df

# --- 2. Preprocessing ---
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['genre_list']  = df['genre'].str.split(',')
    df['num_pages']   = pd.to_numeric(df['num_pages'], errors='coerce')

    def assign_bucket(pages):
        for label, (lo, hi) in SIZE_BUCKETS.items():
            if lo <= pages < hi:
                return label
        return None

    df['size_bucket'] = df['num_pages'].apply(assign_bucket)
    return df

# --- 4. Build user profile ---
def build_user_profile(prefs: dict, full_df: pd.DataFrame) -> dict:
    # Genre vector
    mlb = MultiLabelBinarizer()
    mlb.fit(full_df['genre_list'].dropna())
    user_genre_vec = mlb.transform([prefs['genres']])[0]

    # Length scaler & target
    scaler = MinMaxScaler()
    scaler.fit(full_df['num_pages'].values.reshape(-1,1))
    lo, hi = SIZE_BUCKETS[prefs['size_bucket']]
    if np.isinf(hi):
        hi = full_df['num_pages'].max()
    midpoint = (lo + hi) / 2
    length_target = scaler.transform([[midpoint]])[0][0]

    # Reference centroid
    ref_vecs = []
    for ref in prefs['references']:
        g_vec   = mlb.transform([ref['genre_list']])[0]
        p_norm  = scaler.transform([[ref['num_pages']]])[0][0]
        ref_vecs.append(np.concatenate([g_vec, [p_norm]]))

    if ref_vecs:
        reference_centroid = np.mean(ref_vecs, axis=0)
    else:
        reference_centroid = np.zeros(len(mlb.classes_) + 1)

    return {
        'mlb':                mlb,
        'scaler':             scaler,
        'genre_vector':       user_genre_vec,
        'length_target':      length_target,
        'reference_centroid': reference_centroid,
        'size_bucket':        prefs['size_bucket']
    }

# --- 5. Score candidates ---
def score_candidates(niche_df: pd.DataFrame, profile: dict, weights: dict) -> pd.DataFrame:
    scored  = niche_df.copy()
    mlb     = profile['mlb']
    scaler  = profile['scaler']
    user_g  = profile['genre_vector']
    bucket  = profile['size_bucket']
    ref_c   = profile['reference_centroid']
    has_ref = np.linalg.norm(ref_c) > 0

    def compute_score(row):
        # Genre match
        book_g   = mlb.transform([row['genre_list']])[0]
        genre_score = (np.dot(book_g, user_g) 
                       / user_g.sum()) if user_g.sum()>0 else 0.0

        # Length match
        pages = row['num_pages']
        lo, hi = SIZE_BUCKETS[bucket]
        if lo <= pages < hi:
            length_score = 1.0
        else:
            span = lo if np.isinf(hi) else hi - lo
            dist = (lo - pages) if pages < lo else (pages - hi)
            length_score = max(0.0, 1 - (dist/span))

        # Reference similarity
        if has_ref:
            book_vec = np.concatenate([
                book_g,
                [scaler.transform([[pages]])[0][0]]
            ])
            sim = 1 - cosine(book_vec, ref_c)
            ref_score = 0.0 if np.isnan(sim) else sim
        else:
            ref_score = 0.0

        return (weights['genre']*genre_score
              + weights['length']*length_score
              + weights['reference']*ref_score)

    scored['score'] = scored.apply(compute_score, axis=1)
    return scored

# --- 6. Select top N (tie‐break by average_rating) ---
def select_top_n(scored_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        scored_df
        .sort_values(by=['score', 'average_rating'], ascending=[False, False])
        .head(n)
    )

# --- Helper: parse Flask form into prefs dict ---
def parse_form_to_prefs(form) -> dict:
    genres      = form.getlist('genres')
    size_bucket = form['bucket']
    references  = []

    for i in (1, 2, 3):
        val = form.get(f'ref{i}', 'none')
        if val != 'none':
            references.append(json.loads(val))

    return {
        'genres':      genres,
        'size_bucket': size_bucket,
        'references':  references
    }
