from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import json

from functions_module import (
    load_data, preprocess,
    build_user_profile, score_candidates,
    select_top_n, SIZE_BUCKETS, WEIGHTS
)

app = Flask(__name__)
app.secret_key = 'replace-this-with-a-secure-random-string'

# --- Startup: load & preprocess once ---
full_df, niche_df = load_data('books.xlsx', 'over 4 and under 200.xlsx')
full_df  = preprocess(full_df)
niche_df = preprocess(niche_df)

all_genres = sorted({g for sub in full_df['genre_list'] for g in sub})

def sample_pair(prefs, previous_choice=None):
    """
    Return two books (as list of dicts) filtered by prefs,
    carrying forward previous_choice into the next round.
    """
    # 1) filter full_df by genres & bucket
    mask_g  = full_df['genre_list'].apply(lambda gl: bool(set(gl) & set(prefs['genres'])))
    mask_b  = full_df['size_bucket'] == prefs['size_bucket']
    candidates = full_df[mask_g & mask_b].copy()

    # 2) sort & take top 50
    if 'ratings_count' in candidates.columns:
        candidates.sort_values('ratings_count', ascending=False, inplace=True)
    top50 = candidates.head(50)

    # 3a) Round 1: pick any two, preserving orig_index
    if previous_choice is None:
        pair_df = (
            top50
            .sample(2)
            .reset_index()               # bring the DataFrame index into a column named 'index'
            .rename(columns={'index':'orig_index'})
        )

    # 3b) Rounds 2+ : keep previous + one new
    else:
        prev_idx = previous_choice['orig_index']
        prev_row = full_df.loc[prev_idx:prev_idx]  # slice to keep as DataFrame
        prev_df  = (
            prev_row
            .reset_index()               # bring index into column 'index'
            .rename(columns={'index':'orig_index'})
        )

        pool_df = top50[top50.index != prev_idx]
        new_df  = (
            pool_df
            .sample(1)
            .reset_index()
            .rename(columns={'index':'orig_index'})
        )

        pair_df = pd.concat([prev_df, new_df], ignore_index=True)

    # 4) return as list of dicts
    return pair_df.to_dict(orient='records')

@app.route('/', methods=['GET','POST'])
def preferences():
    if request.method == 'POST':
        session['prefs'] = {
            'genres':      request.form.getlist('genres'),
            'size_bucket': request.form['bucket'],
            'references':  []
        }
        return redirect(url_for('reference_round', round_num=1))

    # human-readable buckets for dropdown
    bucket_options = [
        {
          'key': k,
          'lo': int(lo),
          'hi_display': 'and above' if np.isinf(hi) else int(hi)
        }
        for k,(lo,hi) in SIZE_BUCKETS.items()
    ]
    return render_template(
        'preferences.html',
        genres=all_genres,
        buckets=bucket_options
    )

@app.route('/reference/<int:round_num>', methods=['GET','POST'])
def reference_round(round_num):
    prefs = session.get('prefs')
    if not prefs or round_num not in (1,2,3):
        return redirect(url_for('preferences'))

    if request.method == 'POST':
        choice = request.form.get('choice')
        if choice and choice != 'none':
            ref = json.loads(choice)
            prefs['references'].append(ref)
            session['prefs'] = prefs

        if round_num < 3:
            return redirect(url_for('reference_round', round_num=round_num+1))
        else:
            return redirect(url_for('recommend'))

    prev = prefs['references'][-1] if prefs['references'] and round_num>1 else None
    pair = sample_pair(prefs, previous_choice=prev)
    return render_template(
        'reference.html',
        round_num=round_num,
        pair=pair
    )

@app.route('/recommend')
def recommend():
    prefs = session.get('prefs')
    if not prefs:
        return redirect(url_for('preferences'))

    # Build profile & score on niche_df
    profile = build_user_profile(prefs, full_df)
    scored  = score_candidates(niche_df, profile, WEIGHTS)
    top10   = select_top_n(scored, n=10)

    return render_template(
        'results.html',
        prefs=prefs,
        weights=WEIGHTS,
        recommendations=top10.to_dict(orient='records')
    )

if __name__ == '__main__':
    app.run(debug=True)
