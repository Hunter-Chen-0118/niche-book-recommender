# niche-book-recommender
0. Introduction
I built a Python‑based recommendation engine whose goal is to surface high‑quality, low‑visibility books that typically fall outside mainstream algorithms. By filtering for titles with an average rating above 4.0 but fewer than 200 reviews, the system focuses on a curated “niche” subset that often goes overlooked on Goodreads.
This report walks through each component of the program, explaining the logic and why the recommendations align with user preferences.

**recommendation.ipynb is where the program is located, reference that only

1. Data Loading
Files:


books.xlsx is the full catalog of all books (titles, authors, pages, ratings, etc.).


over 4 and under 200.xlsx is a smaller “niche” subset we’ll ultimately score and recommend from.


Function:

 python
CopyEdit
full_df, niche_df = load_data(full_path, niche_path)
 simply reads both spreadsheets into Pandas DataFrames (full_df, niche_df).



2. Preprocessing
Before we can match user tastes, we normalize and bucket key attributes:
Genre parsing

 python
CopyEdit
df['genre_list'] = df['genre'].str.split(',')
 turns "Fantasy,Adventure" into ['Fantasy','Adventure'].


Page counts

 python
CopyEdit
df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')
 ensures pages are numeric.


Size buckets
 We define four buckets by page‐count ranges:

 python
CopyEdit
SIZE_BUCKETS = {
  'short':      (0,   250),
  'medium':     (250, 500),
  'long':       (500, 700),
  'extra_long': (700, ∞)
}
 Each book is assigned a size_bucket based on its page count.



3. Gathering User Preferences
When you run the program, it:
Displays available genres (collected from full_df['genre_list']).


Prompts you to select one or more by index.


Shows size buckets and asks you to choose one.


Three “reference‐book” rounds—pairwise comparisons from top‐rated books in your chosen genres & size bucket:


In each round you pick the one you’ve read (or enter ‘n’ if neither).


We store your chosen rows in prefs['references'].


This builds a small set of exemplar books that reflect your detailed taste beyond just “Fantasy” or “medium length.”

4. Building the User Profile
Once you’ve made your choices, we distill them into numerical targets:
Genre vector


We fit a MultiLabelBinarizer on all possible genres.


Your selected genres become a binary vector 𝐠ₙ where 1s mark chosen genres.


Length target


We scale all page counts to [0,1] via a MinMaxScaler.


We take the midpoint of your bucket (e.g. 250–500 → 375), scale it, and call that 𝐿ₜ.


Reference centroid


For each reference book you picked, we form a vector [genre_bits…, scaled_pages].


We average these to get a single “centroid” vector 𝐜ᴿ.


If you made no reference picks, 𝐜ᴿ is all zeros (so we ignore reference similarity).


Resulting profile:
python
CopyEdit
{
  'mlb':    <binarizer>,
  'scaler': <page‐scaler>,
  'genre_vector':   𝐠ₙ,
  'length_target':  𝐿ₜ,
  'reference_centroid':  𝐜ᴿ,
  'size_bucket':     your_choice
}


5. Scoring Each Candidate
Every book in the niche set is scored by three sub‐scores:
Genre match (weight = 0.50)
 genre_score=gbook  ⋅  gn∑gn \text{genre\_score} = \frac{\mathbf{g}_{\text{book}}\;\cdot\;\mathbf{g}_n}{\sum \mathbf{g}_n}genre_score=∑gn​gbook​⋅gn​​
 → fraction of your chosen genres the book covers.


Length match (weight = 0.25)


If pages ∈ your bucket, length_score = 1.


Else
 length_score=max⁡ ⁣(0,  1−distance outside bucketbucket width) \text{length\_score} = \max\!\Bigl(0,\;1 - \frac{\text{distance outside bucket}}{\text{bucket width}}\Bigr)length_score=max(0,1−bucket widthdistance outside bucket​)
→ penalizes how far a book’s length is from your ideal range.


Reference similarity (weight = 0.25)


Construct the book’s combined vector [gbook,  scaled_pages][\mathbf{g}_{\text{book}},\; \text{scaled\_pages}][gbook​,scaled_pages].


Compute cosine similarity to your reference centroid 𝐜ᴿ:
 ref_score=1−cos⁡(vbook, cR) \text{ref\_score} = 1 - \cos\bigl(\mathbf{v}_{\text{book}},\,\mathbf{c}^R\bigr)ref_score=1−cos(vbook​,cR)
→ how much the candidate “feels like” the books you explicitly picked.


Finally, the overall score is the weighted sum:
python
CopyEdit
score = 0.5*genre_score + 0.25*length_score + 0.25*ref_score


6. Selecting Top N
We then sort all niche books by
score (descending)


tiebreaker: average_rating (descending)


(optionally) further by ratings_count if needed


This ensures that among equally‐scored books, the more highly rated by the broader community comes first.

7. Web Application

I packaged the entire recommendation engine into a simple Flask web app, using three template routes to mirror the console flow (preferences, reference rounds, and results) and Jinja2 templates for the UI. It runs locally on my machine, and I opted not to deploy it publicly since I didn’t want to incur hosting costs.

To run the web:
Clone the repo:
 git clone https://github.com/your-username/your-repo.git && cd your-repo


Create and activate a virtual environment:
 python3 -m venv venv && source venv/bin/activate (on Windows: venv\Scripts\activate)


Install dependencies:
 pip install -r requirements.txt


Set the Flask app variable:
 export FLASK_APP=app.py (on Windows PowerShell: set FLASK_APP=app.py)


Run the server:
 flask run


Open your browser to http://127.0.0.1:5000/

8. Why These Recommendations Make Sense
Strong Genre Alignment
 Books sharing more of your chosen genres get a big boost (50% of the total).


Appropriate Length
 A 350-page novel when you want “medium” scores a perfect 1.0. A 700-page epic is penalized smoothly rather than binary accept/reject—so you still see long books, but ranked lower.


Reference‐Book Echo
 If in round 2 you picked Memoirs of a Geisha, the system learns your affinity for historical literary fiction at ~430 pages—so other novels with a similar “flavor” rise in the list.


Community Validation
 When two titles both match your profile equally well, the one with the higher community average rating appears first.



