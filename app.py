# ============================================================
# app.py — Movie Recommender (Streamlit Frontend)
# Connects directly to: hybrid.py, cbf_model.py, cf_model.py, data_loader.py
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import random
import pandas as pd

# ── IMPORT YOUR FRIEND'S BACKEND DIRECTLY ────────────────────
from data_loader import movies_df
from cbf_model  import find_similar_movies
from hybrid     import hybrid_recommend, cold_start_recommend

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL CSS (Y2K RETRO STYLE) ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace !important;
    background-color: #e8d8f0 !important;
    color: #1a1a2e !important;
}
.stApp { background: #e8d8f0 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem !important; }

.win-card {
    background: #fdf6ec;
    border: 3px solid #1a1a2e;
    box-shadow: 5px 5px 0 #1a1a2e;
    margin-bottom: 14px;
}
.win-bar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 5px 10px;
    border-bottom: 2px solid #1a1a2e;
    font-family: 'VT323', monospace;
    font-size: 15px; letter-spacing: 2px;
}
.dot { width:12px; height:12px; border-radius:50%;
       border:2px solid #1a1a2e; display:inline-block; margin-right:3px; }
.dot-r{background:#f08080} .dot-y{background:#f0d060} .dot-g{background:#80d0a0}

.top-bar { background:#8090d0; border:3px solid #1a1a2e; box-shadow:5px 5px 0 #1a1a2e; margin-bottom:14px; }
.url-row { background:#fdf6ec; padding:6px 12px; display:flex; align-items:center;
           gap:8px; border-top:2px solid #1a1a2e; font-family:'VT323',monospace; font-size:14px; }
.url-box { flex:1; background:white; border:2px solid #1a1a2e;
           padding:2px 8px; font-family:'VT323',monospace; font-size:14px; }
.hearts-row { text-align:center; font-family:'VT323',monospace; font-size:12px;
              color:#e8a0b0; letter-spacing:4px; padding:3px;
              background:#fdf6ec; border-top:2px solid #1a1a2e; }

.sec-card { border:3px solid #1a1a2e; box-shadow:4px 4px 0 #1a1a2e;
            padding:12px; margin-bottom:8px; font-family:'VT323',monospace; }
.sec-label { font-size:11px; color:#666; letter-spacing:1px; margin-bottom:4px; }
.sec-title { font-size:20px; letter-spacing:1px; }
.sec-sub   { font-size:11px; color:#666; margin-top:3px; }

.movie-card { background:#fdf6ec; border:3px solid #1a1a2e;
              box-shadow:4px 4px 0 #1a1a2e; margin-bottom:10px; }
.movie-poster { width:100%; height:110px; display:flex; align-items:center;
                justify-content:center; flex-direction:column;
                border-bottom:2px solid #1a1a2e; font-family:'VT323',monospace;
                font-size:12px; color:#666; gap:4px; }
.movie-body  { padding:10px; }
.movie-title { font-family:'VT323',monospace; font-size:19px; letter-spacing:1px; }
.movie-meta  { font-size:10px; color:#666; margin:3px 0; }
.movie-stars { color:#d4a020; font-size:14px; }
.movie-review{ font-size:10px; line-height:1.5; color:#444;
               margin-top:6px; border-top:1px dashed #bbb; padding-top:6px; }

.swipe-card  { background:#fdf6ec; border:3px solid #1a1a2e;
               box-shadow:6px 6px 0 #1a1a2e; margin:10px auto; max-width:340px; }
.swipe-poster{ height:150px; display:flex; align-items:center; justify-content:center;
               flex-direction:column; border-bottom:3px solid #1a1a2e;
               font-family:'VT323',monospace; font-size:13px; color:#666; }
.swipe-body  { padding:12px; }
.swipe-title { font-family:'VT323',monospace; font-size:26px; }
.swipe-meta  { font-size:10px; color:#666; margin:4px 0; }
.swipe-stars { color:#d4a020; font-size:16px; }
.swipe-review{ font-size:10px; line-height:1.5; color:#444;
               border-top:1px dashed #bbb; padding-top:6px; margin-top:6px; }

div.stButton > button {
    font-family:'VT323',monospace !important; font-size:17px !important;
    letter-spacing:2px !important; border:3px solid #1a1a2e !important;
    border-radius:0 !important; box-shadow:4px 4px 0 #1a1a2e !important;
    padding:6px 18px !important; background:#fdf6ec !important;
    color:#1a1a2e !important; transition:all .1s !important;
}
div.stButton > button:hover {
    transform:translate(3px,3px) !important;
    box-shadow:none !important;
    background:#1a1a2e !important; color:#fdf6ec !important;
}
.stTabs [data-baseweb="tab"] {
    font-family:'VT323',monospace !important;
    font-size:16px !important; letter-spacing:1px !important; border-radius:0 !important;
}
.stTabs [aria-selected="true"] { background:#1a1a2e !important; color:#fdf6ec !important; }
.stTextInput input {
    font-family:'VT323',monospace !important; font-size:16px !important;
    border:2px solid #1a1a2e !important; border-radius:0 !important;
    background:#fdf6ec !important;
}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────
POSTER_COLORS = ["#c0c8f0","#f5c8d4","#b8e8d8","#f8e8a0","#d8c8f0"]

def stars(n):
    try:
        n = float(n)
        f = round(n / 2)       # convert 10-point to 5-star
        f = max(0, min(5, f))
        return "★" * f + "☆" * (5 - f)
    except:
        return "☆☆☆☆☆"

def poster_html(idx, height=110):
    bg = POSTER_COLORS[idx % len(POSTER_COLORS)]
    return f"""<div class="movie-poster" style="background:{bg};height:{height}px">
        <span style="font-size:26px">🎬</span>
        <span>ADD IMAGE HERE</span></div>"""

def win_bar(title, color="#c0c8f0"):
    return f"""<div class="win-bar" style="background:{color}">
        <div><span class="dot dot-r"></span><span class="dot dot-y"></span><span class="dot dot-g"></span></div>
        <span style="font-family:'VT323',monospace;letter-spacing:2px">{title}</span>
        <span>✕</span></div>"""

def add_to_watchlist(movie):
    if not any(m.get("title") == movie.get("title") for m in st.session_state.watchlist):
        st.session_state.watchlist.append(movie)
        st.success(f"♥ Saved: {movie.get('title','')}")
    else:
        st.info(f"Already saved: {movie.get('title','')}")

# ── BACKEND DATA HELPERS ──────────────────────────────────────
@st.cache_data
def get_hot_movies(n=10):
    """Top rated movies from the real dataset."""
    top = movies_df.sort_values(
        by=["vote_average","vote_count"], ascending=[False,False]
    ).head(n)
    return top.to_dict(orient="records")

@st.cache_data
def get_new_movies(n=10):
    """Most recent movies from the real dataset."""
    recent = movies_df.sort_values("year", ascending=False).head(n)
    return recent.to_dict(orient="records")

@st.cache_data
def get_by_genre(genre, n=10):
    """Filter real dataset by genre."""
    filtered = movies_df[
        movies_df["genres"].str.contains(genre, case=False, na=False)
    ].sort_values("vote_average", ascending=False).head(n)
    return filtered.to_dict(orient="records")

def get_random_movie():
    return movies_df.sample(1).iloc[0].to_dict()

def movie_card_html(m, idx):
    title  = m.get("title", "Unknown")
    year   = m.get("year", "N/A")
    source = m.get("source", "")
    genres = m.get("genres", "")
    rating = m.get("vote_average", m.get("popularity", m.get("final_score", "—")))
    bg     = POSTER_COLORS[idx % len(POSTER_COLORS)]
    return f"""
    <div class="movie-card">
        <div class="movie-poster" style="background:{bg};height:110px">
            <span style="font-size:26px">🎬</span>
            <span>ADD IMAGE HERE</span>
        </div>
        <div class="movie-body">
            <div class="movie-title">{title}</div>
            <div class="movie-meta">{year} · {source} · {genres[:40]}</div>
            <div class="movie-stars">{stars(rating)} {rating}</div>
        </div>
    </div>"""


# ── SESSION STATE ─────────────────────────────────────────────
for key, val in {
    "watchlist": [], "page": "Home",
    "swipe_queue": [], "swipe_idx": 0,
    "selected_genres": [], "random_movie": None,
    "search_results": None, "hybrid_results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── TOP BAR ───────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="win-bar" style="background:#8090d0">
        <div><span class="dot dot-r"></span><span class="dot dot-y"></span><span class="dot dot-g"></span></div>
        <span style="font-family:'VT323',monospace;font-size:18px;letter-spacing:3px;color:#fdf6ec">MOVIE RECOMMENDER</span>
        <span style="font-family:'VT323',monospace;font-size:13px;color:#fdf6ec;
              background:#e8a0b0;border:2px solid #fdf6ec;padding:1px 8px">★ LOOKING FOR PICKS !!!</span>
    </div>
    <div class="url-row">
        <span>↺</span>
        <div class="url-box">https://www.movie-recommender.com</div>
        <span style="background:#b8e8d8;border:2px solid #1a1a2e;padding:2px 6px">✉</span>
    </div>
    <div class="hearts-row">♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥ ♥</div>
</div>""", unsafe_allow_html=True)

# ── NAV ───────────────────────────────────────────────────────
pages = ["Home","Discover","Browse","Search","My List"]
cols  = st.columns(len(pages))
for col, pg in zip(cols, pages):
    with col:
        lbl = f"▶ {pg.upper()}" if st.session_state.page == pg else pg.upper()
        if st.button(lbl, key=f"nav_{pg}", use_container_width=True):
            st.session_state.page = pg
            st.rerun()

st.markdown("---")


# ════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════
if st.session_state.page == "Home":
    st.markdown(f"""<div class="win-card">
        {win_bar("HOME.exe","#f5c8d4")}
        <div style="padding:10px 14px;border-bottom:2px solid #1a1a2e;background:#fdf6ec">
            <div style="font-family:'VT323',monospace;font-size:13px;color:#666;
                        letter-spacing:1px;margin-bottom:6px">&gt; SEARCH MOVIES</div>
            <div style="display:flex;gap:8px;align-items:center">
                <div style="flex:1;border:2px solid #1a1a2e;background:white;
                            padding:5px 10px;font-family:'VT323',monospace;font-size:15px;
                            color:#aaa">search a title, genre or language...</div>
                <div style="background:#1a1a2e;color:#fdf6ec;padding:6px 14px;
                            font-family:'VT323',monospace;font-size:15px;
                            border:2px solid #1a1a2e;cursor:pointer">SEARCH</div>
            </div>
        </div>
        <div style="padding:12px 16px;font-family:'VT323',monospace;font-size:24px;
                    border-bottom:2px solid #1a1a2e">
            what are we watching tonight? ✦
        </div></div>""", unsafe_allow_html=True)

    # Functional search bar
    home_query = st.text_input("", placeholder="search a title, genre or language...",
                               label_visibility="collapsed", key="home_search")
    if home_query.strip():
        if st.button("🔍 SEARCH", use_container_width=True, key="home_search_btn"):
            st.session_state.page = "Search"
            st.session_state["prefill_search"] = home_query.strip()
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="sec-card" style="background:#c0c8f0">
            <div class="sec-label">🎬 SECTION 01</div>
            <div class="sec-title">NEW DROPS</div>
            <div class="sec-sub">latest from the dataset</div></div>""", unsafe_allow_html=True)
        if st.button("OPEN →", key="h_new", use_container_width=True):
            st.session_state.page = "Browse"; st.rerun()

        st.markdown("""<div class="sec-card" style="background:#b8e8d8">
            <div class="sec-label">🃏 SECTION 03</div>
            <div class="sec-title">CHOOSE YOUR NEXT MOVIE</div>
            <div class="sec-sub">swipe to pick</div></div>""", unsafe_allow_html=True)
        if st.button("OPEN →", key="h_disc", use_container_width=True):
            st.session_state.page = "Discover"; st.rerun()

    with c2:
        st.markdown("""<div class="sec-card" style="background:#f5c8d4">
            <div class="sec-label">🔥 SECTION 02</div>
            <div class="sec-title">HOT NOW</div>
            <div class="sec-sub">top rated picks</div></div>""", unsafe_allow_html=True)
        if st.button("OPEN →", key="h_hot", use_container_width=True):
            st.session_state.page = "Browse"; st.rerun()

        st.markdown("""<div class="sec-card" style="background:#f8e8a0">
            <div class="sec-label">🎲 SECTION 04</div>
            <div class="sec-title">RANDOM PICK</div>
            <div class="sec-sub">from real dataset</div></div>""", unsafe_allow_html=True)
        if st.button("SPIN 🎲", key="h_spin", use_container_width=True):
            st.session_state.random_movie = get_random_movie(); st.rerun()

    st.markdown("""<div class="sec-card" style="background:#d8c8f0">
        <div class="sec-label">📺 SECTION 05</div>
        <div class="sec-title">ANIME · K-DRAMA · SHOWS · SITCOMS · BOLLYWOOD</div>
        <div class="sec-sub">browse by category</div></div>""", unsafe_allow_html=True)
    if st.button("BROWSE ALL →", key="h_browse", use_container_width=True):
        st.session_state.page = "Browse"; st.rerun()

    if st.session_state.random_movie:
        m = st.session_state.random_movie
        bg = POSTER_COLORS[0]
        st.markdown(f"""<div class="win-card" style="margin-top:14px">
            {win_bar("RANDOM PICK.exe","#f8e8a0")}
            <div class="movie-poster" style="background:{bg};height:130px">
                <span style="font-size:30px">🎬</span>
                <span>ADD IMAGE HERE</span>
            </div>
            <div class="movie-body">
                <div class="movie-title">{m.get('title','')}</div>
                <div class="movie-meta">{m.get('year','N/A')} · {m.get('source','')} · {str(m.get('genres',''))[:50]}</div>
                <div class="movie-stars">{stars(m.get('vote_average',0))} {m.get('vote_average','—')}/10</div>
            </div></div>""", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            if st.button("↻ SPIN AGAIN", use_container_width=True):
                st.session_state.random_movie = get_random_movie(); st.rerun()
        with cb:
            if st.button("♥ SAVE", use_container_width=True):
                add_to_watchlist(st.session_state.random_movie)


# ════════════════════════════════════════════════════════════
# DISCOVER — genre-based swipe queue using real data
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "Discover":
    st.markdown(f"""<div class="win-card">
        {win_bar("CHOOSE YOUR NEXT MOVIE","#b8e8d8")}
        <div style="padding:8px 12px;font-size:11px;color:#666;border-bottom:1px dashed #bbb">
            &gt; pick genres → generate → swipe through real movies
        </div></div>""", unsafe_allow_html=True)

    GENRES = ["Action","Comedy","Drama","Horror","Romance","Thriller",
              "Sci-Fi","Animation","Crime","Mystery","Fantasy","Musical"]

    st.markdown("<div style='font-family:VT323,monospace;font-size:18px;margin-bottom:8px'>SELECT GENRES:</div>",
                unsafe_allow_html=True)
    gcols = st.columns(4)
    for i, g in enumerate(GENRES):
        with gcols[i % 4]:
            sel = g in st.session_state.selected_genres
            if st.button(f"{'✓ ' if sel else ''}{g}", key=f"g_{g}", use_container_width=True):
                if sel: st.session_state.selected_genres.remove(g)
                else:   st.session_state.selected_genres.append(g)
                st.rerun()

    if st.button("▶ GENERATE PICKS", use_container_width=True):
        sel = st.session_state.selected_genres
        if sel:
            # Use real backend: cold_start_recommend per genre
            pool = []
            for g in sel:
                res = cold_start_recommend(g, n=4)
                if isinstance(res, list):
                    pool.extend(res)
        else:
            pool = get_hot_movies(8)
        random.shuffle(pool)
        st.session_state.swipe_queue = pool[:6]
        st.session_state.swipe_idx  = 0
        st.rerun()

    if st.session_state.swipe_queue:
        idx   = st.session_state.swipe_idx
        queue = st.session_state.swipe_queue
        if idx < len(queue):
            m  = queue[idx]
            bg = POSTER_COLORS[idx % len(POSTER_COLORS)]
            title   = m.get("title","Unknown")
            year    = m.get("year","N/A")
            source  = m.get("source","")
            genres  = m.get("genres","")
            rating  = m.get("popularity", m.get("vote_average","—"))
            st.markdown(f"""<div class="win-card">
                {win_bar(f"CARD {idx+1} OF {len(queue)}","#c0c8f0")}
                <div class="swipe-poster" style="background:{bg}">
                    <span style="font-size:34px">🎬</span>
                    <span style="font-family:'VT323',monospace;font-size:12px;color:#666">ADD IMAGE HERE</span>
                </div>
                <div style="background:#d8c8f0;border-bottom:2px solid #1a1a2e;
                            padding:4px 12px;display:flex;justify-content:space-between;
                            font-family:'VT323',monospace;font-size:13px">
                    <span>{str(genres)[:35]}</span><span>{source}</span>
                </div>
                <div class="swipe-body">
                    <div class="swipe-title">{title}</div>
                    <div class="swipe-meta">{year}</div>
                    <div class="swipe-stars">{stars(rating)} {rating}/10</div>
                </div></div>""", unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1:
                if st.button("✕  SKIP", use_container_width=True, key="skip"):
                    st.session_state.swipe_idx += 1; st.rerun()
            with sc2:
                if st.button("♥  SAVE", use_container_width=True, key="save_sw"):
                    add_to_watchlist(m)
                    st.session_state.swipe_idx += 1; st.rerun()
        else:
            st.markdown(f"""<div class="win-card">
                {win_bar("DONE!","#b8e8d8")}
                <div style="text-align:center;padding:40px;font-family:'VT323',monospace;
                            font-size:22px;color:#666">✦ END OF QUEUE ✦<br>
                    <span style="font-size:14px">check your list!</span></div>
                </div>""", unsafe_allow_html=True)
            if st.button("↻ NEW QUEUE", use_container_width=True):
                st.session_state.swipe_queue = []; st.session_state.swipe_idx = 0; st.rerun()


# ════════════════════════════════════════════════════════════
# BROWSE — real data from movies_df
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "Browse":
    st.markdown(f"""<div class="win-card">
        {win_bar("BROWSE.exe","#f8e8a0")}</div>""", unsafe_allow_html=True)

    SECTIONS = {
        "🎬 New":       get_new_movies(10),
        "🔥 Hot Now":   get_hot_movies(10),
        "🎌 Anime":     get_by_genre("Animation", 8),
        "🌸 K-Drama":   get_by_genre("Drama", 8),
        "🎭 Action":    get_by_genre("Action", 8),
        "😂 Comedy":    get_by_genre("Comedy", 8),
        "💀 Horror":    get_by_genre("Horror", 8),
        "🇮🇳 Bollywood": list(movies_df[movies_df["source"]=="Bollywood"]
                              .sort_values("vote_average",ascending=False)
                              .head(8).to_dict(orient="records")),
    }

    tabs = st.tabs(list(SECTIONS.keys()))
    for tab, (section_name, section_movies) in zip(tabs, SECTIONS.items()):
        with tab:
            if not section_movies:
                st.write("No movies found.")
                continue
            cols = st.columns(2)
            for i, m in enumerate(section_movies):
                with cols[i % 2]:
                    st.markdown(movie_card_html(m, i), unsafe_allow_html=True)
                    if st.button("+ MY LIST", key=f"add_{section_name}_{i}", use_container_width=True):
                        add_to_watchlist(m)


# ════════════════════════════════════════════════════════════
# SEARCH — uses your friend's find_similar_movies + hybrid
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "Search":
    st.markdown(f"""<div class="win-card">
        {win_bar("SEARCH.exe","#c0c8f0")}
        <div style="padding:8px 12px;font-size:11px;color:#666;border-bottom:1px dashed #bbb">
            &gt; search a movie title → get AI-powered similar picks
        </div></div>""", unsafe_allow_html=True)

    prefill = st.session_state.pop("prefill_search", "")
    query   = st.text_input("ENTER MOVIE TITLE:", value=prefill,
                            placeholder="e.g. Inception, Dilwale, The Dark Knight")
    user_id = st.text_input("ENTER USER ID (1–943) for personalised picks — leave blank to skip:",
                            placeholder="e.g. 42")

    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("🔍 FIND SIMILAR MOVIES", use_container_width=True):
            if query.strip():
                with st.spinner("Finding similar movies..."):
                    result = find_similar_movies(query.strip(), n=6)
                st.session_state.search_results  = result
                st.session_state.hybrid_results  = None
            else:
                st.warning("Please enter a movie title.")

    with sc2:
        if st.button("✦ HYBRID RECOMMEND (CF + CBF)", use_container_width=True):
            if query.strip() and user_id.strip():
                with st.spinner("Running hybrid model..."):
                    result = hybrid_recommend(user_id.strip(), query.strip(), n=6)
                st.session_state.hybrid_results  = result
                st.session_state.search_results  = None
            elif not query.strip():
                st.warning("Please enter a movie title.")
            else:
                st.warning("Please enter a User ID (1–943) for hybrid recommendations.")

    # ── DISPLAY SIMILAR RESULTS ──────────────────────────────
    if st.session_state.search_results:
        r = st.session_state.search_results
        if "error" in r:
            st.error(r["error"])
        elif "multiple_matches" in r:
            st.warning(r["message"])
            for opt in r["multiple_matches"]:
                st.write(f"  • {opt}")
        else:
            st.markdown(f"""<div class="win-card">
                {win_bar(f"SIMILAR TO: {r['searched_for']}","#f5c8d4")}
            </div>""", unsafe_allow_html=True)
            cols = st.columns(2)
            for i, m in enumerate(r["recommendations"]):
                with cols[i % 2]:
                    bg = POSTER_COLORS[i % len(POSTER_COLORS)]
                    st.markdown(f"""<div class="movie-card">
                        <div class="movie-poster" style="background:{bg};height:100px">
                            <span style="font-size:24px">🎬</span>
                            <span>ADD IMAGE HERE</span>
                        </div>
                        <div class="movie-body">
                            <div class="movie-title">{m['title']}</div>
                            <div class="movie-meta">{m['year']} · {m['source']} · {m['genres'][:35]}</div>
                            <div class="movie-stars">Similarity: {m['similarity']}</div>
                        </div></div>""", unsafe_allow_html=True)
                    if st.button("+ MY LIST", key=f"sr_{i}", use_container_width=True):
                        add_to_watchlist(m)

    # ── DISPLAY HYBRID RESULTS ───────────────────────────────
    if st.session_state.hybrid_results:
        r = st.session_state.hybrid_results
        if "error" in r:
            st.error(r["error"])
        else:
            st.markdown(f"""<div class="win-card">
                {win_bar(f"PICKS FOR USER {r['user_id']} · {r['mode'].upper()}","#b8e8d8")}
            </div>""", unsafe_allow_html=True)
            cols = st.columns(2)
            for i, m in enumerate(r["recommendations"]):
                with cols[i % 2]:
                    bg = POSTER_COLORS[i % len(POSTER_COLORS)]
                    st.markdown(f"""<div class="movie-card">
                        <div class="movie-poster" style="background:{bg};height:100px">
                            <span style="font-size:24px">🎬</span>
                            <span>ADD IMAGE HERE</span>
                        </div>
                        <div class="movie-body">
                            <div class="movie-title">{m['title']}</div>
                            <div class="movie-meta">{m['year']} · {m['source']} · {m['genres'][:35]}</div>
                            <div class="movie-stars">
                                Score: {m['final_score']} &nbsp;|&nbsp;
                                CF: {m['cf_score']} &nbsp;|&nbsp; CBF: {m['cbf_score']}
                            </div>
                        </div></div>""", unsafe_allow_html=True)
                    if st.button("+ MY LIST", key=f"hr_{i}", use_container_width=True):
                        add_to_watchlist(m)


# ════════════════════════════════════════════════════════════
# MY LIST
# ════════════════════════════════════════════════════════════
elif st.session_state.page == "My List":
    st.markdown(f"""<div class="win-card">
        {win_bar("MY LIST.exe","#f5c8d4")}</div>""", unsafe_allow_html=True)

    if not st.session_state.watchlist:
        st.markdown("""<div style="text-align:center;padding:60px;
            font-family:'VT323',monospace;font-size:22px;color:#666">
            [ EMPTY ]<br><span style="font-size:14px">save films to see them here</span>
        </div>""", unsafe_allow_html=True)
    else:
        for i, m in enumerate(st.session_state.watchlist):
            bg = POSTER_COLORS[i % len(POSTER_COLORS)]
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"""
                <div style="display:flex;gap:10px;align-items:center;
                            padding:10px;border-bottom:2px solid #1a1a2e">
                    <div style="width:44px;height:56px;background:{bg};
                                border:2px solid #1a1a2e;display:flex;align-items:center;
                                justify-content:center;font-size:9px;
                                font-family:'VT323',monospace;text-align:center;color:#666">
                        ADD<br>IMG</div>
                    <div>
                        <div style="font-family:'VT323',monospace;font-size:18px">{m.get('title','')}</div>
                        <div style="font-size:10px;color:#666">{m.get('year','N/A')} · {m.get('source','')} · {str(m.get('genres',''))[:40]}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"rm_{i}", use_container_width=True):
                    st.session_state.watchlist.pop(i); st.rerun()


# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-family:'VT323',monospace;font-size:13px;color:#888;
            margin-top:20px;padding:8px;border:3px solid #1a1a2e;
            background:#fdf6ec;box-shadow:3px 3px 0 #1a1a2e;letter-spacing:1px">
    ♥ MOVIE RECOMMENDER · EST. 2025 · LOADING... ♥
</div>""", unsafe_allow_html=True)
