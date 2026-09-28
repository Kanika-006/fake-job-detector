import sys
import html
from pathlib import Path

import streamlit as st
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.predict import model, vectorizer, predict_job,detect_suspicious_signals


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fraudsense · Fake Job Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# ICON SET
# Hand-drawn inline SVGs (stroke-based, currentColor) so the
# whole UI stays self-contained — no external icon CDN.
# ============================================================

def icon(name: str, size: int = 20, stroke: float = 1.8) -> str:
    paths = {
        "shield": '<path d="M12 2 4 5v6c0 5 3.4 8.5 8 10 4.6-1.5 8-5 8-10V5l-8-3Z"/>',
        "shield-check": (
            '<path d="M12 2 4 5v6c0 5 3.4 8.5 8 10 4.6-1.5 8-5 8-10V5l-8-3Z"/>'
            '<path d="m9 12 2 2 4-4"/>'
        ),
        "shield-alert": (
            '<path d="M12 2 4 5v6c0 5 3.4 8.5 8 10 4.6-1.5 8-5 8-10V5l-8-3Z"/>'
            '<path d="M12 8v4"/><path d="M12 15.5h.01"/>'
        ),
        "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
        "file-search": (
            '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h9"/>'
            '<path d="M14 2v5a1 1 0 0 0 1 1h5"/>'
            '<circle cx="12.5" cy="15.5" r="2.5"/><path d="m16 19-1.6-1.6"/>'
        ),
        "gauge": (
            '<path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"/>'
            '<path d="M12 3v2"/><path d="m6.3 6.3 1.4 1.4"/>'
            '<path d="M3 12h2"/><path d="m17.7 7.7 1.4-1.4"/>'
            '<path d="M19 12h2"/><path d="m13.6 13.6 2.8 2.8"/>'
        ),
        "arrow-down-left": '<path d="M17 7 7 17"/><path d="M17 17H7V7"/>',
        "arrow-up-right": '<path d="M7 17 17 7"/><path d="M7 7h10v10"/>',
        "scale": (
            '<path d="M12 3v18"/><path d="M5 8h14"/>'
            '<path d="M5 8 2 15a3 3 0 0 0 6 0L5 8Z"/>'
            '<path d="M19 8l-3 7a3 3 0 0 0 6 0l-3-7Z"/>'
        ),
        "info": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-5"/><path d="M12 8h.01"/>',
        "brain": (
            '<path d="M9 3a3 3 0 0 0-3 3v1a3 3 0 0 0-2 5.5A3 3 0 0 0 6 18a3 3 0 0 0 3 3'
            'c1 0 2-.4 2.6-1.1"/>'
            '<path d="M15 3a3 3 0 0 1 3 3v1a3 3 0 0 1 2 5.5A3 3 0 0 1 18 18a3 3 0 0 1-3 3'
            'c-1 0-2-.4-2.6-1.1"/>'
            '<path d="M9 3v18"/><path d="M15 3v18"/>'
        ),
    }
    body = paths.get(name, "")
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    )


# ============================================================
# GLOBAL CSS
# ============================================================

st.html("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --ink: #14192b;
        --ink-soft: #4b5468;
        --paper: #faf9f6;
        --line: #e4e1d8;
        --verified: #0f7a52;
        --verified-soft: #e5f3ec;
        --flagged: #b3392f;
        --flagged-soft: #f8e8e6;
        --amber: #c8862a;
    }

    .stApp {
        background: var(--paper);
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--ink);
    }

    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 3rem;
        max-width: 1080px;
    }

    h1, h2, h3, .heading-font {
        font-family: 'Space Grotesk', sans-serif;
    }

    .mono {
        font-family: 'IBM Plex Mono', monospace;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        padding-top: 0.5rem;
        padding-bottom: 1.4rem;
        margin-bottom: 1.8rem;
        border-bottom: 2px solid var(--ink);
    }

    # .hero-badge {
    #     flex-shrink: 0;
    #     width: 52px;
    #     height: 52px;
    #     border-radius: 12px;
    #     background: var(--ink);
    #     color: var(--paper);
    #     display: flex;
    #     align-items: center;
    #     justify-content: center;
    # }

    .hero-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0;
        letter-spacing: -0.01em;
    }

    .hero-title svg {
        color: var(--ink);
        flex-shrink: 0;
    }

    .hero-subtitle {
        color: var(--ink-soft);
        font-size: 0.98rem;
        margin: 0.15rem 0 0 0;
    }


    /* =====================================================
       SECTION LABEL (icon + small caps-free heading)
       ===================================================== */

    .section-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--ink);
        margin: 1.6rem 0 0.7rem 0;
    }

    .section-label svg { color: var(--ink-soft); }

    .section-hint {
        color: var(--ink-soft);
        font-size: 0.88rem;
        margin: -0.3rem 0 0.9rem 1.75rem;
    }


    /* =====================================================
       VERDICT CARD
       ===================================================== */

    .verdict-card {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 1.5px solid var(--line);
        background: white;
    }

    .verdict-card.genuine { border-left: 5px solid var(--verified); }
    .verdict-card.fraud { border-left: 5px solid var(--flagged); }

    .verdict-icon {
        flex-shrink: 0;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .verdict-icon.genuine { background: var(--verified-soft); color: var(--verified); }
    .verdict-icon.fraud { background: var(--flagged-soft); color: var(--flagged); }

    .verdict-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .verdict-title.genuine { color: var(--verified); }
    .verdict-title.fraud { color: var(--flagged); }

    .verdict-description {
        color: var(--ink-soft);
        font-size: 0.92rem;
    }


    /* =====================================================
       GAUGE CARD
       ===================================================== */

    .gauge-card {
        background: white;
        border-radius: 10px;
        padding: 1.3rem;
        border: 1.5px solid var(--line);
        text-align: center;
        min-height: 210px;
    }

    .gauge-label {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        color: var(--ink-soft);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    .gauge-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.7rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }


    /* =====================================================
       EXPLANATION CARD
       ===================================================== */

    .explanation-card {
        background: white;
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        border: 1.5px solid var(--line);
        min-height: 210px;
    }

    .explanation-title {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.02rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.7rem;
    }

    .explanation-text {
        color: var(--ink-soft);
        line-height: 1.55;
        font-size: 0.92rem;
    }

    .explanation-list {
        color: var(--ink);
        line-height: 1.7;
        font-size: 0.9rem;
        padding-left: 0;
        list-style: none;
        margin: 0.6rem 0;
    }

    .explanation-list li {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .dot.genuine { background: var(--verified); }
    .dot.fraud { background: var(--flagged); }


    /* =====================================================
       SIGNAL CARDS
       ===================================================== */

    .signal-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1.5px solid var(--line);
    }

    .signal-title {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.98rem;
        color: var(--ink);
        margin-bottom: 0.8rem;
    }

    .signal-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid var(--line);
        font-size: 0.9rem;
    }

    .signal-row:last-child { border-bottom: none; }

    .signal-feature {
        color: var(--ink);
        font-weight: 500;
    }

    .signal-value {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .signal-value.fraud { color: var(--flagged); }
    .signal-value.genuine { color: var(--verified); }


    /* =====================================================
       INFO CARD
       ===================================================== */

    .info-box {
        display: flex;
        gap: 0.7rem;
        background: #f2f0ea;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.95rem 1.1rem;
        color: var(--ink-soft);
        margin-top: 1rem;
        line-height: 1.55;
        font-size: 0.88rem;
    }

    .info-box svg {
        flex-shrink: 0;
        margin-top: 0.15rem;
        color: var(--amber);
    }

    .info-box b { color: var(--ink); }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        color: #9b9788;
        font-size: 0.82rem;
        margin-top: 2.6rem;
        padding-top: 1.1rem;
        border-top: 1px solid var(--line);
    }


    /* =====================================================
       ANALYZE BUTTON
       ===================================================== */

    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        background: var(--ink) !important;
        color: var(--paper) !important;
        border: none !important;
    }

    .stButton > button:hover {
        background: #2a3350 !important;
    }

    .stTextArea textarea {
        border-radius: 8px !important;
        border-color: var(--line) !important;
        font-size: 0.95rem !important;
    }

</style>
""")


# ============================================================
# HERO
# ============================================================

st.html(f"""
<div class="hero">
    <div>
        <p class="hero-title">{icon("shield-check", 32, 2.2)} Fraudsense</p>
        <p class="hero-subtitle">
            Explainable NLP screening for suspicious job postings —
            paste a listing, see the verdict and the evidence behind it.
        </p>
    </div>
</div>
""")


# ============================================================
# INPUT SECTION
# ============================================================

st.html(f"""
<div class="section-label">{icon("file-search")} Paste a job posting</div>
<div class="section-hint">
    Include the title, company details, description and requirements —
    more text gives the model more evidence to work with.
</div>
""")

job_text = st.text_area(
    "Job posting",
    height=260,
    placeholder=(
        "Paste the job title, company information, description, "
        "requirements, benefits, or the complete job posting here..."
    ),
    label_visibility="collapsed"
)

analyze_button = st.button("Analyze posting", type="primary")


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not job_text.strip():

        st.warning("Please paste a job posting before analyzing.")

    else:

        # ====================================================
        # PREDICTION
        # ====================================================

        result, decision_score = predict_job(job_text)
        suspicious_signals = detect_suspicious_signals(job_text)
        if suspicious_signals:
            st.markdown("### Suspicious signals detected")

            for signal_name, explanation in suspicious_signals:
                st.markdown(
                f"**⚠ {signal_name}**  \n"
                f"{explanation}"
                )
        else:
            st.markdown("### Suspicious signals")
            st.caption("No predefined suspicious signals detected.")

        st.caption(
            "These are heuristic warning signals, not proof that a job posting is fraudulent."
        )
            


        # ====================================================
        # LOCAL EXPLANATION
        # ====================================================

        x = vectorizer.transform([job_text])

        feature_indices = x.indices
        feature_values = x.data

        feature_names = vectorizer.get_feature_names_out()

        contributions = (
            feature_values *
            model.coef_[0, feature_indices]
        )

        explanation_df = pd.DataFrame({
            "feature": feature_names[feature_indices],
            "contribution": contributions
        })

        fraud_signals = (
            explanation_df
            .sort_values("contribution", ascending=False)
            .head(5)
        )

        genuine_signals = (
            explanation_df
            .sort_values("contribution", ascending=True)
            .head(5)
        )


        # ====================================================
        # VERDICT CARD
        # ====================================================

        if result == "Likely Genuine":

            st.html(f"""
            <div class="verdict-card genuine">
                <div class="verdict-icon genuine">{icon("shield-check", 24, 2)}</div>
                <div>
                    <div class="verdict-title genuine">Likely Genuine</div>
                    <div class="verdict-description">
                        The model's decision score falls on the genuine
                        side of its decision boundary.
                    </div>
                </div>
            </div>
            """)

        else:

            st.html(f"""
            <div class="verdict-card fraud">
                <div class="verdict-icon fraud">{icon("shield-alert", 24, 2)}</div>
                <div>
                    <div class="verdict-title fraud">Potentially Fraudulent</div>
                    <div class="verdict-description">
                        The model's decision score falls on the fraudulent
                        side of its decision boundary.
                    </div>
                </div>
            </div>
            """)


        # ====================================================
        # SCORE GAUGE + EXPLANATION
        # ====================================================

        score_col, explanation_col = st.columns([1, 2])


        with score_col:

            gauge_min, gauge_max = -3.0, 3.0
            clamped = max(gauge_min, min(gauge_max, decision_score))
            norm = (clamped - gauge_min) / (gauge_max - gauge_min)
            needle_rotation = -90 + 180 * norm

            side_text = "Genuine side" if decision_score < 0 else "Fraudulent side"
            side_color = "var(--verified)" if decision_score < 0 else "var(--flagged)"

            st.html(f"""
            <div class="gauge-card">

                <div class="gauge-label">{icon("gauge", 15, 2)} DECISION SCORE</div>

                <svg width="100%" height="92" viewBox="0 0 200 110" style="margin-top: 0.4rem;">
                    <defs>
                        <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stop-color="#0f7a52"/>
                            <stop offset="50%" stop-color="#c8862a"/>
                            <stop offset="100%" stop-color="#b3392f"/>
                        </linearGradient>
                    </defs>
                    <path d="M 20 100 A 80 80 0 0 1 180 100"
                          fill="none" stroke="url(#gaugeGrad)" stroke-width="14"
                          stroke-linecap="round"/>
                    <line x1="100" y1="100" x2="100" y2="34"
                          stroke="#14192b" stroke-width="3" stroke-linecap="round"
                          transform="rotate({needle_rotation} 100 100)"/>
                    <circle cx="100" cy="100" r="6" fill="#14192b"/>
                </svg>

                <div class="gauge-score mono" style="color:{side_color};">
                    {decision_score:.3f}
                </div>
                <div class="section-hint" style="margin: 0.2rem 0 0 0; color: var(--ink-soft);">
                    {side_text} of boundary
                </div>

            </div>
            """)


        with explanation_col:

            side_word = "genuine" if decision_score < 0 else "fraudulent"

            st.html(f"""
            <div class="explanation-card">

                <div class="explanation-title">{icon("scale", 18)} What does this score mean?</div>

                <div class="explanation-text">
                    The model uses <b>0</b> as its decision boundary.

                    <ul class="explanation-list">
                        <li><span class="dot genuine"></span> <b>Negative score</b> — genuine side</li>
                        <li><span class="dot fraud"></span> <b>Positive score</b> — fraudulent side</li>
                        <li><span class="dot" style="background:var(--amber);"></span> <b>Near 0</b> — close to the boundary, less confident</li>
                    </ul>

                    This posting falls on the <b>{side_word}</b> side of the boundary.
                </div>

            </div>
            """)


        # ====================================================
        # SCORE WARNING
        # ====================================================

        st.html(f"""
        <div class="info-box">
            {icon("info", 18)}
            <div>
                <b>This score is not a percentage or probability.</b>
                A score of -0.866, for example, does not mean the posting
                is 86.6% genuine — it's a raw distance from the model's
                decision boundary.
            </div>
        </div>
        """)


        # ====================================================
        # EXPLAINABILITY HEADER
        # ====================================================

        st.html(f"""
        <div class="section-label">{icon("brain")} Why did the model decide this?</div>
        <div class="section-hint">
            The text features that contributed most strongly to the
            model's decision for this specific posting.
        </div>
        """)


        genuine_col, fraud_col = st.columns(2)


        with genuine_col:

            genuine_rows = ""

            for _, row in genuine_signals.iterrows():

                feature = html.escape(str(row["feature"]))
                contribution = float(row["contribution"])

                genuine_rows += f"""
                <div class="signal-row">
                    <span class="signal-feature">{feature}</span>
                    <span class="signal-value genuine">{contribution:.3f}</span>
                </div>
                """

            st.html(f"""
            <div class="signal-card">
                <div class="signal-title">{icon("arrow-down-left", 17, 2)} Toward genuine</div>
                {genuine_rows}
            </div>
            """)


        with fraud_col:

            fraud_rows = ""

            for _, row in fraud_signals.iterrows():

                feature = html.escape(str(row["feature"]))
                contribution = float(row["contribution"])

                fraud_rows += f"""
                <div class="signal-row">
                    <span class="signal-feature">{feature}</span>
                    <span class="signal-value fraud">+{contribution:.3f}</span>
                </div>
                """

            st.html(f"""
            <div class="signal-card">
                <div class="signal-title">{icon("arrow-up-right", 17, 2)} Toward fraud</div>
                {fraud_rows}
            </div>
            """)


        # ====================================================
        # EXPLANATION NOTE
        # ====================================================

        st.html(f"""
        <div class="info-box">
            {icon("info", 18)}
            <div>
                <b>How to read these signals:</b> positive contributions push
                the model toward the <b>fraudulent</b> class, negative values
                push it toward <b>genuine</b>. These are patterns learned from
                training data, not proof that a posting is fraudulent or genuine.
            </div>
        </div>
        """)


# ============================================================
# FOOTER
# ============================================================

st.html(f"""
<div class="footer">
    {icon("search", 14, 2)}
    <span>Fraudsense · TF-IDF + Linear SVM · Explainable NLP</span>
</div>
""")