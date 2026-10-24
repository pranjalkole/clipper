"""
Main Streamlit application entrypoint for Clipper:
Enterprise Multimodal Semantic Search & Agentic Intelligence Platform.
"""

import streamlit as st

from src.frontend.client import ApiClient
from src.frontend.components.agent_inspector import render_agent_inspector
from src.frontend.components.product_grid import render_product_grid
from src.frontend.components.search_bar import render_search_bar
from src.frontend.components.sidebar import render_sidebar
from src.models.schemas import ProductCandidate, SearchResponse


def init_session_state() -> None:
    """Initializes session state defaults across user interactions."""
    defaults = {
        "search_response": None,
        "search_mode": None,
        "search_query_display": None,
        "cir_ref_bytes": None,
        "cir_ref_name": None,
        "cir_mod_prompt": "",
        "active_query_to_run": None,
        "pending_filter_updates": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def inject_custom_styles() -> None:
    """Injects custom CSS for modern card designs, badges, and clean spacing."""
    st.markdown(
        """
        <style>
        /* Card styling */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            border-radius: 8px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }

        /* Metric badge styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
            font-weight: 700;
        }

        /* Hero banner title */
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #6c757d;
            margin-bottom: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Streamlit application main routine."""
    st.set_page_config(
        page_title="Clipper | Multimodal Semantic Search",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    inject_custom_styles()

    # Apply staged filter updates before sidebar widgets are instantiated
    if st.session_state.get("pending_filter_updates"):
        pending = st.session_state.pop("pending_filter_updates")
        for k, v in pending.items():
            st.session_state[k] = v

    client = ApiClient()

    # Render Sidebar with faceted filters and agent controls
    controls = render_sidebar(client)

    # Hero Banner
    st.markdown('<div class="hero-title">🛍️ Clipper Fashion Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'Enterprise Multimodal Semantic Search powered by <b>Apple MobileCLIP2-S0</b>, '
        '<b>Qdrant Vector DB</b>, <b>LangGraph Multi-Agent Reflection</b>, and <b>Google Gemini 1.5 Flash</b>.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Dual Column Responsive Layout
    col_main, col_inspector = st.columns([13, 7], gap="medium")

    with col_main:
        # Multimodal Search Input Bar
        render_search_bar(
            client=client,
            filters=controls["filters"],
            limit=controls["limit"],
            confidence_threshold=controls["confidence_threshold"],
            max_retries=controls["max_retries"],
        )

        st.markdown("---")

        # Search Results Gallery
        search_res = st.session_state.get("search_response")
        query_disp = st.session_state.get("search_query_display")

        if query_disp:
            st.markdown(f"#### 📦 Results for: *{query_disp}*")
        else:
            st.markdown("#### 📦 Catalog Candidates")

        candidates: list[ProductCandidate] = []
        if search_res:
            candidates = search_res.candidates

        render_product_grid(candidates, num_columns=3, base_url=client.base_url)

    with col_inspector:
        # Agent Reasoning & Self-Reflection Inspector
        render_agent_inspector(st.session_state.get("search_response"))


if __name__ == "__main__":
    main()
