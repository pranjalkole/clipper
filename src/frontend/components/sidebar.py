"""
Sidebar component for faceted filtering, agent hyperparameter tuning, and observability.
"""

from typing import Any
import streamlit as st

from src.frontend.client import ApiClient
from src.models.schemas import SearchFilterParams


def render_sidebar(client: ApiClient) -> dict[str, Any]:
    """
    Renders sidebar containing backend status, faceted catalog filters,
    agent reflection controls, and real-time observability telemetry.

    Returns:
        Dictionary containing configured 'filters', 'limit', 'confidence_threshold',
        and 'max_retries'.
    """
    with st.sidebar:
        st.markdown("## ⚙️ System & Controls")

        # ---------------------------------------------------------
        # 1. Gateway Status & Health
        # ---------------------------------------------------------
        health = client.check_health()
        if health and health.status == "healthy":
            st.success("🟢 **FastAPI Gateway**: Connected\n\n"
                       f"• Qdrant: {'✅ Online' if health.qdrant_connected else '❌ Error'}\n\n"
                       f"• Embedder: {'✅ Ready' if health.embedder_loaded else '❌ Error'}")
        elif health and health.status == "degraded":
            st.warning("🟡 **FastAPI Gateway**: Degraded\n\n"
                       f"• Qdrant: {'✅' if health.qdrant_connected else '❌'}\n\n"
                       f"• Embedder: {'✅' if health.embedder_loaded else '❌'}")
        else:
            st.error(f"🔴 **Gateway Offline**\n\nCannot reach `{client.base_url}`. Start backend with:\n`uvicorn src.api.app:app`")

        st.markdown("---")

        # ---------------------------------------------------------
        # 2. Faceted Catalog Filters
        # ---------------------------------------------------------
        st.markdown("### 🏷️ Faceted Filters")

        if st.button("🔄 Reset Filters", use_container_width=True):
            st.session_state["filter_price_range"] = (0.0, 200.0)
            st.session_state["filter_min_rating"] = 1.0
            st.session_state["filter_categories"] = []
            st.session_state["filter_sub_categories"] = []
            st.session_state["filter_genders"] = []
            st.session_state["filter_colors"] = []
            st.session_state["filter_brands"] = []
            st.session_state["filter_in_stock"] = False
            st.rerun()

        price_range = st.slider(
            "Price Range ($)",
            min_value=0.0,
            max_value=200.0,
            value=st.session_state.get("filter_price_range", (0.0, 200.0)),
            step=5.0,
            key="filter_price_range",
        )

        min_rating = st.slider(
            "Minimum Rating",
            min_value=1.0,
            max_value=5.0,
            value=st.session_state.get("filter_min_rating", 1.0),
            step=0.1,
            key="filter_min_rating",
        )

        master_categories = st.multiselect(
            "Master Category",
            options=["Apparel", "Footwear", "Accessories", "Personal Care", "Sporting Goods"],
            default=st.session_state.get("filter_categories", []),
            key="filter_categories",
        )

        sub_categories = st.multiselect(
            "Sub Category",
            options=["Topwear", "Bottomwear", "Shoes", "Watches", "Bags", "Eyewear", "Innerwear", "Fragrance"],
            default=st.session_state.get("filter_sub_categories", []),
            key="filter_sub_categories",
        )

        genders = st.multiselect(
            "Gender",
            options=["Men", "Women", "Unisex", "Boys", "Girls"],
            default=st.session_state.get("filter_genders", []),
            key="filter_genders",
        )

        colors = st.multiselect(
            "Color",
            options=[
                "Black", "White", "Blue", "Navy Blue", "Red", "Grey",
                "Green", "Yellow", "Pink", "Purple", "Brown", "Beige", "Orange",
                "Silver", "Gold", "Charcoal", "Maroon", "Olive", "Khaki", "Multi"
            ],
            default=st.session_state.get("filter_colors", []),
            key="filter_colors",
        )

        brands = st.multiselect(
            "Brand",
            options=[
                "Nike", "Puma", "Adidas", "Levi's", "United Colors of Benetton",
                "Tommy Hilfiger", "Fastrack", "Titan", "Casio", "Reebok", "Fila", "Converse"
            ],
            default=st.session_state.get("filter_brands", []),
            key="filter_brands",
        )

        in_stock_only = st.checkbox(
            "In-Stock Inventory Only",
            value=st.session_state.get("filter_in_stock", False),
            key="filter_in_stock",
        )

        # Assemble SearchFilterParams
        filter_params = SearchFilterParams(
            min_price=price_range[0] if price_range[0] > 0 else None,
            max_price=price_range[1] if price_range[1] < 200 else None,
            min_rating=min_rating if min_rating > 1.0 else None,
            master_category=master_categories if master_categories else None,
            sub_category=sub_categories if sub_categories else None,
            gender=genders if genders else None,
            base_colour=colors if colors else None,
            brand=brands if brands else None,
            in_stock_only=True if in_stock_only else None,
        )

        st.markdown("---")

        # ---------------------------------------------------------
        # 3. Multi-Agent & Search Hyperparameters
        # ---------------------------------------------------------
        st.markdown("### 🧠 Agent Controls")

        limit = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=30,
            value=st.session_state.get("agent_top_k", 10),
            step=1,
            key="agent_top_k",
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.40,
            max_value=0.95,
            value=st.session_state.get("agent_threshold", 0.70),
            step=0.05,
            help="Candidate similarity below this threshold triggers agent self-reflection critique.",
            key="agent_threshold",
        )

        max_retries = st.slider(
            "Max Reflection Loops",
            min_value=0,
            max_value=3,
            value=st.session_state.get("agent_retries", 2),
            step=1,
            help="Maximum query reformulation and filter relaxation attempts.",
            key="agent_retries",
        )

        # ---------------------------------------------------------
        # 4. Observability & Telemetry Panel
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📊 Live Observability")
        metrics = client.get_metrics()
        if metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("P95 Latency", f"{metrics.p95_latency_ms:.0f} ms")
                st.metric("Total Queries", metrics.total_requests)
            with col2:
                st.metric("P50 Latency", f"{metrics.p50_latency_ms:.0f} ms")
                st.metric("Sim. Cost", f"${metrics.simulated_cost_usd:.4f}")
        else:
            st.caption("Observability metrics unavailable.")

        return {
            "filters": filter_params,
            "limit": limit,
            "confidence_threshold": confidence_threshold,
            "max_retries": max_retries,
        }
