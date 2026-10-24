"""
Agent reasoning, reflection critique, and stylist advice introspection panel.
"""

from typing import Any
import streamlit as st

from src.models.schemas import ComposedSearchResponse, SearchResponse


def render_agent_inspector(response: SearchResponse | ComposedSearchResponse | None) -> None:
    """
    Renders the LangGraph Multi-Agent execution trace and reasoning inspector.
    """
    st.markdown("### 🧠 Agent Reasoning & Reflection")

    if response is None:
        st.info("Execute a search query to inspect the multi-agent reasoning flow and critique loops.")
        return

    # Header Metrics
    col_lat, col_cnt = st.columns(2)
    with col_lat:
        st.metric("Total Latency", f"{response.latency_ms:.1f} ms")
    with col_cnt:
        st.metric("Candidates Returned", response.total_found)

    st.markdown("---")

    # -------------------------------------------------------------
    # 1. Composed Image Retrieval (WISER CIR) Inspection
    # -------------------------------------------------------------
    if isinstance(response, ComposedSearchResponse):
        st.markdown("#### 🎯 WISER Zero-Shot CIR Flow")

        # Step 1: Synthesized Target Caption
        with st.container(border=True):
            st.markdown("**1. Target Description Synthesis (WISER Editor)**")
            st.info(f"*{response.target_caption}*")
            st.caption(f"Input modification prompt: *'{response.modification_text}'*")

        # Step 2: Verifier Confidence
        with st.container(border=True):
            st.markdown("**2. Candidate Verification**")
            score_pct = int(response.confidence_score * 100)
            if response.verified:
                st.success(f"✅ **High Confidence Verified**: `{score_pct}%` (Threshold met)")
            else:
                st.warning(f"⚠️ **Low Confidence**: `{score_pct}%` (Deeper thinking triggered)")

            st.write(f"- Iterations: `{response.iterations_count}`")

        # Step 3: Deeper Thinking Refinements
        if response.refinement_notes:
            with st.container(border=True):
                st.markdown("**3. Deeper Thinking & Discrepancy Refinements**")
                for note in response.refinement_notes:
                    st.caption(f"• {note}")

    # -------------------------------------------------------------
    # 2. Standard Search / Multi-Agent RAG Inspection
    # -------------------------------------------------------------
    elif isinstance(response, SearchResponse):
        st.markdown("#### 🔄 LangGraph Multi-Agent Trace")

        # Step 1: Router Intent Extraction
        with st.container(border=True):
            st.markdown("**1. Router Intent & Constraint Extraction**")
            initial_filters = response.extracted_filters or response.filters_applied
            if initial_filters:
                st.write("Identified faceted constraints from query:")
                # Formatted constraint summary
                col_attr1, col_attr2 = st.columns(2)
                with col_attr1:
                    if "brand" in initial_filters:
                        st.markdown(f"🏷️ **Brand**: `{initial_filters['brand']}`")
                    if "base_colour" in initial_filters:
                        st.markdown(f"🎨 **Color**: `{initial_filters['base_colour']}`")
                    if "gender" in initial_filters:
                        st.markdown(f"👤 **Gender**: `{initial_filters['gender']}`")
                with col_attr2:
                    if "master_category" in initial_filters or "sub_category" in initial_filters:
                        cat_str = f"{initial_filters.get('master_category', '')} > {initial_filters.get('sub_category', '')}".strip(" >")
                        st.markdown(f"📂 **Category**: `{cat_str}`")
                    if "max_price" in initial_filters:
                        st.markdown(f"💵 **Max Price**: `${initial_filters['max_price']:.2f}`")
                    if "min_rating" in initial_filters:
                        st.markdown(f"★ **Min Rating**: `{initial_filters['min_rating']}`")

                with st.expander("View Raw Extracted Constraints"):
                    st.json(initial_filters)
            else:
                st.caption("No strict scalar filters extracted; relying on full semantic embedding matching.")

        # Step 2: Self-Reflection & Critique Loop
        with st.container(border=True):
            st.markdown("**2. Self-Reflection & Candidate Critique**")
            if response.reflection_notes:
                st.write("Critique & filter relaxation trace:")
                for note in response.reflection_notes:
                    st.info(f"🔄 {note}")

                if response.relaxed_filters:
                    st.markdown("**Relaxation Adjustments:**")
                    for rf in response.relaxed_filters:
                        st.caption(f"• {rf}")

                if response.filters_applied and response.filters_applied != initial_filters:
                    with st.expander("Final Active Filters After Relaxation"):
                        st.json(response.filters_applied)
            else:
                st.success("✅ First-pass candidate pool met confidence thresholds. No filter relaxation required.")

        # Step 3: Gemini 1.5 Flash Stylist Advice & Synthesis
        with st.container(border=True):
            st.markdown("**3. Grounded Stylist Advice (Gemini 1.5 Flash)**")
            if response.reasoning:
                st.markdown(response.reasoning)
            else:
                st.caption("Stylist advice synthesis unavailable or skipped.")
