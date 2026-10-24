"""
Multimodal search input component supporting Text, Visual, and Composed Image Retrieval (CIR).
"""

import base64
import io
from typing import Any
from PIL import Image
import streamlit as st

from src.frontend.client import ApiClient, ApiClientError
from src.models.schemas import SearchFilterParams, WISERConfig


SUGGESTED_QUERIES = [
    "Men navy blue slim fit formal shirt",
    "Nike black running shoes under $100",
    "Women floral summer casual dress",
    "Classic leather chronograph watch",
    "Casual canvas sneakers for men",
]


def render_search_bar(
    client: ApiClient,
    filters: SearchFilterParams,
    limit: int = 10,
    confidence_threshold: float = 0.70,
    max_retries: int = 2,
) -> None:
    """
    Renders multimodal search tabs and dispatches requests to the FastAPI Gateway.
    """
    tab_text, tab_visual, tab_cir = st.tabs([
        "📝 Text Search",
        "📷 Visual Similarity",
        "🔄 Composed Retrieval (WISER CIR)",
    ])

    # -------------------------------------------------------------
    # 1. Text Search Tab
    # -------------------------------------------------------------
    with tab_text:
        st.markdown("##### 🔎 Search fashion catalog by descriptive text or styling query")
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            text_query = st.text_input(
                "Search Query",
                value=st.session_state.get("text_query_value", ""),
                placeholder="e.g. 'navy blue slim fit summer polo', 'red running shoes'",
                label_visibility="collapsed",
                key="text_search_input",
            )
        with col_btn:
            do_search_text = st.button("Search", key="btn_text_search", use_container_width=True, type="primary")

        # Query Chips / Suggestions
        st.caption("Suggested queries:")
        chip_cols = st.columns(len(SUGGESTED_QUERIES))
        for i, q in enumerate(SUGGESTED_QUERIES):
            with chip_cols[i]:
                if st.button(q, key=f"chip_{i}", use_container_width=True):
                    st.session_state["text_query_value"] = q
                    st.session_state["active_query_to_run"] = ("text", q)
                    st.rerun()

        if do_search_text and text_query.strip():
            st.session_state["active_query_to_run"] = ("text", text_query.strip())
            st.rerun()

    # -------------------------------------------------------------
    # 2. Visual Similarity Tab
    # -------------------------------------------------------------
    with tab_visual:
        st.markdown("##### 🖼️ Upload a reference garment or snap a live photo")
        vis_col1, vis_col2 = st.columns([1, 1])
        with vis_col1:
            uploaded_file = st.file_uploader(
                "Upload Image (JPG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
                key="visual_uploader",
            )
        with vis_col2:
            use_camera = st.checkbox("Use Live Camera Capture", value=False, key="toggle_camera")
            camera_file = None
            if use_camera:
                camera_file = st.camera_input("Take a photo", key="camera_input")

        target_file = camera_file or uploaded_file
        if target_file is not None:
            col_preview, col_act = st.columns([2, 3])
            with col_preview:
                st.image(target_file, caption="Query Reference Image", use_container_width=True)
            with col_act:
                st.write("**Image Ready for Retrieval**")
                st.caption(f"File: `{target_file.name}` | Size: `{len(target_file.getvalue()) / 1024:.1f} KB`")
                if st.button("Search Visually Similar Products", key="btn_visual_search", type="primary"):
                    st.session_state["active_query_to_run"] = ("image", target_file.getvalue())
                    st.rerun()

    # -------------------------------------------------------------
    # 3. Composed Image Retrieval (WISER CIR) Tab
    # -------------------------------------------------------------
    with tab_cir:
        st.markdown("##### 🎯 Composed Retrieval: Reference Image + Modification Prompt")
        st.caption("Zero-shot image modification (e.g. *'same silhouette but in olive green with zipper'*).")

        cir_ref_bytes: bytes | None = st.session_state.get("cir_ref_bytes")
        cir_ref_name: str = st.session_state.get("cir_ref_name", "Uploaded Image")

        cir_col1, cir_col2 = st.columns([1, 2])
        with cir_col1:
            cir_uploader = st.file_uploader(
                "Upload Reference Image",
                type=["jpg", "jpeg", "png", "webp"],
                key="cir_uploader",
            )
            if cir_uploader:
                cir_ref_bytes = cir_uploader.getvalue()
                cir_ref_name = cir_uploader.name

            if cir_ref_bytes:
                st.image(cir_ref_bytes, caption=f"Reference: {cir_ref_name}", use_container_width=True)
                if st.button("Clear Reference Image", key="clear_cir_ref"):
                    st.session_state["cir_ref_bytes"] = None
                    st.session_state["cir_ref_name"] = None
                    st.rerun()
            else:
                st.info("Upload an image or click **'Modify (CIR)'** on any product card below.")

        with cir_col2:
            mod_prompt = st.text_area(
                "Natural Language Modification Instruction",
                placeholder="e.g. 'same shoes but in navy blue and under $75', 'same shirt in white with short sleeves'",
                value=st.session_state.get("cir_mod_prompt", ""),
                height=110,
                key="cir_prompt_input",
            )
            enable_deeper_thinking = st.checkbox(
                "Enable WISER Deeper Thinking Refinement",
                value=True,
                help="Applies multi-turn discrepancy analysis and target caption reformulation if score < threshold.",
            )

            if st.button("Execute Composed Search", key="btn_cir_search", type="primary", disabled=not bool(mod_prompt.strip())):
                w_cfg = WISERConfig(
                    confidence_threshold=confidence_threshold,
                    max_refinement_retries=max_retries if enable_deeper_thinking else 0,
                )
                st.session_state["active_query_to_run"] = (
                    "cir",
                    {"image_bytes": cir_ref_bytes, "mod_text": mod_prompt.strip(), "wiser_config": w_cfg},
                )
                st.rerun()

    # -------------------------------------------------------------
    # 4. Dispatch Active Query Execution
    # -------------------------------------------------------------
    active_query = st.session_state.get("active_query_to_run")
    if active_query:
        st.session_state["active_query_to_run"] = None
        q_type, q_data = active_query

        with st.spinner("Executing multimodal search & multi-agent reflection..."):
            try:
                if q_type == "text":
                    res = client.search_text(query_text=q_data, filters=filters, limit=limit)
                    st.session_state["search_response"] = res
                    st.session_state["search_mode"] = "text"
                    st.session_state["search_query_display"] = q_data
                elif q_type == "image":
                    res = client.search_image(image_bytes=q_data, filters=filters, limit=limit)
                    st.session_state["search_response"] = res
                    st.session_state["search_mode"] = "image"
                    st.session_state["search_query_display"] = "Visual Similarity Query"
                elif q_type == "cir":
                    res = client.search_hybrid(
                        image_bytes=q_data.get("image_bytes"),
                        modification_text=q_data["mod_text"],
                        filters=filters,
                        limit=limit,
                        wiser_config=q_data.get("wiser_config"),
                    )
                    st.session_state["search_response"] = res
                    st.session_state["search_mode"] = "cir"
                    st.session_state["search_query_display"] = f"CIR: {q_data['mod_text']}"
            except ApiClientError as err:
                st.error(f"Search failed: {err}")
            except Exception as exc:
                st.error(f"Unexpected error executing search: {exc}")
