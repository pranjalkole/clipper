"""
Responsive product card gallery component with score attribution and interactive quick-actions.
"""

from typing import Any
import httpx
import streamlit as st

from src.models.schemas import ProductCandidate


def _get_product_image_url(product_id: int, base_url: str | None = None) -> str:
    """Returns the local API endpoint URL for a product image."""
    base = (base_url or "http://127.0.0.1:8000").rstrip("/")
    return f"{base}/catalog/images/{product_id}"


def _fetch_image_bytes(url: str) -> bytes | None:
    """Attempts to fetch image bytes from remote URL with short timeout."""
    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                return resp.content
    except Exception:
        pass
    return None


def render_product_card(product: ProductCandidate, index: int, base_url: str | None = None) -> None:
    """Renders a single product candidate card with details and actions."""
    if product.image_url and product.image_url.startswith("http"):
        image_url = product.image_url
    elif product.image_url:
        base = (base_url or "http://127.0.0.1:8000").rstrip("/")
        image_url = f"{base}{product.image_url}"
    else:
        image_url = _get_product_image_url(product.id, base_url=base_url)

    # Card Container
    with st.container(border=True):
        # Product Image
        try:
            st.image(image_url, use_container_width=True)
        except Exception:
            st.caption("🖼️ Image preview unavailable")

        # Title & Brand
        st.markdown(f"**{product.product_display_name}**")
        st.caption(f"🏷️ **{product.brand}** • {product.gender} • {product.master_category}")

        # Price & Rating
        col_p, col_r = st.columns([1, 1])
        with col_p:
            st.markdown(f"### **${product.price:.2f}**")
        with col_r:
            st.markdown(f"★ **{product.rating:.1f}** `({product.rating_count})`")
            if product.in_stock:
                st.caption("🟢 In Stock")
            else:
                st.caption("🔴 Out of Stock")

        # Similarity Score Meter
        score_val = max(0.0, min(1.0, float(product.score)))
        score_pct = int(score_val * 100)
        st.progress(score_val, text=f"Confidence: {score_pct}% ({product.retrieval_path})")

        # Score Attribution Expander
        has_wiser = any([
            product.wiser_t2i_score is not None,
            product.wiser_comp_score is not None,
            product.wiser_mod_score is not None,
            product.wiser_vis_score is not None,
        ])

        with st.expander("🔍 Score Attribution Breakdown"):
            st.write(f"- **Overall Confidence**: `{score_val:.3f}`")
            st.write(f"- **Retrieval Method**: `{product.retrieval_path}`")
            if has_wiser:
                if product.wiser_t2i_score is not None:
                    st.write(f"- **Target Text Match (T2I)**: `{product.wiser_t2i_score:.3f}`")
                if product.wiser_comp_score is not None:
                    st.write(f"- **Visual Composition (Comp)**: `{product.wiser_comp_score:.3f}`")
                if product.wiser_mod_score is not None:
                    st.write(f"- **Modification Alignment (Mod)**: `{product.wiser_mod_score:.3f}`")
                if product.wiser_vis_score is not None:
                    st.write(f"- **Visual Feature Match (Vis)**: `{product.wiser_vis_score:.3f}`")
            if product.match_reasons:
                st.write("**Match Reasons:**")
                for r in product.match_reasons:
                    st.caption(f"• {r}")

        # Quick Action Buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔍 Similar", key=f"btn_sim_{product.id}_{index}", use_container_width=True):
                img_bytes = _fetch_image_bytes(image_url)
                if img_bytes:
                    # Stage filter updates to be applied before widgets are instantiated
                    pending: dict[str, list[str]] = {}
                    if product.master_category:
                        pending["filter_categories"] = [product.master_category]
                    if product.sub_category:
                        pending["filter_sub_categories"] = [product.sub_category]
                    if product.base_colour and str(product.base_colour).lower() != "unknown":
                        pending["filter_colors"] = [product.base_colour]
                    st.session_state["pending_filter_updates"] = pending
                    st.session_state["active_query_to_run"] = ("image", img_bytes)
                    st.rerun()
                else:
                    st.warning("Could not download product image for similarity search.")
        with btn_col2:
            if st.button("✏️ Modify (CIR)", key=f"btn_cir_{product.id}_{index}", use_container_width=True):
                img_bytes = _fetch_image_bytes(image_url)
                st.session_state["cir_ref_bytes"] = img_bytes
                st.session_state["cir_ref_name"] = f"Product #{product.id} ({product.brand})"
                st.session_state["cir_mod_prompt"] = f"same style as {product.product_display_name} but in "
                st.success(f"Loaded '{product.brand}' into CIR tab!")
                st.rerun()


def render_product_grid(
    candidates: list[ProductCandidate],
    num_columns: int = 3,
    base_url: str | None = None,
) -> None:
    """
    Renders a responsive multi-column grid of product candidate cards.
    """
    if not candidates:
        st.info("No product candidates to display. Try entering a query above or relaxing filters.")
        return

    cols = st.columns(num_columns)
    for i, product in enumerate(candidates):
        col = cols[i % num_columns]
        with col:
            render_product_card(product, index=i, base_url=base_url)
