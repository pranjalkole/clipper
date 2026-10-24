import base64
import io
import logging
import math
import re
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from PIL import Image
from qdrant_client.models import Filter, ScoredPoint

from src.core.config import settings
from src.core.llm import get_chat_model
from src.db.qdrant import QdrantCatalogClient, get_qdrant_client
from src.models.embedder import MultimodalEmbedder, get_embedder
from src.models.schemas import (
    ComposedSearchRequest,
    ComposedSearchResponse,
    ProductCandidate,
    SearchFilterParams,
    WISERConfig,
    WISERRefinement,
    WISERVerificationResult,
)

logger = logging.getLogger(__name__)


def l2_normalize(vector: list[float]) -> list[float]:
    """Computes L2 unit normalization of a float vector."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm < 1e-12:
        return [0.0] * len(vector)
    return [x / norm for x in vector]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Computes cosine similarity between two unit-normalized vectors via dot product."""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}")
    return float(sum(a * b for a, b in zip(vec_a, vec_b)))


class WISERComposer:
    """
    Component for multimodal feature composition in MobileCLIP2-S0's 512-dimensional
    shared embedding space.
    """

    @staticmethod
    def compose_embeddings(
        v_img: list[float],
        v_mod: list[float],
        alpha: float = 1.0,
    ) -> list[float]:
        """
        Composes reference image embedding with modification text embedding:
        v_comp = norm(v_img + alpha * v_mod)

        Args:
            v_img: 512d normalized visual embedding of reference image.
            v_mod: 512d normalized text embedding of modification instruction.
            alpha: Scaling coefficient for modification direction.

        Returns:
            512d unit-norm composed vector representation.
        """
        if len(v_img) != len(v_mod):
            raise ValueError(
                f"Dimension mismatch between image ({len(v_img)}) and text ({len(v_mod)})"
            )

        composed = [img + alpha * mod for img, mod in zip(v_img, v_mod)]
        return l2_normalize(composed)


class WISEREditor:
    """
    Multimodal Editor node generating target descriptive captions (T_target)
    from (reference_image, modification_text).
    """

    @staticmethod
    def _image_to_base64_data_uri(image: Image.Image) -> str:
        """Converts PIL image into a data URI string."""
        buffered = io.BytesIO()
        rgb_img = image.convert("RGB") if image.mode != "RGB" else image
        rgb_img.save(buffered, format="JPEG", quality=85)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    @classmethod
    def generate_target_caption(
        cls,
        modification_text: str,
        reference_image: Image.Image | None = None,
        catalog_context: dict[str, Any] | None = None,
        llm: BaseChatModel | None = None,
    ) -> str:
        """
        Synthesizes a target descriptive caption T_target capturing preserved visual
        invariants from reference_image and updates requested in modification_text.

        Falls back gracefully to deterministic rule-based synthesis if LLM is unavailable
        or in offline environments.
        """
        if llm is not None:
            try:
                system_instruction = (
                    "You are an expert fashion search editor. Given a reference product image "
                    "and a user modification request, synthesize a detailed, search-optimized "
                    "target product caption for CLIP text-to-image retrieval. Describe the target "
                    "product's visual appearance: item type, primary color, pattern, cut, fabric, "
                    "and distinguishing styling features. Output ONLY the descriptive target caption."
                )

                content_parts: list[dict[str, Any] | str] = []
                if reference_image is not None:
                    data_uri = cls._image_to_base64_data_uri(reference_image)
                    content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})

                prompt_text = (
                    f"Modification instruction: '{modification_text}'.\n"
                    f"Context: {catalog_context or {}}.\n"
                    "Synthesize the target product caption:"
                )
                content_parts.append({"type": "text", "text": prompt_text})

                message = HumanMessage(content=content_parts)
                response = llm.invoke([message])
                caption = str(response.content).strip()
                if caption and len(caption) > 5:
                    logger.info("WISER Editor generated target caption via LLM: '%s'", caption)
                    return caption
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM target caption generation failed (%s). Using fallback.", exc)

        # Deterministic heuristic synthesis fallback (offline-friendly)
        context = catalog_context or {}
        gender = context.get("gender", "")
        master_category = context.get("master_category", "Apparel")
        article_type = context.get("article_type", "Item")
        brand = context.get("brand", "")

        parts = [p for p in [gender, brand, article_type, master_category] if p]
        base_desc = " ".join(parts) if parts else "fashion product"

        # Clean modification text
        clean_mod = modification_text.strip().rstrip(".")
        if clean_mod.lower().startswith("same but"):
            clean_mod = clean_mod[8:].strip()
        elif clean_mod.lower().startswith("same in"):
            clean_mod = clean_mod[7:].strip()
        elif clean_mod.lower().startswith("same"):
            clean_mod = clean_mod[4:].strip()

        fallback_caption = f"{base_desc} featuring {clean_mod}".strip()
        logger.debug("WISER Editor heuristic fallback caption: '%s'", fallback_caption)
        return fallback_caption


class WISERVerifier:
    """
    Adaptive Fusion Verifier node evaluating candidate pool alignment across:
      1. Target description semantic alignment (s_T2I)
      2. Composed visual feature alignment (s_comp)
      3. Direct modification prompt adherence (s_mod)
      4. Reference image preservation (s_vis)
    """

    @staticmethod
    def verify_and_fuse(
        candidate_points: list[ScoredPoint],
        v_t2i: list[float],
        v_comp: list[float],
        v_mod: list[float],
        v_img: list[float] | None = None,
        config: WISERConfig | None = None,
        path_provenance: dict[int, str] | None = None,
    ) -> WISERVerificationResult:
        """
        Evaluates candidate points and computes adaptive fused scores.

        Args:
            candidate_points: List of ScoredPoint instances retrieved from Qdrant.
            v_t2i: 512d normalized target caption embedding.
            v_comp: 512d normalized composed embedding.
            v_mod: 512d normalized modification text embedding.
            v_img: Optional 512d normalized reference image embedding.
            config: WISERConfig hyperparameters and weights.
            path_provenance: Mapping of candidate ID to retrieval path name ('t2i' or 'comp').

        Returns:
            WISERVerificationResult containing ranked ProductCandidate objects and confidence score.
        """
        cfg = config or WISERConfig()
        provenance = path_provenance or {}

        # Deduplicate points by id
        unique_points: dict[int, ScoredPoint] = {}
        for pt in candidate_points:
            pt_id = int(pt.id)
            if pt_id not in unique_points:
                unique_points[pt_id] = pt

        candidates: list[ProductCandidate] = []
        for pt_id, pt in unique_points.items():
            payload = pt.payload or {}
            vector = getattr(pt, "vector", None)

            # Compute individual alignment scores
            if vector is not None and isinstance(vector, list) and len(vector) == len(v_t2i):
                s_t2i = cosine_similarity(vector, v_t2i)
                s_comp = cosine_similarity(vector, v_comp)
                s_mod = cosine_similarity(vector, v_mod)
                s_vis = cosine_similarity(vector, v_img) if v_img else None
            else:
                # If vector is not present in ScoredPoint, use retrieval score and payload signals
                pt_score = float(pt.score) if pt.score is not None else 0.5
                path = provenance.get(pt_id, "unknown")
                if path == "t2i":
                    s_t2i = pt_score
                    s_comp = pt_score * 0.9
                else:
                    s_comp = pt_score
                    s_t2i = pt_score * 0.9
                s_mod = (s_t2i + s_comp) / 2.0
                s_vis = s_comp

            # Adaptive Fused Score
            fused_score = cfg.w_t2i * s_t2i + cfg.w_comp * s_comp + cfg.w_mod * s_mod

            candidate = ProductCandidate.from_payload_and_score(
                payload=payload,
                score=round(fused_score, 4),
                retrieval_path=provenance.get(pt_id, "wiser_fused"),
                wiser_t2i_score=round(s_t2i, 4),
                wiser_comp_score=round(s_comp, 4),
                wiser_mod_score=round(s_mod, 4),
                wiser_vis_score=round(s_vis, 4) if s_vis is not None else None,
            )
            candidates.append(candidate)

        # Sort candidates descending by fused score
        candidates.sort(key=lambda c: c.score, reverse=True)

        for rank_idx, cand in enumerate(candidates, start=1):
            cand.rank = rank_idx

        top_confidence = candidates[0].score if candidates else 0.0
        verified = top_confidence >= cfg.confidence_threshold

        return WISERVerificationResult(
            candidates=candidates,
            top_confidence=round(top_confidence, 4),
            verified=verified,
            confidence_threshold=cfg.confidence_threshold,
            pool_size=len(candidates),
        )


class WISERRefiner:
    """
    Deeper Thinking Refiner node executing self-reflection when verifier confidence < threshold.
    """

    @classmethod
    def diagnose_and_refine(
        cls,
        top_candidates: list[ProductCandidate],
        modification_text: str,
        current_target_caption: str,
        llm: BaseChatModel | None = None,
    ) -> WISERRefinement:
        """
        Analyzes candidate attributes against modification instructions to identify
        unmet constraints, producing a refined target caption and suggested filter tweaks.
        """
        if llm is not None:
            try:
                candidate_summary = [
                    f"- {c.product_display_name} (brand={c.brand}, price=${c.price}, rating={c.rating}, score={c.score})"
                    for c in top_candidates[:3]
                ]
                prompt = (
                    "You are a search critique and query refinement expert. A composed image retrieval query failed "
                    f"to achieve high confidence for modification instruction: '{modification_text}'.\n"
                    f"Current target caption was: '{current_target_caption}'.\n"
                    f"Retrieved candidate items were:\n" + "\n".join(candidate_summary) + "\n\n"
                    "1. Explain what attributes or styles are missing in the retrieved candidates.\n"
                    "2. Reformulate a stronger, more specific target caption emphasizing unmet attributes.\n"
                    "Output format:\n"
                    "DISCREPANCY: <one sentence explanation>\n"
                    "REFINED_CAPTION: <refined descriptive caption>"
                )
                response = llm.invoke(prompt)
                text = str(response.content)

                discrepancy = "Retrieved items exhibited low alignment with modification constraints."
                refined_caption = current_target_caption

                for line in text.splitlines():
                    if line.startswith("DISCREPANCY:"):
                        discrepancy = line.replace("DISCREPANCY:", "").strip()
                    elif line.startswith("REFINED_CAPTION:"):
                        refined_caption = line.replace("REFINED_CAPTION:", "").strip()

                return WISERRefinement(
                    discrepancy_analysis=discrepancy,
                    refined_target_caption=refined_caption,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM refiner failed (%s). Falling back to rule-based refinement.", exc)

        # Rule-based fallback refinement
        discrepancy = f"Low similarity confidence for modification: '{modification_text}'."
        refined_caption = f"{current_target_caption}, strictly matching {modification_text}"
        return WISERRefinement(
            discrepancy_analysis=discrepancy,
            refined_target_caption=refined_caption,
        )


class WISERComposedRetriever:
    """
    High-level orchestrator executing the WISER Composed Image Retrieval pipeline
    adapted for MobileCLIP2-S0 and Qdrant.
    """

    def __init__(
        self,
        embedder: MultimodalEmbedder | None = None,
        client: QdrantCatalogClient | None = None,
        llm: BaseChatModel | None = None,
        config: WISERConfig | None = None,
    ):
        self.embedder = embedder or get_embedder()
        self.client = client or get_qdrant_client()
        self.config = config or WISERConfig()

        if llm is not None:
            self.llm: BaseChatModel | None = llm
        else:
            try:
                self.llm = get_chat_model(temperature=0.0)
            except Exception:  # noqa: BLE001
                self.llm = None
                logger.info("Operating WISER in offline rule-based mode (no active LLM key).")

    def retrieve(
        self,
        request: ComposedSearchRequest,
        catalog_context: dict[str, Any] | None = None,
    ) -> ComposedSearchResponse:
        """
        Executes complete Retrieve-Verify-Refine loop for a composed image retrieval query.
        """
        t0 = time.perf_counter()
        cfg = request.wiser_config or self.config
        ref_image = request.load_image()
        mod_text = request.modification_text.strip()
        refinement_notes: list[str] = []

        # Step 1: Feature Extraction
        # Encode modification text
        v_mod = self.embedder.embed_text(mod_text)

        # Encode reference image (if provided; fallback to text if none)
        if ref_image is not None:
            v_img = self.embedder.embed_image(ref_image)
        else:
            # Fallback: encode reference text or empty prompt
            v_img = self.embedder.embed_text("fashion product")

        # Step 2: Feature Composition (I2I path)
        v_comp = WISERComposer.compose_embeddings(v_img, v_mod, alpha=cfg.alpha)

        # Build Qdrant scalar filter
        qdrant_filter: Filter | None = None
        if request.filters:
            qdrant_filter = self.client.build_filter(
                master_category=request.filters.master_category,
                sub_category=request.filters.sub_category,
                article_type=request.filters.article_type,
                gender=request.filters.gender,
                base_colour=request.filters.base_colour,
                brand=request.filters.brand,
                min_price=request.filters.min_price,
                max_price=request.filters.max_price,
                min_rating=request.filters.min_rating,
                in_stock_only=request.filters.in_stock_only,
                tags=request.filters.tags,
            )

        # Step 3: Editor (Wider Search)
        target_caption = WISEREditor.generate_target_caption(
            modification_text=mod_text,
            reference_image=ref_image,
            catalog_context=catalog_context,
            llm=self.llm,
        )

        iteration = 0
        pool_limit = request.limit * cfg.pool_factor

        while iteration <= cfg.max_refinement_retries:
            iteration += 1

            # Encode target caption (T2I path)
            v_t2i = self.embedder.embed_text(target_caption)

            # Wider Search: Dual-Path Retrieval
            # Path A: Text-to-Image (T2I)
            results_t2i = self.client.search(
                query_vector=v_t2i,
                filter_conditions=qdrant_filter,
                limit=pool_limit,
                with_vectors=True,
            )

            # Path B: Visual Composed (I2I)
            results_comp = self.client.search(
                query_vector=v_comp,
                filter_conditions=qdrant_filter,
                limit=pool_limit,
                with_vectors=True,
            )

            # Candidate pool aggregation with path provenance
            path_provenance: dict[int, str] = {}
            candidate_pool: list[ScoredPoint] = []

            for pt in results_t2i:
                pid = int(pt.id)
                path_provenance[pid] = "t2i"
                candidate_pool.append(pt)

            for pt in results_comp:
                pid = int(pt.id)
                if pid not in path_provenance:
                    path_provenance[pid] = "visual_comp"
                else:
                    path_provenance[pid] = "wiser_fused"
                candidate_pool.append(pt)

            # Step 4: Adaptive Fusion & Verification
            verification = WISERVerifier.verify_and_fuse(
                candidate_points=candidate_pool,
                v_t2i=v_t2i,
                v_comp=v_comp,
                v_mod=v_mod,
                v_img=v_img,
                config=cfg,
                path_provenance=path_provenance,
            )

            # Check verification gate
            if verification.verified or iteration > cfg.max_refinement_retries:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return ComposedSearchResponse(
                    modification_text=mod_text,
                    target_caption=target_caption,
                    candidates=verification.candidates[: request.limit],
                    total_found=len(verification.candidates),
                    confidence_score=verification.top_confidence,
                    verified=verification.verified,
                    iterations_count=iteration,
                    latency_ms=round(elapsed_ms, 2),
                    refinement_notes=refinement_notes,
                )

            # Step 5: Deeper Thinking (Self-Reflection)
            refinement = WISERRefiner.diagnose_and_refine(
                top_candidates=verification.candidates[:3],
                modification_text=mod_text,
                current_target_caption=target_caption,
                llm=self.llm,
            )
            refinement_notes.append(
                f"Iteration {iteration}: {refinement.discrepancy_analysis}"
            )
            target_caption = refinement.refined_target_caption

        # Fallback return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ComposedSearchResponse(
            modification_text=mod_text,
            target_caption=target_caption,
            candidates=verification.candidates[: request.limit],
            total_found=len(verification.candidates),
            confidence_score=verification.top_confidence,
            verified=verification.verified,
            iterations_count=iteration,
            latency_ms=round(elapsed_ms, 2),
            refinement_notes=refinement_notes,
        )
