import argparse
import logging
import sys
import time
from typing import Any

from datasets import load_dataset
from PIL import Image

from src.core.config import settings
from src.db.qdrant import QdrantCatalogClient, get_qdrant_client
from src.models.embedder import MultimodalEmbedder, get_embedder
from src.pipeline.enrichment import enrich_catalog_item

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingestion")


def ingest_catalog(
    limit: int | None = None,
    batch_size: int = 32,
    recreate: bool = False,
    streaming: bool = False,
    dataset_name: str | None = None,
    client: QdrantCatalogClient | None = None,
    embedder: MultimodalEmbedder | None = None,
) -> dict[str, Any]:
    """
    Ingests e-commerce catalog dataset into Qdrant vector database.

    Args:
        limit: Maximum number of catalog items to index (None = entire dataset).
        batch_size: Batch size for visual embedding inference and Qdrant upserts.
        recreate: If True, drops and recreates the Qdrant collection before ingestion.
        streaming: If True, streams records without loading the full split into memory.
        dataset_name: Hugging Face dataset identifier.
        client: Optional QdrantCatalogClient instance.
        embedder: Optional MultimodalEmbedder instance.

    Returns:
        Summary metrics dictionary including indexed count, elapsed time, and throughput.
    """
    dataset_name = dataset_name or settings.dataset_name
    client = client or get_qdrant_client()
    embedder = embedder or get_embedder()

    logger.info(
        "Starting ingestion pipeline: dataset=%s, limit=%s, batch_size=%d, recreate=%s",
        dataset_name,
        limit,
        batch_size,
        recreate,
    )

    # Step 1: Ensure collection and payload indexes exist
    client.init_collection(recreate=recreate)

    # Step 2: Load dataset
    t0 = time.perf_counter()
    if streaming:
        logger.info("Loading dataset in streaming mode...")
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    else:
        split_spec = f"train[:{limit}]" if limit is not None else "train"
        logger.info("Loading dataset split: %s", split_spec)
        dataset = load_dataset(dataset_name, split=split_spec)

    t_dataset_loaded = time.perf_counter()
    logger.info("Dataset ready in %.2fs", t_dataset_loaded - t0)

    # Step 3: Batch processing and indexing
    indexed_count = 0
    batch_images: list[Image.Image] = []
    batch_payloads: list[dict[str, Any]] = []

    def flush_batch() -> None:
        nonlocal indexed_count, batch_images, batch_payloads
        if not batch_payloads:
            return

        batch_t0 = time.perf_counter()
        # Compute visual embeddings
        vectors = embedder.embed_images(batch_images, batch_size=len(batch_images))
        # Upsert into Qdrant
        client.upsert_catalog_items(batch_payloads, vectors, wait=True)

        indexed_count += len(batch_payloads)
        batch_elapsed = time.perf_counter() - batch_t0
        throughput = len(batch_payloads) / max(batch_elapsed, 1e-5)

        logger.info(
            "Indexed items %d to %d (batch of %d in %.2fs - %.1f items/sec)",
            indexed_count - len(batch_payloads) + 1,
            indexed_count,
            len(batch_payloads),
            batch_elapsed,
            throughput,
        )

        batch_images = []
        batch_payloads = []

    for item in dataset:
        try:
            raw_image = item["image"]
            if not isinstance(raw_image, Image.Image):
                continue
            enriched = enrich_catalog_item(item)
            payload = enriched.to_payload()

            batch_images.append(raw_image)
            batch_payloads.append(payload)

            if len(batch_payloads) >= batch_size:
                flush_batch()

            if limit is not None and indexed_count + len(batch_payloads) >= limit:
                break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error processing item id=%s: %s", item.get("id"), exc)

    # Flush any remaining items in final chunk
    if batch_payloads:
        flush_batch()

    total_elapsed = time.perf_counter() - t_dataset_loaded
    overall_throughput = indexed_count / max(total_elapsed, 1e-5)

    stats = {
        "items_indexed": indexed_count,
        "elapsed_seconds": round(total_elapsed, 2),
        "throughput_items_per_sec": round(overall_throughput, 1),
        "collection_name": client.collection_name,
        "collection_stats": client.get_stats(),
    }

    logger.info(
        "Ingestion completed: %d items indexed in %.2fs (%.1f items/sec). Total collection points: %d",
        stats["items_indexed"],
        stats["elapsed_seconds"],
        stats["throughput_items_per_sec"],
        stats["collection_stats"].get("points_count", 0),
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest e-commerce fashion catalog into Qdrant vector database."
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Maximum number of items to ingest (e.g. 1000 for quick testing).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=settings.ingestion_batch_size,
        help=f"Batch size for embedding and upsert (default: {settings.ingestion_batch_size}).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the Qdrant collection before ingestion.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream dataset records instead of caching.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=settings.dataset_name,
        help=f"Hugging Face dataset identifier (default: {settings.dataset_name}).",
    )

    args = parser.parse_args()

    try:
        stats = ingest_catalog(
            limit=args.limit,
            batch_size=args.batch_size,
            recreate=args.recreate,
            streaming=args.streaming,
            dataset_name=args.dataset,
        )
        print("\n" + "=" * 50)
        print("INGESTION SUMMARY")
        print("=" * 50)
        print(f"Items Indexed     : {stats['items_indexed']}")
        print(f"Elapsed Time      : {stats['elapsed_seconds']} s")
        print(f"Throughput        : {stats['throughput_items_per_sec']} items/sec")
        print(f"Collection Name   : {stats['collection_name']}")
        print(f"Total Points in DB: {stats['collection_stats'].get('points_count', 0)}")
        print("=" * 50)
    except Exception:
        logger.exception("Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
