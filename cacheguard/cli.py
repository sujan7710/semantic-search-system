"""Interactive terminal client.  Usage:  python -m cacheguard.cli"""

from __future__ import annotations

from cacheguard.config import Settings
from cacheguard.service import build_default_service


def main() -> None:
    print("Loading CacheGuard...")
    service = build_default_service(Settings.from_env())
    print("Ready. Type a query, or 'exit' to quit.\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        result = service.handle(query)
        if result.cache_hit:
            print(f"\nCache hit (matched '{result.matched_query}', similarity {result.cache_similarity:.3f})")
        else:
            print("\nCache miss, searched documents")
        for source in result.response.sources[:3]:
            print(f"  [{source.score:.3f}] {source.newsgroup}: {source.snippet[:160]}")
        print(f"  {result.latency_ms:.1f} ms | {service.cache.stats()}\n")


if __name__ == "__main__":
    main()
