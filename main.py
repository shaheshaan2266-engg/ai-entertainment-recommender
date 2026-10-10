"""
Project Entry Point: AI Entertainment Recommender.
Supports running in CLI mode for quick recommendations or starting the Interactive Web Server.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from ai_entertainment_recommender.recommender import UnifiedEntertainmentRecommender


def run_cli(query: str, mode: str, era: str, genre: str, top_n: int):
    print("=" * 65)
    print(f"🎬 AI Entertainment Recommender (Query: '{query}' | Mode: {mode})")
    print("=" * 65)

    recommender = UnifiedEntertainmentRecommender(auto_train=True)
    res = recommender.recommend(query=query, mode=mode, era_filter=era, genre_filter=genre, top_n=top_n)

    if res.get("identified"):
        info = res["identified"]
        print(f"\n[Matched Entity: {info['type']}] -> {info['title']}")

    if res.get("movies"):
        print("\n--- 🎬 Top Movie Recommendations ---")
        for i, m in enumerate(res["movies"], 1):
            dir_str = f"Dir: {m.get('director')}" if m.get('director') else ""
            print(f"{i}. {m['title']} ({m.get('release_year', 'N/A')}) {dir_str}")
            print(f"   Score: {m.get('similarity_score')} | Reason: {m.get('reason')}")

    if res.get("songs"):
        print("\n--- 🎵 Top Song / Soundtrack Recommendations ---")
        for i, s in enumerate(res["songs"], 1):
            print(f"{i}. {s['title']} ({s.get('movie_title', 'Single')}) - Singer: {s.get('singer', 'Various')}")
            print(f"   Score: {s.get('similarity_score')} | Reason: {s.get('reason')}")

    print("\n=" * 65)


def run_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = True):
    from app import app
    print(f"Starting AI Entertainment Recommender Web Server at http://{host}:{port}/ ...")
    app.run(host=host, port=port, debug=debug)


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Entertainment Recommender (Indian Cinema & Music)")
    parser.add_argument("--cli", action="store_true", help="Run recommendation in CLI mode")
    parser.add_argument("--server", action="store_true", help="Launch interactive web application server")
    parser.add_argument("--query", type=str, default="3 Idiots", help="Search query (movie, song, artist)")
    parser.add_argument("--mode", type=str, choices=["movies", "songs", "both"], default="both", help="Recommendation mode")
    parser.add_argument("--era", type=str, default="All", help="Filter by era (e.g. 'Latest (2020-2026)', '90s Golden Romance')")
    parser.add_argument("--genre", type=str, default="All", help="Filter by genre (e.g. 'Comedy', 'Action', 'Romance')")
    parser.add_argument("--top_n", type=int, default=6, help="Number of recommendations to return")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")

    args = parser.parse_args()

    if args.server:
        run_server(port=args.port)
    elif args.cli or len(sys.argv) > 1:
        run_cli(query=args.query, mode=args.mode, era=args.era, genre=args.genre, top_n=args.top_n)
    else:
        # Default behavior: run a demo and show instructions
        print("Running quick demo query... (use --server to launch the Web UI)")
        run_cli(query="Rockstar", mode="both", era="All", genre="All", top_n=5)
        print("\nTip: To start the fluid Web UI, run: python main.py --server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
