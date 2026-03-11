#!/usr/bin/env python3
"""
DevOracle CLI — test ingestion and querying from the terminal.

Usage:
  python cli.py ingest --repo vercel/next.js --max-files 100
  python cli.py query "Where is authentication handled?"
  python cli.py status
"""

import argparse
import json
import sys
import logging

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint

# Suppress noisy loggers during CLI use
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

console = Console()


def cmd_ingest(args):
    from ingestion.github_loader import GitHubLoader
    from ingestion.chunker import CodeChunker
    from retrieval.vector_store import VectorStore

    console.print(f"\n[bold cyan]🔄 Ingesting:[/] {args.repo}\n")

    loader = GitHubLoader(repo_name=args.repo)
    chunker = CodeChunker()
    store = VectorStore()

    # Print repo info
    try:
        summary = loader.get_repo_summary()
        table = Table(title="Repository Info")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        for k, v in summary.items():
            table.add_row(str(k), str(v))
        console.print(table)
    except Exception:
        pass

    all_docs = []
    file_count = 0

    with console.status("[bold green]Streaming files from GitHub...") as status:
        for repo_file in loader.stream_files(max_files=args.max_files):
            chunks = chunker.chunk_file(repo_file)
            all_docs.extend(chunks)
            file_count += 1
            status.update(f"[bold green]Loaded {file_count} files, {len(all_docs)} chunks...")

            if len(all_docs) >= 200:
                store.add_documents(all_docs)
                all_docs = []

    if all_docs:
        store.add_documents(all_docs)

    stats = store.collection_stats()
    console.print(Panel(
        f"[green]✅ Ingestion complete![/]\n"
        f"Files processed: {file_count}\n"
        f"Total chunks in store: {stats['total_chunks']}\n"
        f"Unique files indexed: {stats['unique_files']}",
        title="Done",
    ))


def cmd_query(args):
    from retrieval.vector_store import VectorStore
    from retrieval.rag_engine import RAGEngine

    store = VectorStore()
    stats = store.collection_stats()

    if stats["total_chunks"] == 0:
        console.print("[red]❌ No data ingested yet. Run `python cli.py ingest --repo owner/repo` first.[/]")
        sys.exit(1)

    engine = RAGEngine(vector_store=store)

    console.print(f"\n[bold cyan]🔍 Query:[/] {args.question}\n")

    with console.status("[bold green]Thinking..."):
        response = engine.query(args.question)

    console.print(Panel(Markdown(response.answer), title="📖 Answer", border_style="green"))

    if response.sources:
        console.print("\n[bold]📁 Sources used:[/]")
        for src in response.sources:
            console.print(f"  • [cyan]{src['path']}[/]  [dim]{src.get('url', '')}[/]")


def cmd_status(args):
    from retrieval.vector_store import VectorStore
    store = VectorStore()
    stats = store.collection_stats()
    console.print(Panel(json.dumps(stats, indent=2), title="📊 Vector Store Status"))


def main():
    parser = argparse.ArgumentParser(
        description="DevOracle CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a GitHub repo")
    ingest_parser.add_argument("--repo", required=True, help="owner/repo e.g. vercel/next.js")
    ingest_parser.add_argument("--max-files", type=int, default=200)

    # query
    query_parser = subparsers.add_parser("query", help="Ask a question about the codebase")
    query_parser.add_argument("question", help="Your question in natural language")

    # status
    subparsers.add_parser("status", help="Show vector store stats")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
