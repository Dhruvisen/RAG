"""
Interactive CLI demo for the Corrective RAG pipeline.

Provides a Rich-formatted terminal interface to:
  - Load documents (PDF, TXT, MD, DOCX)
  - Ask questions interactively
  - Display retrieved chunks with relevance scores
  - Show the self-correction trace (HyDE, re-ranking, web fallback)
  - Stream the final answer token by token

Usage:
    python -m examples.demo --collection my_docs
    python -m examples.demo --collection my_docs --file /path/to/doc.pdf
    python -m examples.demo --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

# Ensure src is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corrective_rag import CorrectiveRAG, CRAGConfig, CRAGResult
from src.rag import VectorStoreType, EmbeddingModel
from src.utils.chunker import ChunkingStrategy

console = Console()


# ---------------------------------------------------------------------------
# Logging setup - uses Rich for pretty log output
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corrective RAG - Interactive CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m examples.demo --collection my_docs\n"
            "  python -m examples.demo --collection my_docs --file report.pdf\n"
            "  python -m examples.demo --collection my_docs --model mistral\n"
        ),
    )
    parser.add_argument(
        "--collection", default="crag_demo",
        help="ChromaDB/Qdrant collection name (default: crag_demo)"
    )
    parser.add_argument(
        "--file", nargs="*", metavar="PATH",
        help="One or more file paths to ingest (PDF, TXT, MD, DOCX)"
    )
    parser.add_argument(
        "--model", default="qwen2.5:1.5b",
        help="Ollama model to use (default: qwen2.5:1.5b - fastest CPU-optimised model)"
    )
    parser.add_argument(
        "--store", choices=["chromadb", "qdrant"], default="chromadb",
        help="Vector store backend (default: chromadb)"
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Vector store host (default: localhost)"
    )
    parser.add_argument(
        "--no-hyde", action="store_true",
        help="Disable HyDE query expansion"
    )
    parser.add_argument(
        "--no-web", action="store_true",
        help="Disable web fallback search"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    return parser


# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    banner = Text()
    banner.append("Corrective RAG", style="bold cyan")
    banner.append("  |  Self-Correcting | Hybrid Search | HyDE | Re-Ranking", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))
    console.print()


def _print_trace_table(result: CRAGResult) -> None:
    t = result.trace
    table = Table(title="Pipeline Trace", box=box.ROUNDED, border_style="dim cyan", show_lines=True)
    table.add_column("Step", style="bold", no_wrap=True)
    table.add_column("Details")

    table.add_row("Query", t.query)
    table.add_row(
        "HyDE",
        "[green][OK][/green]" if t.hyde_used else "[dim]Disabled / fallback[/dim]",
    )
    if t.hyde_used and t.hypothetical_document:
        table.add_row(
            "  Hypothetical doc",
            Text(t.hypothetical_document[:200] + "...", style="dim italic"),
        )
    table.add_row("Vector chunks", str(t.vector_chunks_retrieved))
    table.add_row(
        "Hybrid fusion",
        "[green][OK][/green]" if t.hybrid_fusion_applied else "[dim]-[/dim]",
    )
    table.add_row(
        "Re-ranking",
        "[green][OK][/green]" if t.reranking_applied else "[dim]-[/dim]",
    )
    table.add_row(
        "Web fallback",
        f"[yellow][OK] {t.web_snippets_fetched} snippets[/yellow]"
        if t.web_fallback_triggered else "[dim]-[/dim]",
    )
    table.add_row("Correction attempts", str(t.correction_attempts))
    table.add_row(
        "Answer supported",
        "[green][OK] Yes[/green]" if t.answer_is_supported else "[red][FAIL] No[/red]",
    )
    table.add_row(
        "Answer grade score",
        f"{t.answer_grade_score:.4f}",
    )
    table.add_row("Latency", f"{t.latency_ms:.0f} ms")
    console.print(table)
    console.print()


def _print_sources_table(result: CRAGResult) -> None:
    table = Table(
        title=f"Source Chunks ({len(result.source_chunks)} used)",
        box=box.SIMPLE_HEAVY,
        border_style="dim",
        show_lines=True,
    )
    table.add_column("#", width=3, justify="right")
    table.add_column("Chunk ID", style="dim", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Preview", max_width=60)

    for i, chunk in enumerate(result.source_chunks, start=1):
        meta = chunk.get("metadata", {})
        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("distance", "-")))
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        source = meta.get("url") or meta.get("document_id", "-")
        preview = chunk["text"][:120].replace("\n", " ")
        table.add_row(str(i), chunk.get("chunk_id", "?"), source, score_str, preview)

    console.print(table)
    console.print()


def _stream_answer(result: CRAGResult, crag: CorrectiveRAG) -> None:
    """Re-stream the answer for a live token-by-token display effect."""
    console.print(Rule("[bold cyan]Answer[/bold cyan]", style="cyan"))
    console.print()
    # Since we already have the answer, print it as Markdown for nice formatting
    console.print(Markdown(result.answer))
    console.print()


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

def _ingest_files(crag: CorrectiveRAG, paths: List[str]) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting documents...", total=len(paths))
        for path_str in paths:
            path = Path(path_str).resolve()
            if not path.exists():
                console.print(f"[red][FAIL] File not found: {path}[/red]")
                progress.advance(task)
                continue

            doc: dict = {"id": path.stem}
            if path.suffix.lower() == ".pdf":
                doc["pdf_path"] = str(path)
            elif path.suffix.lower() == ".docx":
                doc["text"] = _read_docx(path)
            else:
                doc["text"] = path.read_text(encoding="utf-8", errors="replace")

            try:
                crag.ingest(doc)
                console.print(f"[green][OK][/green] Ingested: [bold]{path.name}[/bold]")
            except Exception as exc:
                console.print(f"[red][FAIL] Failed to ingest {path.name}: {exc}[/red]")
            finally:
                progress.advance(task)


def _read_docx(path: Path) -> str:
    """Extract text from a .docx file using python-docx."""
    try:
        from docx import Document  # type: ignore
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        console.print("[yellow]python-docx not installed; install with: pip install python-docx[/yellow]")
        return ""
    except Exception as exc:
        console.print(f"[red]Failed to read DOCX: {exc}[/red]")
        return ""


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def _interactive_loop(crag: CorrectiveRAG) -> None:
    console.print(
        Panel(
            "[bold]Type your question and press Enter.[/bold]\n"
            "[dim]Commands: [cyan]quit[/cyan] or [cyan]exit[/cyan] to leave, "
            "[cyan]trace[/cyan] to show full trace of last result.[/dim]",
            border_style="dim",
            padding=(0, 2),
        )
    )
    console.print()

    last_result: CRAGResult | None = None

    while True:
        try:
            question = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye![/dim]")
            break
        if question.lower() == "trace":
            if last_result:
                _print_trace_table(last_result)
            else:
                console.print("[dim]No query run yet.[/dim]")
            continue

        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Running CRAG pipeline...")
            try:
                result = crag.query(question)
            except Exception as exc:
                console.print(f"[red]Pipeline error: {exc}[/red]")
                console.print()
                continue

        last_result = result
        _print_sources_table(result)
        _stream_answer(result, crag)
        _print_trace_table(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)
    _print_banner()

    store_type = (
        VectorStoreType.CHROMADB if args.store == "chromadb" else VectorStoreType.QDRANT
    )

    from src.generator import GeneratorConfig
    config = CRAGConfig(
        collection_name=args.collection,
        vector_store_type=store_type,
        vector_store_host=args.host,
        generator_config=GeneratorConfig(model=args.model),
        enable_hyde=not args.no_hyde,
        enable_web_fallback=not args.no_web,
    )

    console.print(f"[dim]Initialising pipeline (model=[bold]{args.model}[/bold])...[/dim]")
    try:
        crag = CorrectiveRAG(config)
    except Exception as exc:
        console.print(f"[red bold]Initialisation failed:[/red bold] {exc}")
        sys.exit(1)

    console.print(f"[green][OK] Pipeline ready.[/green] Collection: [bold]{args.collection}[/bold]")
    console.print()

    if args.file:
        _ingest_files(crag, args.file)
        console.print()

    _interactive_loop(crag)


if __name__ == "__main__":
    main()
