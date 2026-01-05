"""Export service for CSV and PDF generation."""
import csv
import io
from datetime import datetime
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.export import Export
from app.models.run import BenchmarkItemResult, BenchmarkRun, BenchmarkRunBenchmark

logger = get_logger(__name__)


async def generate_csv_export(
    db: AsyncSession,
    run: BenchmarkRun,
    include_items: bool = True,
) -> tuple[str, bytes]:
    """
    Generate CSV export for a run.
    
    Returns:
        Tuple of (filename, content_bytes)
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    header = [
        "run_id",
        "model_slug",
        "subset_pct",
        "run_status",
        "run_started_at",
        "run_completed_at",
        "overall_score",
        "benchmark_name",
        "benchmark_status",
        "benchmark_score",
        "sampled_items",
        "completed_items",
    ]

    if include_items:
        header.extend([
            "item_id",
            "prompt",
            "response",
            "expected",
            "is_correct",
            "score",
            "latency_ms",
            "error",
        ])

    writer.writerow(header)

    # Get benchmarks
    result = await db.execute(
        select(BenchmarkRunBenchmark).where(BenchmarkRunBenchmark.run_id == run.id)
    )
    run_benchmarks = result.scalars().all()

    for rb in run_benchmarks:
        base_row = [
            run.id,
            run.model_slug,
            run.subset_pct,
            run.status,
            run.started_at.isoformat() if run.started_at else "",
            run.completed_at.isoformat() if run.completed_at else "",
            run.overall_score or "",
            rb.benchmark_name,
            rb.status,
            rb.score or "",
            rb.sampled_items,
            rb.completed_items,
        ]

        if include_items:
            # Get item results
            item_result = await db.execute(
                select(BenchmarkItemResult)
                .where(BenchmarkItemResult.run_benchmark_id == rb.id)
                .order_by(BenchmarkItemResult.created_at)
            )
            items = item_result.scalars().all()

            if items:
                for item in items:
                    row = base_row + [
                        item.item_id,
                        (item.prompt or "")[:500],  # Truncate for CSV
                        (item.response or "")[:500],
                        (item.expected or "")[:500],
                        item.is_correct,
                        item.score or "",
                        item.latency_ms or "",
                        item.error or "",
                    ]
                    writer.writerow(row)
            else:
                # No items, still write benchmark summary row
                writer.writerow(base_row + ["", "", "", "", "", "", "", ""])
        else:
            writer.writerow(base_row)

    content = output.getvalue().encode("utf-8")
    filename = f"benchmark_run_{run.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

    return filename, content


async def generate_pdf_export(
    db: AsyncSession,
    run: BenchmarkRun,
) -> tuple[str, bytes]:
    """
    Generate PDF report for a run.
    
    Returns:
        Tuple of (filename, content_bytes)
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    styles = getSampleStyleSheet()
    story: list[Any] = []

    # Title
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
    )
    story.append(Paragraph("Benchmark Run Report", title_style))
    story.append(Spacer(1, 12))

    # Run summary
    summary_style = styles["Normal"]
    story.append(Paragraph(f"<b>Run ID:</b> {run.id}", summary_style))
    story.append(Paragraph(f"<b>Model:</b> {run.model_slug}", summary_style))
    story.append(Paragraph(f"<b>Subset:</b> {run.subset_pct}%", summary_style))
    story.append(Paragraph(f"<b>Status:</b> {run.status}", summary_style))
    if run.started_at:
        story.append(Paragraph(f"<b>Started:</b> {run.started_at.isoformat()}", summary_style))
    if run.completed_at:
        story.append(Paragraph(f"<b>Completed:</b> {run.completed_at.isoformat()}", summary_style))
    if run.overall_score is not None:
        story.append(Paragraph(f"<b>Overall Score:</b> {run.overall_score:.2%}", summary_style))
    story.append(Spacer(1, 20))

    # Get benchmarks
    result = await db.execute(
        select(BenchmarkRunBenchmark).where(BenchmarkRunBenchmark.run_id == run.id)
    )
    run_benchmarks = result.scalars().all()

    # Benchmark results table
    story.append(Paragraph("Benchmark Results", styles["Heading2"]))
    story.append(Spacer(1, 8))

    table_data = [["Benchmark", "Status", "Score", "Items (Sampled/Completed)"]]
    for rb in run_benchmarks:
        score_str = f"{rb.score:.2%}" if rb.score is not None else "-"
        items_str = f"{rb.sampled_items}/{rb.completed_items}"
        table_data.append([rb.benchmark_name, rb.status, score_str, items_str])

    table = Table(table_data, colWidths=[2.5 * inch, 1.2 * inch, 1 * inch, 2 * inch])
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ])
    )
    story.append(table)
    story.append(Spacer(1, 20))

    # Notes
    story.append(Paragraph("Notes", styles["Heading2"]))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            f"This report was generated on {datetime.utcnow().isoformat()}. "
            f"The benchmark run evaluated {len(run_benchmarks)} benchmarks with a {run.subset_pct}% "
            f"sample of items for deterministic subset evaluation.",
            summary_style,
        )
    )

    if run.subset_pct < 100:
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                f"<b>Note:</b> This run used deterministic subsampling at {run.subset_pct}%. "
                f"Metrics should be interpreted with this context. Full benchmark evaluation "
                f"requires running at 100%.",
                summary_style,
            )
        )

    # Build PDF
    doc.build(story)
    content = buffer.getvalue()
    filename = f"benchmark_report_{run.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

    return filename, content


async def save_export(
    db: AsyncSession,
    run_id: str,
    format: str,
    filename: str,
    content: bytes,
) -> Export:
    """Save export metadata."""
    export = Export(
        run_id=run_id,
        format=format,
        filename=filename,
        content_size=len(content),
    )
    db.add(export)
    await db.commit()
    return export


async def get_export(db: AsyncSession, run_id: str, format: str) -> Optional[Export]:
    """Get existing export if available."""
    result = await db.execute(
        select(Export)
        .where(Export.run_id == run_id, Export.format == format)
        .order_by(Export.generated_at.desc())
    )
    return result.scalar_one_or_none()
















