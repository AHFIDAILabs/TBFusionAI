"""
Main entry point for TBFusionAI application.

Provides CLI interface for:
- Running pipelines (with smart skip logic)
- Starting API server
- Model training and evaluation

CRITICAL FIX: Properly handles Typer OptionInfo objects to prevent path errors.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from clean_cache import clean_project
from src.config import get_config
from src.logger import get_logger, setup_logger

# Initialize Typer app
app = typer.Typer(
    name="tbfusionai",
    help="TBFusionAI: AI-powered TB detection using cough sound analysis",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


def check_pipeline_status() -> dict:
    """Check which pipeline stages have been completed by verifying artifacts."""
    config = get_config()
    status = {
        "data_ingestion": False,
        "data_processing": False,
        "model_training": False,
        "model_evaluation": False,
    }

    # Check for artifact existence to determine completion status
    if (config.paths.dataset_path / "raw_data").exists():
        status["data_ingestion"] = True

    if (config.paths.labeled_data_path / "wav2vec2_balanced_ctgan.csv").exists():
        status["data_processing"] = True

    if (config.paths.models_path / "training_metadata.joblib").exists():
        status["model_training"] = True

    if (config.paths.models_path / "cost_sensitive_ensemble_model.joblib").exists():
        status["model_evaluation"] = True

    return status


def display_pipeline_status():
    """Display current pipeline status in a table."""
    status = check_pipeline_status()
    table = Table(
        title="Pipeline Status", show_header=True, header_style="bold magenta"
    )
    table.add_column("Stage", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Description", style="dim")

    stages = [
        ("data_ingestion", "1. Data Ingestion", "Download CODA TB dataset"),
        (
            "data_processing",
            "2. Data Processing",
            "Extract audio features & balance data",
        ),
        ("model_training", "3. Model Training", "Train multiple ML models"),
        ("model_evaluation", "4. Model Evaluation", "Create ensemble model"),
    ]

    for key, name, desc in stages:
        icon = "✅ Complete" if status[key] else "⏳ Pending"
        table.add_row(name, icon, desc)

    console.print(table)


@app.command()
def ingest_data(
    force: bool = typer.Option(False, "--force", help="Force re-run even if completed")
):
    """Run data ingestion pipeline."""
    status = check_pipeline_status()

    if status["data_ingestion"] and not force:
        console.print("[yellow]⚠ Data already ingested. Use --force to re-run[/yellow]")
        return

    console.print("\n[bold blue]Starting Data Ingestion[/bold blue]\n")

    try:
        from src.pipelines.data_ingestion import run_data_ingestion

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting data...", total=None)
            asyncio.run(run_data_ingestion())
            progress.update(task, completed=True)

        console.print("[green]✓ Data ingestion completed![/green]")

    except Exception as e:
        console.print(f"[bold red]✗ Data ingestion failed: {e}[/bold red]")
        logger.error(f"Data ingestion failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def process_data(
    force: bool = typer.Option(False, "--force", help="Force re-run even if completed")
):
    """Run data processing pipeline."""
    status = check_pipeline_status()

    if not status["data_ingestion"]:
        console.print("[bold red]✗ Data ingestion must be completed first![/bold red]")
        sys.exit(1)

    if status["data_processing"] and not force:
        console.print(
            "[yellow]⚠ Data already processed. Use --force to re-run[/yellow]"
        )
        return

    console.print("\n[bold blue]Starting Data Processing[/bold blue]\n")
    console.print(
        "[yellow]⚠ This may take 60+ minutes for audio feature extraction[/yellow]\n"
    )

    try:
        from src.pipelines.data_processing import run_data_processing

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing data...", total=None)
            result = asyncio.run(run_data_processing())
            progress.update(task, completed=True)

        console.print(f"\n[green]✓ Data processing completed![/green]")
        console.print(f"[dim]Final dataset shape: {result.shape}[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Data processing failed: {e}[/bold red]")
        logger.error(f"Data processing failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def train_models(
    data_path: Optional[str] = typer.Option(
        None, "--data-path", help="Path to training data CSV"
    ),
    force: bool = typer.Option(False, "--force", help="Force re-run even if completed"),
):
    """
    Run model training pipeline.

    FIXED: Properly handles Typer's OptionInfo objects to prevent path errors.
    """
    status = check_pipeline_status()

    if not status["data_processing"]:
        console.print("[bold red]✗ Data processing must be completed first![/bold red]")
        sys.exit(1)

    if status["model_training"] and not force:
        console.print(
            "[yellow]⚠ Models already trained. Use --force to re-run[/yellow]"
        )
        return

    console.print("\n[bold blue]Starting Model Training Pipeline[/bold blue]\n")
    console.print("[yellow]⚠ This may take 30+ minutes[/yellow]\n")

    try:
        from src.pipelines.model_training import run_model_training

        # CRITICAL FIX: Check if data_path is actually a string
        # Typer creates OptionInfo objects even when option is not provided
        clean_path = None
        if data_path is not None and isinstance(data_path, str) and data_path.strip():
            clean_path = Path(data_path)
            if not clean_path.exists():
                console.print(
                    f"[bold red]✗ Data path does not exist: {clean_path}[/bold red]"
                )
                sys.exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training models...", total=None)
            results = asyncio.run(run_model_training(clean_path))
            progress.update(task, completed=True)

        console.print(f"\n[bold green]✓ Training completed successfully![/bold green]")
        console.print(f"[dim]Best model: {results['best_model']}[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Model training failed: {e}[/bold red]")
        logger.error(f"Model training failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def evaluate_models(
    force: bool = typer.Option(False, "--force", help="Force re-run even if completed")
):
    """Run model evaluation and ensemble creation."""
    status = check_pipeline_status()

    if not status["model_training"]:
        console.print("[bold red]✗ Model training must be completed first![/bold red]")
        sys.exit(1)

    if status["model_evaluation"] and not force:
        console.print(
            "[yellow]⚠ Models already evaluated. Use --force to re-run[/yellow]"
        )
        return

    console.print("\n[bold blue]Starting Model Evaluation Pipeline[/bold blue]\n")
    console.print("[yellow]⚠ This may take 10+ minutes[/yellow]\n")

    try:
        from src.pipelines.model_evaluation import run_model_evaluation

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating models...", total=None)
            results = asyncio.run(run_model_evaluation())
            progress.update(task, completed=True)

        console.print(
            f"\n[bold green]✓ Evaluation completed successfully![/bold green]"
        )
        console.print(f"[dim]Ensemble created with enhanced reject option[/dim]")

    except Exception as e:
        console.print(f"[bold red]✗ Model evaluation failed: {e}[/bold red]")
        logger.error(f"Model evaluation failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def run_pipeline(
    force: bool = typer.Option(False, "--force", help="Force re-run all stages"),
    clean: bool = typer.Option(
        False, "--clean", help="Clean artifacts before starting"
    ),
):
    """
    Run the complete ML pipeline (all stages).

    Stages:
    1. Data Ingestion
    2. Data Processing
    3. Model Training
    4. Model Evaluation
    """
    console.print(
        Panel.fit(
            "[bold cyan]TBFusionAI Complete ML Pipeline[/bold cyan]\n"
            "This will run all necessary stages to train models",
            border_style="cyan",
        )
    )

    # Clean if requested
    if clean:
        console.print("\n[bold yellow]🧹 Cleaning artifacts...[/bold yellow]")
        clean_project()

    # Show initial status
    console.print()
    display_pipeline_status()
    console.print()

    # Run all stages in sequence
    try:
        # Stage 1: Data Ingestion
        console.print(
            Panel("[bold]Stage 1/4: Data Ingestion[/bold]", border_style="blue")
        )
        ingest_data(force=force)
        console.print()

        # Stage 2: Data Processing
        console.print(
            Panel("[bold]Stage 2/4: Data Processing[/bold]", border_style="blue")
        )
        process_data(force=force)
        console.print()

        # Stage 3: Model Training
        console.print(
            Panel("[bold]Stage 3/4: Model Training[/bold]", border_style="blue")
        )
        train_models(force=force)
        console.print()

        # Stage 4: Model Evaluation
        console.print(
            Panel("[bold]Stage 4/4: Model Evaluation[/bold]", border_style="blue")
        )
        evaluate_models(force=force)
        console.print()

        # Final summary
        console.print(
            Panel.fit(
                "[bold green]🏁 PIPELINE COMPLETED SUCCESSFULLY![/bold green]\n\n"
                "All models trained and ensemble created.\n"
                "Ready for inference and deployment.",
                border_style="green",
            )
        )

    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]⚠ Pipeline interrupted by user[/bold yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n\n[bold red]✗ Pipeline failed: {e}[/bold red]")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def status():
    """Show current pipeline status."""
    console.print()
    display_pipeline_status()
    console.print()


@app.command()
def clean(confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")):
    """Clean all artifacts and cached data."""
    if not confirm:
        response = typer.confirm(
            "This will delete all artifacts, models, and processed data. Continue?",
            default=False,
        )
        if not response:
            console.print("[yellow]Cancelled[/yellow]")
            return

    console.print("\n[bold yellow]🧹 Cleaning project artifacts...[/bold yellow]")
    clean_project()
    console.print("[green]✓ Cleanup complete![/green]\n")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    workers: int = typer.Option(4, "--workers", help="Number of worker processes"),
):
    """Start the FastAPI server for inference."""
    console.print(
        Panel.fit(
            f"[bold cyan]Starting TBFusionAI API Server[/bold cyan]\n\n"
            f"Host: {host}:{port}\n"
            f"Workers: {workers}\n"
            f"Auto-reload: {reload}",
            border_style="cyan",
        )
    )

    # Check if models are trained
    status = check_pipeline_status()
    if not status["model_training"]:
        console.print("\n[bold red]✗ No trained models found![/bold red]")
        console.print("[yellow]Run 'python main.py run-pipeline' first[/yellow]\n")
        sys.exit(1)

    try:
        import uvicorn

        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=1 if reload else workers,
            log_level="info",
        )
    except Exception as e:
        console.print(f"[bold red]✗ Server failed: {e}[/bold red]")
        logger.error(f"API server failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def version():
    """Show TBFusionAI version information."""
    config = get_config()
    console.print(
        Panel.fit(
            f"[bold cyan]TBFusionAI[/bold cyan]\n\n"
            f"Version: {config.api.app_version}\n"
            f"Description: {config.api.app_description}",
            border_style="cyan",
        )
    )


def main():
    """Main entry point."""
    setup_logger()
    app()


if __name__ == "__main__":
    main()
