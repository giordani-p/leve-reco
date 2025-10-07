#!/usr/bin/env python3
"""
CLI para migração de embeddings para PostgreSQL + pgvector.

Uso:
    python helpers/migrate_embeddings.py --from-files files/trails.json
    python helpers/migrate_embeddings.py --from-api
    python helpers/migrate_embeddings.py --validate
    python helpers/migrate_embeddings.py --clear-all --confirm
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reco.config import RecoConfig
from reco.migration.migrator import DataMigrator, MigrationStats

app = typer.Typer(help="Migrate embeddings to PostgreSQL + pgvector")
console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_stats(stats: MigrationStats):
    """Print migration statistics in a nice table."""
    table = Table(title="Migration Statistics")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Details", style="yellow")
    
    table.add_row("Total Trails", str(stats.total_trails), "Trails loaded from source")
    table.add_row("Processed", str(stats.processed_trails), f"{stats.success_rate:.1f}% success rate")
    table.add_row("New Embeddings", str(stats.new_embeddings), "First-time embeddings")
    table.add_row("Updated", str(stats.updated_embeddings), "Updated existing embeddings")
    table.add_row("Skipped", str(stats.skipped_trails), "No changes needed")
    table.add_row("Errors", str(stats.error_trails), "Failed to process")
    
    if stats.duration_seconds:
        table.add_row("Duration", f"{stats.duration_seconds:.1f}s", "Total migration time")
    
    console.print(table)


@app.command()
def from_files(
    file_path: str = typer.Argument(..., help="Path to trails JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size for processing"),
):
    """Migrate embeddings from local JSON file."""
    setup_logging(verbose)
    
    console.print(f"[bold blue]Migrating from file: {file_path}[/bold blue]")
    
    # Carrega configuração
    cfg = RecoConfig()
    
    # Cria migrator com tamanho de lote customizado
    migrator = DataMigrator(cfg, batch_size=batch_size)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Migrating embeddings...", total=None)
            
            # Executa migração
            stats = migrator.migrate_from_files(file_path)
            
            progress.update(task, description="Migration completed!")
        
        # Imprime resultados
        print_stats(stats)
        
        if stats.error_trails > 0:
            console.print(f"[red]Warning: {stats.error_trails} trails failed to process[/red]")
            sys.exit(1)
        else:
            console.print("[green]Migration completed successfully![/green]")
    
    except Exception as e:
        console.print(f"[red]Migration failed: {e}[/red]")
        sys.exit(1)
    
    finally:
        migrator.close()


@app.command()
def from_api(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size for processing"),
):
    """Migrate embeddings from API."""
    setup_logging(verbose)
    
    console.print("[bold blue]Migrating from API[/bold blue]")
    
    # Carrega configuração
    cfg = RecoConfig()
    
    # Cria migrator com tamanho de lote customizado
    migrator = DataMigrator(cfg, batch_size=batch_size)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Migrating embeddings...", total=None)
            
            # Executa migração
            stats = migrator.migrate_from_api()
            
            progress.update(task, description="Migration completed!")
        
        # Imprime resultados
        print_stats(stats)
        
        if stats.error_trails > 0:
            console.print(f"[red]Warning: {stats.error_trails} trails failed to process[/red]")
            sys.exit(1)
        else:
            console.print("[green]Migration completed successfully![/green]")
    
    except Exception as e:
        console.print(f"[red]Migration failed: {e}[/red]")
        sys.exit(1)
    
    finally:
        migrator.close()


@app.command()
def validate(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Validate migration status."""
    setup_logging(verbose)
    
    console.print("[bold blue]Validating migration status[/bold blue]")
    
    # Carrega configuração
    cfg = RecoConfig()
    
    # Create migrator
    migrator = DataMigrator(cfg)
    
    try:
        # Obtém status
        status = migrator.get_migration_status()
        
        # Imprime tabela de status
        table = Table(title="Migration Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Embeddings", str(status['total_embeddings']))
        table.add_row("Model Versions", ", ".join(status['model_versions']))
        table.add_row("Last Updated", str(status['last_updated'] or "Never"))
        table.add_row("Cached Hashes", str(status['cached_hashes']))
        
        console.print(table)
        
        # Valida
        validation = migrator.validate_migration()
        
        if validation['validation_passed']:
            console.print("[green]Validation passed![/green]")
        else:
            console.print("[red]Validation failed![/red]")
            for issue in validation['issues']:
                console.print(f"[red]  - {issue}[/red]")
            sys.exit(1)
    
    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        sys.exit(1)
    
    finally:
        migrator.close()


@app.command()
def cleanup(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be cleaned up"),
):
    """Clean up orphaned embeddings."""
    setup_logging(verbose)
    
    console.print("[bold blue]Cleaning up orphaned embeddings[/bold blue]")
    
    # Carrega configuração
    cfg = RecoConfig()
    
    # Create migrator
    migrator = DataMigrator(cfg)
    
    try:
        # Obtém embeddings atuais
        status = migrator.get_migration_status()
        console.print(f"Current embeddings: {status['total_embeddings']}")
        
        if dry_run:
            console.print("[yellow]Dry run mode - no changes will be made[/yellow]")
            # Em uma implementação real, você consultaria embeddings órfãos
            console.print("Would clean up orphaned embeddings (implementation needed)")
        else:
            # Por enquanto, não podemos determinar facilmente IDs de trilhas válidas sem acesso à API
            console.print("[yellow]Cleanup requires valid trail IDs - not implemented yet[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Cleanup failed: {e}[/red]")
        sys.exit(1)
    
    finally:
        migrator.close()


@app.command()
def clear_all(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm deletion without prompt"),
):
    """Clear ALL embeddings from the database."""
    setup_logging(verbose)
    
    console.print("[bold red]WARNING: This will delete ALL embeddings from the database![/bold red]")
    
    if not confirm:
        console.print("[yellow]Use --confirm flag to proceed with deletion[/yellow]")
        return
    
    # Carrega configuração
    cfg = RecoConfig()
    
    # Create migrator
    migrator = DataMigrator(cfg)
    
    try:
        # Obtém status atual
        status = migrator.get_migration_status()
        total_embeddings = status['total_embeddings']
        
        console.print(f"[yellow]Found {total_embeddings} embeddings to delete[/yellow]")
        
        if total_embeddings == 0:
            console.print("[green]No embeddings found - nothing to delete[/green]")
            return
        
        # Limpa todos os embeddings
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Clearing all embeddings...", total=None)
            
            # Executa query de limpeza usando conexão de banco do indexer
            query = f"DELETE FROM {cfg.RECO_SCHEMA}.trail_embeddings"
            with migrator.indexer.db.get_cursor() as cursor:
                cursor.execute(query)
                cursor.connection.commit()
            
            progress.update(task, description="All embeddings cleared!")
        
        console.print("[green]All embeddings have been cleared from the database![/green]")
        
        # Verifica exclusão
        new_status = migrator.get_migration_status()
        console.print(f"[blue]Remaining embeddings: {new_status['total_embeddings']}[/blue]")
    
    except Exception as e:
        console.print(f"[red]Clear all failed: {e}[/red]")
        sys.exit(1)
    
    finally:
        migrator.close()


if __name__ == "__main__":
    app()
