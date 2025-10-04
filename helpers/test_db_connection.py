#!/usr/bin/env python3
"""
Script de teste para verificar conexão com PostgreSQL + pgvector.

Uso:
    python test_db_connection.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from reco.config import RecoConfig
from reco.database.connection import get_connection
from rich.console import Console
from rich.table import Table

console = Console()


def test_connection():
    """Test database connection and basic operations."""
    console.print("[bold blue]Testing PostgreSQL + pgvector connection...[/bold blue]")
    
    try:
        # Override host for local testing (app local + DB Docker)
        import os
        original_host = os.environ.get('POSTGRES_HOST')
        os.environ['POSTGRES_HOST'] = 'localhost'  # Docker mapeia para localhost:5432
        
        # Load configuration
        cfg = RecoConfig()
        
        # Restore original host
        if original_host:
            os.environ['POSTGRES_HOST'] = original_host
        # Mask password in URL for security
        masked_url = cfg.DATABASE_URL
        if '@' in masked_url and ':' in masked_url.split('@')[0]:
            user_pass, rest = masked_url.split('@', 1)
            if ':' in user_pass:
                user, _ = user_pass.split(':', 1)
                masked_url = f"{user}:***@{rest}"
        
        console.print(f"Database URL: {masked_url}")
        console.print(f"Schema: {cfg.RECO_SCHEMA}")
        
        # Test connection
        db = get_connection(cfg)
        console.print("[green]✓ Connection established[/green]")
        
        # Test basic query
        with db.get_cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            console.print(f"[green]✓ PostgreSQL version: {version[0]}[/green]")
        
        # Test pgvector extension
        with db.get_cursor() as cursor:
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            vector_ext = cursor.fetchone()
            if vector_ext:
                console.print("[green]✓ pgvector extension found[/green]")
            else:
                console.print("[red]✗ pgvector extension not found[/red]")
                return False
        
        # Test schema existence
        with db.get_cursor() as cursor:
            cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'reco';")
            reco_schema = cursor.fetchone()
            if reco_schema:
                console.print("[green]✓ Schema 'reco' exists[/green]")
            else:
                console.print("[yellow]Schema 'reco' not found - run migration first[/yellow]")
        
        # Test table existence
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'reco' AND table_name = 'trail_embeddings';
            """)
            table = cursor.fetchone()
            if table:
                console.print("[green]✓ Table 'reco.trail_embeddings' exists[/green]")
            else:
                console.print("[yellow]Table 'reco.trail_embeddings' not found - run migration first[/yellow]")
        
        # Test vector operations
        try:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT '[1,2,3]'::vector;")
                result = cursor.fetchone()
                console.print("[green]✓ Vector operations working[/green]")
        except Exception as e:
            console.print(f"[red]✗ Vector operations failed: {e}[/red]")
            return False
        
        # Get database stats
        stats = db.get_embedding_stats()
        
        # Print stats table
        table = Table(title="Database Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Embeddings", str(stats.get('total_embeddings', 0)))
        table.add_row("Model Versions", ", ".join(stats.get('model_versions', [])))
        table.add_row("Last Updated", str(stats.get('last_updated', 'Never')))
        
        console.print(table)
        
        console.print("[bold green]✓ All tests passed![/bold green]")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Connection failed: {e}[/red]")
        return False
    
    finally:
        try:
            db.close()
        except:
            pass


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
