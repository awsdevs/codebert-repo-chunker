#!/usr/bin/env python
"""Main script to process a repository"""
import click
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.master_pipeline import MasterPipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

@click.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output directory', default='output')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(repo_path: str, output: str, verbose: bool):
    """Process a repository and create chunks"""
    
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        click.echo(f"Processing repository: {repo_path}")
        
        # Initialize pipeline
        pipeline = MasterPipeline()
        
        # Process repository
        result = pipeline.process_repository(Path(repo_path))
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("Processing Complete!")
        click.echo("="*60)
        click.echo(f"Files processed: {result['statistics']['files_processed']}")
        click.echo(f"Files skipped: {result['statistics']['files_skipped']}")
        click.echo(f"Chunks created: {result['statistics']['chunks_created']}")
        click.echo(f"Processing time: {result['report']['processing_time']}")
        
        if result['statistics']['errors']:
            click.echo(f"\nErrors encountered: {len(result['statistics']['errors'])}")
            for error in result['statistics']['errors'][:5]:
                click.echo(f"  - {error['file']}: {error['error']}")
        
        click.echo(f"\nReport saved to output/reports/")
        
    except Exception as e:
        logger.error(f"Failed to process repository: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()