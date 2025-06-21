# weakly/cli/main.py
import click
from pathlib import Path
from typing import Optional

from ..config.config_manager import ConfigManager
from ..core.experiment import Experiment
from ..core.output_manager import OutputManager


@click.command()
@click.option(
    "--config", 
    "-c", 
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to config.cfg file"
)
@click.option(
    "--grid-search", 
    is_flag=True, 
    help="Run grid search instead of single experiment"
)
@click.option(
    "--output-dir", 
    "-o", 
    type=click.Path(path_type=Path),
    default="output",
    help="Base output directory (default: output)"
)
@click.option(
    "--verbose", 
    "-v", 
    is_flag=True, 
    help="Enable verbose logging"
)
def main(
    config: Path, 
    grid_search: bool, 
    output_dir: Path, 
    verbose: bool
) -> None:
    """Weakly - Run experiments with weakly supervision learning."""
    try:
        # Load configuration
        config_data = ConfigManager.load_config(str(config))
        
        # Setup output manager
        output_manager = OutputManager(base_dir=output_dir)
        
        # Create and run experiment
        experiment = Experiment(config_data, output_manager, verbose=verbose)
        
        if grid_search:
            click.echo("🔍 Starting grid search...")
            results = experiment.run_grid_search()
        else:
            click.echo("🚀 Starting single experiment...")
            results = experiment.run()
            
        click.echo(f"✅ Experiment completed! Results saved to: {experiment.output_path}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
