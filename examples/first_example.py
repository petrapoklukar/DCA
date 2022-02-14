import numpy as np
from dca.DCA import DCA
from dca.schemes import (
    DCALoggers,
    DelaunayGraphParams,
    ExperimentDirs,
    GeomCAParams,
    HDBSCANParams,
    REData,
)
import typer


app = typer.Typer()


@app.command()
def first_example(cleanup: int = 0):
    # Set path to the output folders
    experiment_path = "output/first_example"
    experiment_id = "template_id1"

    # Load data
    R = np.load("representations/first_example/R.npy")
    E = np.load("representations/first_example/E.npy")

    # Generate input parameters
    data_config = REData(R=R, E=E)
    experiment_config = ExperimentDirs(
        experiment_dir=experiment_path,
        experiment_id=experiment_id,
    )
    graph_config = DelaunayGraphParams()
    hdbscan_config = HDBSCANParams()
    geomCA_config = GeomCAParams()

    # Initialize loggers
    exp_loggers = DCALoggers(experiment_config.logs_dir)

    # Run DCA
    dca = DCA(
        experiment_config,
        graph_config,
        hdbscan_config,
        geomCA_config,
        loggers=exp_loggers,
    )
    dca_scores = dca.fit(data_config)

    if cleanup:
        dca.cleanup()  # Optional cleanup

    return dca_scores


if __name__ == "__main__":
    typer.run(first_example)
