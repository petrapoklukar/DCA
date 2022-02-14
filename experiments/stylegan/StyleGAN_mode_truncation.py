import os
import pickle
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
def stylegan_mode_truncation(
    truncation: float, num_samples: str = "", cleanup: int = 1
):
    # Set parameters
    experiment_path = f"output/stylegan"
    tr = str(truncation).replace(".", "")
    experiment_id = f"truncation{tr}"

    # Load truncated representations
    path_to_dataset = "representations/stylegan/"
    path_to_representations = os.path.join(
        path_to_dataset,
        "stylegan_truncation{0:.1f}_{1}representations.pkl".format(
            truncation, num_samples
        ),
    )
    with open(path_to_representations, "rb") as f:
        data = pickle.load(f)
        R = data["ref_features"]
        E = data["eval_features"]

    data_config = REData(R=R, E=E, input_array_dir=os.path.join(experiment_id, "logs"))

    experiment_config = ExperimentDirs(
        experiment_dir=experiment_path,
        experiment_id=experiment_id,
        precomputed_folder=os.path.join(experiment_id, "logs"),
    )

    graph_config = DelaunayGraphParams(
        T=5000,
        unfiltered_edges_dir=os.path.join(experiment_id, "logs"),
        filtered_edges_dir=os.path.join(experiment_id, "logs"),
        sphere_coverage=0.7,
    )
    hdbscan_config = HDBSCANParams(clusterer_dir=os.path.join(experiment_id, "logs"))
    geomCA_config = GeomCAParams()
    exp_loggers = DCALoggers(experiment_config.logs_dir)

    dca = DCA(
        experiment_config,
        graph_config,
        hdbscan_config,
        geomCA_config,
        loggers=exp_loggers,
    )

    dca_scores = dca.fit(data_config)
    if cleanup:
        dca.cleanup()

    return dca_scores


if __name__ == "__main__":
    typer.run(stylegan_mode_truncation)
