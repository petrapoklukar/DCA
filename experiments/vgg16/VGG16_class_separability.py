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
def vgg16_class_separability(version_id: str, cleanup: int = 1):
    # Set parameters
    repr_level = "feat_lin1"
    experiment_path = "output/vgg16/"
    experiment_id = version_id

    path_to_dataset = f"representations/vgg16/{version_id}"
    path_to_Rfeatures = os.path.join(path_to_dataset, "Rfeatures.pkl")
    if os.path.isfile(path_to_Rfeatures):
        with open(path_to_Rfeatures, "rb") as f:
            Rdata = pickle.load(f)
    else:
        raise ValueError(f"Input file {path_to_Rfeatures} not found.")

    path_to_Efeatures = os.path.join(path_to_dataset, "Efeatures.pkl")
    if os.path.isfile(path_to_Efeatures):
        with open(path_to_Efeatures, "rb") as f:
            Edata = pickle.load(f)
    else:
        raise ValueError(f"Input file {path_to_Efeatures} not found.")

    R = Rdata[repr_level]
    E = Edata[repr_level]
    data_config = REData(R=R, E=E)

    experiment_config = ExperimentDirs(
        experiment_dir=experiment_path,
        experiment_id=experiment_id,
    )
    graph_config = DelaunayGraphParams(
        filtered_edges_dir=os.path.join(experiment_id, "logs"),
    )
    hdbscan_config = HDBSCANParams(
        clusterer_dir=os.path.join(experiment_id, "logs"),
    )
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
    typer.run(vgg16_class_separability)
