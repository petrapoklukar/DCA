import os
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
import CL_utils
import numpy as np
import pickle

app = typer.Typer()


@app.command()
def CL_mode_truncation(cleanup: int = 1):
    experiment_path = f"output/CL_mode_truncation/"
    dataset_path = "representations/contrastive_learning/"

    # Load representations
    with open(os.path.join(dataset_path, "Df_train.pkl"), "rb") as f:
        Rdata = pickle.load(f)
        Rrepresentations, Rlabels = Rdata["R"], Rdata["class_labels"]

    with open(os.path.join(dataset_path, "Df_holdout.pkl"), "rb") as f:
        Edata = pickle.load(f)
        Erepresentations, Elabels = Edata["E"], Edata["class_labels"]

    # Extract initial R and E splits
    Eclasses = [0]
    n_classes = 12
    R = CL_utils.get_representations_by_class(
        Rrepresentations, Rlabels, [0, 1, 2, 3, 4, 5, 6]
    )
    E = CL_utils.get_representations_by_class(Erepresentations, Elabels, Eclasses)

    output = []
    while len(Eclasses) <= n_classes:
        experiment_id = f"n_Eclasses{len(Eclasses)}"
        print("Current Eclasses: {0}".format(Eclasses))

        data_config = REData(
            R=R, E=E, input_array_dir=os.path.join(experiment_id, "logs")
        )
        experiment_config = ExperimentDirs(
            experiment_dir=experiment_path,
            experiment_id=experiment_id,
            precomputed_folder=os.path.join(experiment_id, "logs"),
        )

        graph_config = DelaunayGraphParams(
            unfiltered_edges_dir=os.path.join(experiment_id, "logs"),
            filtered_edges_dir=os.path.join(experiment_id, "logs"),
        )
        hdbscan_config = HDBSCANParams(
            clusterer_dir=os.path.join(experiment_id, "logs"),
        )
        geomCA_config = GeomCAParams(
            comp_consistency_threshold=0.75, comp_quality_threshold=0.45
        )
        exp_loggers = DCALoggers(experiment_config.logs_dir)
        dca = DCA(
            experiment_config,
            graph_config,
            hdbscan_config,
            geomCA_config,
            loggers=exp_loggers,
        )

        dca_scores = dca.fit(data_config)
        output.append(dca_scores)

        if cleanup:
            dca.cleanup()

        # Add a class to E and obtain the data
        new_class = Eclasses[-1] + 1
        new_class_representations = CL_utils.get_representations_by_class(
            Erepresentations, Elabels, [new_class]
        )
        Eclasses.append(Eclasses[-1] + 1)
        E = np.concatenate([E, new_class_representations])
        print("E set updated with class {0}".format(new_class))

    return output


if __name__ == "__main__":
    typer.run(CL_mode_truncation)
