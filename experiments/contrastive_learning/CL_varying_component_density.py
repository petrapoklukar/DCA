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
def CL_varying_component_density(
    n_iterations: int = 10, perc_to_discard: float = 0.50, cleanup: int = 1
):
    p_discard_str = str(perc_to_discard).replace(".", "")
    experiment_path = f"output/CL_varying_component_density/discardperc{p_discard_str}"
    dataset_path = "representations/contrastive_learning/"

    # Load representations
    with open(os.path.join(dataset_path, "Df_train.pkl"), "rb") as f:
        Rdata = pickle.load(f)
        Rrepresentations, Rlabels = Rdata["R"], Rdata["class_labels"]

    with open(os.path.join(dataset_path, "Df_holdout.pkl"), "rb") as f:
        Edata = pickle.load(f)
        Erepresentations, Elabels = Edata["E"], Edata["class_labels"]

    E_idxs = CL_utils.get_representation_idxs_by_class(Elabels, [0, 1, 2, 3, 4, 5, 6])
    E = Erepresentations[E_idxs]

    iter = 0
    classes_to_prune = [
        np.random.choice(np.arange(7), size=3, replace=False)
        for i in range(n_iterations)
    ]
    output = []
    print("All pruning combinations:", classes_to_prune)
    for c_to_prune in classes_to_prune:
        print("Prunning", c_to_prune)
        R_idxs, R_classes = CL_utils.get_pruned_representation_idxs_by_class(
            Rlabels,
            [0, 1, 2, 3, 4, 5, 6],
            c_to_prune,
            perc_to_discard,
        )
        R = Rrepresentations[R_idxs]

        experiment_id = f"iter{iter}"
        print("Current iteration: {0}".format(iter))

        data_config = REData(
            R=R, E=E, input_array_dir=os.path.join(experiment_id, "logs")
        )
        data_config.visualize = False
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
        iter += 1
    return output


if __name__ == "__main__":
    typer.run(CL_varying_component_density)
