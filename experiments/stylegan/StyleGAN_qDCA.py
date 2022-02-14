import os
import pickle
from dca.DCA import DCA
from dca.schemes import (
    DCALoggers,
    DelaunayGraphParams,
    ExperimentDirs,
    GeomCAParams,
    HDBSCANParams,
    QueryData,
    REData,
)
import typer

app = typer.Typer()


@app.command()
def stylegan_qDCA(run_DCA: int = 1, run_qDCA: int = 1, cleanup: int = 1):
    # Set parameters
    experiment_path = "output/stylegan_qDCA"
    experiment_id = "truncation10"

    # Load truncated representations
    path_to_dataset = f"representations/stylegan/"
    path_to_representations = os.path.join(path_to_dataset, "stylegan_qDCA_test.pkl")
    with open(path_to_representations, "rb") as f:
        data = pickle.load(f)
        R = data["Rref_features"]
        E = data["Eref_features"]

    init_data_config = REData(R=R, E=E)
    experiment_config = ExperimentDirs(
        experiment_dir=experiment_path,
        experiment_id=experiment_id,
    )
    graph_config = DelaunayGraphParams(
        sphere_coverage=0.7,
    )
    hdbscan_config = HDBSCANParams()
    geomCA_config = GeomCAParams()
    exp_loggers = DCALoggers(experiment_config.logs_dir)

    output = []
    if run_DCA:
        dca = DCA(
            experiment_config,
            graph_config,
            hdbscan_config,
            geomCA_config,
            loggers=exp_loggers,
        )

        dca_scores = dca.fit(init_data_config)
        output.append(dca_scores)

    if run_qDCA:
        Q = data["eval_features"]
        print("Query representations loaded.")
        query_data_config = QueryData(
            Q=Q,
            query_input_array_files_dir=os.path.join(experiment_id, "logs"),
            query_input_array_comp_assignment_filename=f"query_data_comp_assignment.npy",
        )

        dca = DCA(
            experiment_config,
            graph_config,
            hdbscan_config,
            geomCA_config,
            loggers=exp_loggers,
        )

        query_points_to_RE_assignment = dca.process_query_points(
            init_data_config, query_data_config, assign_to_RE=True, return_len=True
        )
        output.append(query_points_to_RE_assignment)

        if cleanup:
            dca.cleanup()

    return output


if __name__ == "__main__":
    typer.run(stylegan_qDCA)
