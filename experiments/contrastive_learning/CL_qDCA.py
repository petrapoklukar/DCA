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
import CL_utils as CL_utils

app = typer.Typer()


@app.command()
def CL_qDCA(
    initial_dataset_filename: str,
    query_dataset_filename: str,
    run_DCA: int = 0,
    run_qDCA: int = 1,
    several_assignments: int = 0,
    cleanup: int = 1,
):
    experiment_path = f"output/CL_qDCA/{initial_dataset_filename}"
    experiment_id = "initial"

    path_to_dataset = (
        f"representations/contrastive_learning/{initial_dataset_filename}_train.pkl"
    )
    with open(os.path.join(path_to_dataset), "rb") as f:
        Rdata = pickle.load(f)
        Rrepresentations, Rlabels = Rdata["R"], Rdata["class_labels"]
        R = CL_utils.get_representations_by_class(
            Rrepresentations, Rlabels, [0, 1, 2, 3, 4, 5, 6]
        )

    path_to_dataset = f"representations/contrastive_learning/remaining_{initial_dataset_filename}_holdout.pkl"
    with open(path_to_dataset, "rb") as f:
        dataE = pickle.load(f)
        Erepresentations, Elabels = dataE["E"], dataE["class_labels"]
        E = CL_utils.get_representations_by_class(
            Erepresentations, Elabels, [0, 1, 2, 3, 4, 5, 6]
        )

    init_data_config = REData(R=R, E=E)
    experiment_config = ExperimentDirs(
        experiment_dir=experiment_path,
        experiment_id=experiment_id,
    )
    graph_config = DelaunayGraphParams()
    hdbscan_config = HDBSCANParams()
    geomCA_config = GeomCAParams(
        comp_consistency_threshold=0.75, comp_quality_threshold=0.45
    )
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
        print("Starting to run qDCA...")
        path_to_dataset = (
            f"representations/contrastive_learning/{query_dataset_filename}.pkl"
        )
        with open(path_to_dataset, "rb") as f:
            query_data = pickle.load(f)
            Q = query_data["Q"]

        if several_assignments:
            query_dataset_filename = query_dataset_filename + "_several_assignments"

        query_data_config = QueryData(
            Q=Q,
            query_input_array_filename=f"query_input_array_{query_dataset_filename}.npy",
            query_input_array_files_dir=os.path.join(experiment_id, "logs"),
            query_input_array_comp_assignment_filename=f"{query_dataset_filename}_comp_assignment.npy",
            query_input_array_considered_comp_list_filename=f"{query_dataset_filename}_considered_comp_list.npy",
            query_input_array_edges_stats_filename=f"{query_dataset_filename}_input_array_edges_stats.npy",
        )

        graph_config.query_unfiltered_edges_filename = (
            f"query_unfiltered_edges_{query_dataset_filename}.npy"
        )
        graph_config.query_unfiltered_edges_len_filename = (
            f"query_unfiltered_edges_len_{query_dataset_filename}.npy"
        )
        graph_config.query_filtered_edges_filename = (
            f"query_filtered_edges_{query_dataset_filename}.npy"
        )
        graph_config.query_filtered_edges_len_filename = (
            f"query_filtered_edges_len_{query_dataset_filename}.npy"
        )

        dca = DCA(
            experiment_config,
            graph_config,
            hdbscan_config,
            geomCA_config,
            exp_loggers,
        )

        (
            query_points_comp_assignment,
            considered_comp_idx_list,
        ) = dca.process_query_points(
            init_data_config,
            query_data_config,
            assign_to_comp=True,
            consider_several_assignments=bool(several_assignments),
        )
        output += [query_points_comp_assignment, considered_comp_idx_list]

        CL_utils.plot_query_points(
            query_points_comp_assignment,
            considered_comp_idx_list,
            query_data,
            query_dataset_filename,
            initial_dataset_filename,
            os.path.join(
                experiment_config.experiment_dir,
                hdbscan_config.input_array_labels_filepath,
            ),
            dca.results_dir,
            dca.DCA_dir,
        )
        if cleanup:
            dca.cleanup()

    return output


if __name__ == "__main__":
    typer.run(CL_qDCA)
