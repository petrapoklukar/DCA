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
from VGG16_utils import _analyze_query_point_assignment

app = typer.Typer()


@app.command()
def vgg16_qDCA(version_id: str, run_DCA: int = 1, run_qDCA: int = 1, cleanup: int = 1):
    repr_level = "feat_lin1"
    experiment_path = "output/vgg16_qDCA/"
    experiment_id = version_id

    # Set parameters
    path_to_dataset = f"representations/vgg16/{version_id}"
    path_to_Rfeatures = os.path.join(path_to_dataset, "sampled_Rfeatures.pkl")
    if os.path.isfile(path_to_Rfeatures):
        with open(path_to_Rfeatures, "rb") as f:
            Rdata = pickle.load(f)
    else:
        raise ValueError(f"Input file {path_to_Rfeatures} not found.")

    path_to_Efeatures = os.path.join(path_to_dataset, "sampled_Efeatures.pkl")
    if os.path.isfile(path_to_Efeatures):
        with open(path_to_Efeatures, "rb") as f:
            Edata = pickle.load(f)
    else:
        raise ValueError(f"Input file {path_to_Efeatures} not found.")

    R = Rdata[repr_level]
    E = Edata[repr_level]
    init_data_config = REData(R=R, E=E)

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

    output = []
    if run_DCA:
        dca = DCA(
            experiment_config,
            graph_config,
            hdbscan_config,
            geomCA_config,
            loggers=exp_loggers,
        )

        dca_scores = dca.fit(
            init_data_config
        )  # Do not call cleanup, output files are needed for qDCA
        output.append(dca_scores)

    if run_qDCA:
        path_to_Qfeatures = os.path.join(path_to_dataset, "query_features.pkl")
        if os.path.isfile(path_to_Qfeatures):
            with open(path_to_Qfeatures, "rb") as f:
                query_data = pickle.load(f)
        else:
            raise ValueError(f"Input file {path_to_Qfeatures} not found.")

        Q = query_data[repr_level]
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
            exp_loggers,
        )

        query_points_to_RE_assignment = dca.process_query_points(
            init_data_config, query_data_config, assign_to_RE=True
        )
        output.append(query_points_to_RE_assignment)

        accuracy = _analyze_query_point_assignment(
            query_data,
            Rdata,
            Edata,
            init_data_config.num_R,
            query_points_to_RE_assignment,
            experiment_config.DCA_dir,
        )
        print("Accuracy: %s", accuracy)
        output.append(accuracy)

        if cleanup:
            dca.cleanup()

    return output


if __name__ == "__main__":
    typer.run(vgg16_qDCA)
