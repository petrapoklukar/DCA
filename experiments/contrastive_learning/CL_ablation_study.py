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
def CL_ablation_delaunay_edge_approximation(cleanup: int = 1):
    experiment_path = f"output/CL_ablation_study/delaunay_edge_approximation"

    output = []
    for T in [10**1, 10**2, 10**3, 10**4, 10**5, 10**6]:
        dataset_path = "representations/contrastive_learning/"
        with open(os.path.join(dataset_path, "balanced_Df_train.pkl"), "rb") as f:
            Rdata = pickle.load(f)
            R = Rdata["R"]

        with open(os.path.join(dataset_path, "balanced_Df_holdout.pkl"), "rb") as f:
            Edata = pickle.load(f)
            E = Edata["E"]

        data_config = REData(R=R, E=E)

        experiment_id = f"T{T}"
        experiment_config = ExperimentDirs(
            experiment_dir=experiment_path,
            experiment_id=experiment_id,
        )

        graph_config = DelaunayGraphParams(
            T=T,
            unfiltered_edges_dir=os.path.join(experiment_id, "logs"),
            filtered_edges_dir=os.path.join(experiment_id, "logs"),
        )
        hdbscan_config = HDBSCANParams(
            clusterer_dir=os.path.join(experiment_id, "logs")
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

    return output


@app.command()
def CL_ablation_delaunay_edge_filtering(cleanup: int = 1):
    experiment_path = f"output/CL_ablation_study/delaunay_edge_filtering"

    output = []
    for sphere_coverage in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        dataset_path = "representations/contrastive_learning/"
        with open(os.path.join(dataset_path, "balanced_Df_train.pkl"), "rb") as f:
            Rdata = pickle.load(f)
            R = Rdata["R"]

        with open(os.path.join(dataset_path, "balanced_Df_holdout.pkl"), "rb") as f:
            Edata = pickle.load(f)
            E = Edata["E"]
        data_config = REData(R=R, E=E)

        experiment_id = f"coverage{sphere_coverage}"
        experiment_config = ExperimentDirs(
            experiment_dir=experiment_path,
            experiment_id=experiment_id,
        )

        graph_config = DelaunayGraphParams(
            filtered_edges_dir=os.path.join(experiment_id, "logs"),
            sphere_coverage=sphere_coverage,
        )
        hdbscan_config = HDBSCANParams(
            clusterer_dir=os.path.join(experiment_id, "logs")
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

    return output


@app.command()
def CL_ablation_hdbscan(cleanup: int = 1):
    experiment_path = f"output/CL_ablation_study/hdbscan"

    output = []
    for min_cluster_size in [3, 5, 10, 20]:
        experiment_id = f"mcs{min_cluster_size}"
        dataset_path = "representations/contrastive_learning/"
        with open(os.path.join(dataset_path, "balanced_Df_train.pkl"), "rb") as f:
            Rdata = pickle.load(f)
            R = Rdata["R"]

        with open(os.path.join(dataset_path, "balanced_Df_holdout.pkl"), "rb") as f:
            Edata = pickle.load(f)
            E = Edata["E"]
        data_config = REData(R=R, E=E)

        experiment_config = ExperimentDirs(
            experiment_dir=experiment_path,
            experiment_id=experiment_id,
        )
        graph_config = DelaunayGraphParams()
        hdbscan_config = HDBSCANParams(
            min_cluster_size=min_cluster_size,
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

    return output


if __name__ == "__main__":
    app()
