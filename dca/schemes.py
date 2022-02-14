from dataclasses import dataclass, field
from pydantic import validator
from typing import Optional, List, Tuple
import numpy as np
import os
import json
import networkx as nx
import umap


@dataclass
class ExperimentDirs:
    experiment_dir: str  # path to experiment directory
    experiment_id: str  # experiment name for given hyperparams
    precomputed_folder: str = (
        "precomputed"  # name of the folder where precomputed files are stored
    )
    precomputed_dir: str = field(init=False)
    DCA_dir: str = field(init=False)
    visualization_dir: str = field(init=False)
    results_dir: str = field(init=False)
    logs_dir: str = field(init=False)

    def __post_init__(self):
        self.precomputed_dir = os.path.join(
            self.experiment_dir, self.precomputed_folder
        )
        self.DCA_dir = os.path.join(self.experiment_dir, self.experiment_id)
        self.visualization_dir = os.path.join(self.DCA_dir, "visualization")
        self.results_dir = os.path.join(self.DCA_dir, "DCA")
        self.logs_dir = os.path.join(self.DCA_dir, "logs")

        if not os.path.exists(self.precomputed_dir):
            os.makedirs(self.precomputed_dir)
        if not os.path.exists(self.DCA_dir):
            os.makedirs(self.DCA_dir)
        if not os.path.exists(self.visualization_dir):
            os.makedirs(self.visualization_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

    def __iter__(self):
        for key in self.__dict__:
            value = getattr(self, key)
            if not isinstance(value, np.ndarray):
                yield key, value


@dataclass
class REData:
    R: np.ndarray
    E: np.ndarray
    input_array_dir: str = "precomputed"
    input_array_filename: str = "input_array.npy"
    input_array_filepath: str = field(init=False)
    num_E: int = field(init=False)
    num_R: int = field(init=False)
    visualize: bool = field(init=False)

    def __post_init__(self):
        self.num_R = self.R.shape[0]
        self.num_E = self.E.shape[0]
        self.visualize = (
            False if (self.num_R + self.num_E > 3000 or self.R.shape[-1] > 20) else True
        )

        self.input_array_filepath = os.path.join(
            self.input_array_dir, self.input_array_filename
        )

    @validator("R", "E")
    def check_shape(self, array):
        if array.dtype == np.float32:
            if len(array.shape) == 2:
                return array
            else:
                raise ValueError(
                    f"R and E expected to be 2D arrays but got {len(array.shape)}D."
                )
        else:
            raise ValueError(f"Input data not of dtype np.float32.")


@dataclass
class QueryData:
    Q: np.ndarray
    num_Q: int = field(init=False)

    query_input_array_files_dir: str
    query_input_array_filename: str = "query_input_array.npy"
    query_input_array_filepath: str = field(init=False)

    query_input_array_comp_assignment_filename: str = (
        "query_input_array_comp_assignment.npy"
    )
    query_input_array_point_assignment_filepath: str = field(init=False)

    query_input_array_considered_comp_list_filename: str = (
        "query_input_array_considered_comp_list.npy"
    )
    query_input_array_considered_comp_list_filepath: str = field(init=False)

    query_input_array_point_assignment_filename: str = (
        "query_input_array_point_assignment.npy"
    )
    query_input_array_point_assignment_filepath: str = field(init=False)

    query_input_array_edges_stats_filename: str = "query_input_array_edges_stats.pkl"
    query_input_array_edges_stats_filepath: str = field(init=False)

    def __post_init__(self):
        self.num_Q = self.Q.shape[0]
        self.query_input_array_filepath = os.path.join(
            self.query_input_array_files_dir, self.query_input_array_filename
        )
        self.query_input_array_point_assignment_filepath = os.path.join(
            self.query_input_array_files_dir,
            self.query_input_array_comp_assignment_filename,
        )
        self.query_input_array_point_assignment_filepath = os.path.join(
            self.query_input_array_files_dir,
            self.query_input_array_point_assignment_filename,
        )

    @validator("Q")
    def check_shape(self, array):
        if array.dtype == np.float32:
            if len(array.shape) == 2:
                return array
            else:
                raise ValueError(
                    f"Query data expected to be a 2D array but got {len(array.shape)}D."
                )
        else:
            raise ValueError(f"Query data not of dtype np.float32.")


@dataclass
class DelaunayGraphParams:
    T: int = 10**4

    unfiltered_edges_dir: str = "precomputed"

    unfiltered_edges_filename: str = "unfiltered_edges.npy"
    unfiltered_edges_filepath: str = field(init=False)
    unfiltered_edges_len_filename: str = "unfiltered_edges_len.npy"
    unfiltered_edges_len_filepath: str = field(init=False)

    filtered_edges_dir: str = "precomputed"
    filtered_edges_filename: str = "filtered_edges.npy"
    filtered_edges_filepath: str = field(init=False)
    filtered_edges_len_filename: str = "filtered_edges_len.npy"
    filtered_edges_len_filepath: str = field(init=False)

    query_unfiltered_edges_filename: str = "query_unfiltered_edges.npy"
    query_unfiltered_edges_len_filename: str = "query_unfiltered_edges_len.npy"
    query_filtered_edges_filename: str = "query_filtered_edges.npy"
    query_filtered_edges_len_filename: str = "query_filtered_edges_len.npy"
    sphere_coverage: float = 1.0

    def __post_init__(self):
        self.unfiltered_edges_filepath = os.path.join(
            self.unfiltered_edges_dir,
            self.unfiltered_edges_filename,
        )
        self.unfiltered_edges_len_filepath = os.path.join(
            self.unfiltered_edges_dir,
            self.unfiltered_edges_len_filename,
        )
        self.filtered_edges_filepath = os.path.join(
            self.filtered_edges_dir,
            self.filtered_edges_filename,
        )
        self.filtered_edges_len_filepath = os.path.join(
            self.filtered_edges_dir,
            self.filtered_edges_len_filename,
        )

    def __iter__(self):
        for key in self.__dict__:
            yield key, getattr(self, key)


@dataclass
class HDBSCANParams:
    min_cluster_size: int = 10
    min_samples: int = 1
    cluster_selection_epsilon: Optional[float] = 0
    clusterer_dir: str = "precomputed"
    clusterer_filename: str = "clusterer.pkl"
    clusterer_filepath: str = field(init=False)
    input_array_labels_filename: str = "input_array_comp_labels.npy"
    input_array_labels_filepath: str = field(init=False)

    def __post_init__(self):
        self.clusterer_filepath = os.path.join(
            self.clusterer_dir,
            self.clusterer_filename,
        )
        self.input_array_labels_filepath = os.path.join(
            self.clusterer_dir,
            self.input_array_labels_filename,
        )

    def __iter__(self):
        for key in self.__dict__:
            yield key, getattr(self, key)


@dataclass
class GeomCAParams:
    comp_consistency_threshold: float = 0.0
    comp_quality_threshold: float = 0.0

    def __iter__(self):
        for key in self.__dict__:
            yield key, getattr(self, key)


@dataclass
class DCALoggers:
    experiment_dir: str
    loggers: dict = field(init=False)
    random_seed: int = 1111

    override: bool = False
    version: int = 0
    qdca_flag: bool = False

    def __post_init__(self):
        existing_logs = [
            f for f in os.listdir(self.experiment_dir) if f.endswith(".log")
        ]
        if not self.override:
            self.version = len(existing_logs) // 2
        qdca_flag = "_qdca" if self.qdca_flag != 0 else ""

        self.loggers = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "time_only": {
                    "format": "[%(asctime)s] :: %(message)s",
                    "datefmt": "%m/%d/%Y %H:%M:%S",
                },
                "simple": {
                    "format": "[%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s - line %(lineno)d] :: %(message)s",
                    "datefmt": "%m/%d/%Y %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                },
                "DCA_stats": {
                    "class": "logging.FileHandler",
                    "level": "DEBUG",
                    "formatter": "simple",
                    "filename": os.path.join(
                        self.experiment_dir,
                        f"version{self.version}{qdca_flag}_experiment_info.log",
                    ),
                    "mode": "w",
                },
                "DCA_results": {
                    "class": "logging.FileHandler",
                    "level": "DEBUG",
                    "formatter": "time_only",
                    "filename": os.path.join(
                        self.experiment_dir,
                        f"version{self.version}{qdca_flag}_output_formatted.log",
                    ),
                    "mode": "w",
                },
                "DCA_elapsed_time": {
                    "class": "logging.FileHandler",
                    "level": "DEBUG",
                    "formatter": "time_only",
                    "filename": os.path.join(
                        self.experiment_dir,
                        f"version{self.version}{qdca_flag}_elapsed_time.log",
                    ),
                    "mode": "w",
                },
            },
            "loggers": {
                "experiment_logger": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": "no",
                },
                "DCA_info_logger": {
                    "level": "DEBUG",
                    "handlers": ["DCA_stats"],
                    "propagate": "no",
                },
                "DCA_result_logger": {
                    "level": "DEBUG",
                    "handlers": ["DCA_results"],
                    "propagate": "no",
                },
                "DCA_time_logger": {
                    "level": "DEBUG",
                    "handlers": ["DCA_elapsed_time"],
                    "propagate": "no",
                },
            },
            "root": {"level": "INFO", "handlers": ["console"]},
        }


@dataclass
class DelaunayGraphComponentStats:
    Ridx: np.ndarray
    Eidx: np.ndarray
    comp_consistency: float
    comp_quality: float
    num_comp_RE_edges: int
    num_total_comp_edges: int
    comp_graph: Optional[nx.Graph] = None
    mean_edge_len: Optional[float] = None
    std_edge_len: Optional[float] = None


@dataclass
class DelaunayGraphNetworkStats:
    num_R: int
    num_E: int
    precision: Optional[float] = None
    recall: Optional[float] = None
    network_consistency: Optional[float] = None
    network_quality: Optional[float] = None
    first_trivial_component_idx: Optional[int] = None
    num_R_points_in_fundcomp: int = 0
    num_E_points_in_fundcomp: int = 0
    num_RE_edges: int = 0
    num_total_edges: int = 0
    num_R_outliers: int = 0
    num_E_outliers: int = 0
    num_fundcomp: int = 0
    num_comp: int = 0
    num_outliercomp: int = 0


@dataclass
class DelaunayGraphVisualizer:
    input_array_filepath: str
    num_R: int
    num_E: int
    input_array_comp_labels_filepath: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    RE_2D_projection: Optional[umap.UMAP] = None
    proj_RE: np.ndarray = field(init=False)

    def __post_init__(self):
        input_data = np.load(self.input_array_filepath)
        if input_data.shape[1] > 2:
            self.RE_2D_projection = umap.UMAP(n_components=2)
            self.RE_2D_projection.fit(input_data)
            self.proj_RE = self.RE_2D_projection.transform(input_data)
        else:
            self.proj_RE = input_data

    def get_input_array_data(self):
        return self.proj_RE
