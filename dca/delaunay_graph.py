import time
import networkx as nx
import numpy as np
import os
import logging
import pickle
from dca.schemes import (
    DelaunayGraphComponentStats,
    DelaunayGraphNetworkStats,
)
from functools import wraps
from typing import List, Optional
import pandas as pd
import dca.delaunay_graph_utils as graph_utils
import gc


# -------------------------------------------------------------------------- #
# Logging settings
# -------------------------------------------------------------------------- #
logger = logging.getLogger("DCA_info_logger")
result_logger = logging.getLogger("DCA_result_logger")


def logger_time(func):
    """Prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(
            "Function {0} executed in {1:f} s".format(func.__name__, end - start)
        )
        result_logger.debug(
            "Elapsed time: {0}-{1:f} s".format(func.__name__, end - start)
        )
        return result

    return wrapper


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
class DelaunayGraph:
    def __init__(self, num_R: int, num_E: int) -> None:
        self.num_R = num_R
        self.num_E = num_E
        self.graph: Optional[nx.Graph] = None
        self.first_trivial_component_idx: Optional[int] = None
        self.comp_stats: List[DelaunayGraphComponentStats] = []
        self.network_stats = DelaunayGraphNetworkStats(num_R=num_R, num_E=num_E)

    def init_Delaunay_graph(
        self, Delaunay_edges: np.ndarray, Delaunay_edges_len: np.ndarray
    ):
        """
        Initializes a Delaunay graph.
        :param Delaunay_edges: array of edges.
        :param Delaunay_edges_len: array of edge lengths.
        """
        self.graph = nx.Graph()
        Delaunay_edge_list = np.concatenate(
            [
                Delaunay_edges[:, 0][:, None],
                Delaunay_edges[:, 1][:, None],
                Delaunay_edges_len.astype(np.float64),
            ],
            axis=1,
        )
        del Delaunay_edges
        del Delaunay_edges_len
        gc.collect()

        self.graph.add_weighted_edges_from(Delaunay_edge_list)

        if nx.number_connected_components(self.graph) > 1:
            raise NotImplementedError(
                "Delaney graph not connected. Perform DelaunayGeomCA on each components separately or increase sphere coverage parameter."
            )

    def load_existing(self, stats_dir: str):
        """
        Loads an existing Delaunay graph.
        :param stats_dir: directory containing results of global and local evaluations.
        """
        with open(os.path.join(stats_dir, "components_stats.pkl"), "rb") as f:
            self.comp_stats = pickle.load(f)

        with open(os.path.join(stats_dir, "network_stats.pkl"), "rb") as f:
            self.network_stats = pickle.load(f)
        self.first_trivial_component_idx = (
            self.network_stats.first_trivial_component_idx
        )

    def distil_edges(self):
        """
        Distils edges from the approximated Delaunay graph.
        :return: list of distilled edges.
        """
        distilled_edges = []
        for idx in range(len(self.comp_stats)):
            distilled_edges += list(self.comp_stats[idx].comp_graph.edges())
        return distilled_edges

    def set_first_trivial_component_idx(self, idx: int):
        """
        Sets the index of the first trivial component.
        :param idx: index of the first trivial components.
        """
        self.network_stats.first_trivial_component_idx = idx
        self.first_trivial_component_idx = idx

    def update_local_stats(
        self,
        comp_R_idxs: np.ndarray,
        comp_E_idxs: np.ndarray,
        comp_consistency: float,
        comp_quality: float,
        num_comp_RE_edges: int,
        num_total_comp_edges: int,
        comp_consistency_threshold: float,
        comp_quality_threshold: float,
        subgraph_RE_comp: Optional[nx.Graph] = None,
    ):
        """
        Updates local evaluation scores with results of the given connected component.
        :param comp_R_idxs: indices of R points in the given component.
        :param comp_E_idxs: indices of E points in the given component.
        :param comp_consistency: component consistency score.
        :param comp_quality: component quality score.
        :param num_comp_RE_edges: number of heterogenous edges in the given component.
        :param num_total_comp_edges: total number of edges in the given component.
        :param comp_consistency_threshold: chosen component consistency eta_c threshold.
        :param comp_quality_threshold: chosen component quality eta_q threshold.
        :param subgraph_RE_comp: graph representing the given component (if given).
        """

        self.comp_stats.append(
            DelaunayGraphComponentStats(
                Ridx=comp_R_idxs,
                Eidx=comp_E_idxs,
                comp_consistency=comp_consistency,
                comp_quality=comp_quality,
                comp_graph=subgraph_RE_comp,
                num_comp_RE_edges=num_comp_RE_edges,
                num_total_comp_edges=num_total_comp_edges,
            )
        )

        self.network_stats.num_comp += 1
        self.network_stats.num_RE_edges += num_comp_RE_edges
        self.network_stats.num_total_edges += num_total_comp_edges

        # Part of global evaluation as in Definition 3.3.
        if (
            comp_consistency > comp_consistency_threshold
            and comp_quality > comp_quality_threshold
        ):
            self.network_stats.num_R_points_in_fundcomp += len(comp_R_idxs)
            self.network_stats.num_E_points_in_fundcomp += len(comp_E_idxs)
            self.network_stats.num_fundcomp += 1

        # Monitor the outliers
        if len(comp_E_idxs) == 0 and len(comp_R_idxs) == 1:
            self.network_stats.num_R_outliers += 1
            self.network_stats.num_outliercomp += 1

        if len(comp_R_idxs) == 0 and len(comp_E_idxs) == 1:
            self.network_stats.num_E_outliers += 1
            self.network_stats.num_outliercomp += 1

    def update_global_stats(self):
        """
        Updates global evaluation scores.
        """
        precision = self.network_stats.num_E_points_in_fundcomp / self.num_E
        recall = self.network_stats.num_R_points_in_fundcomp / self.num_R

        network_consistency = graph_utils._get_graph_consistency(self.num_R, self.num_E)
        network_quality = (
            self.network_stats.num_RE_edges / self.network_stats.num_total_edges
            if self.network_stats.num_total_edges != 0
            else 0
        )

        self.network_stats.precision = precision
        self.network_stats.recall = recall
        self.network_stats.network_consistency = network_consistency
        self.network_stats.network_quality = network_quality

    def get_component_edge_len(
        self,
        comp_consistency_threshold: float,
        comp_quality_threshold: float,
        N: int = 1,
    ):
        """
        Computes the mean and std of lengths of edges contained in fundamental components.
        :param comp_consistency_threshold: chosen component consistency eta_c threshold.
        :param comp_quality_threshold: chosen component quality eta_q threshold.
        :param N: optional parameter loosening the definition of typical edges.
        :return: dataframe consisting of mean and stds per component and list of indices corresponding to fundamental components.
        """
        comp_distance_logger = {}
        considered_comp_idx_list = []

        for comp_idx in range(len(self.comp_stats)):
            # Consider only good components
            if (
                comp_idx < self.network_stats.first_trivial_component_idx
                and self.comp_stats[comp_idx].comp_consistency
                > comp_consistency_threshold
                and self.comp_stats[comp_idx].comp_quality > comp_quality_threshold
            ):
                comp_distances = np.array(
                    list(
                        nx.get_edge_attributes(
                            self.comp_stats[comp_idx].comp_graph, "weight"
                        ).items()
                    ),
                    dtype=object,
                )[:, -1].T
                mean_dist, std_dist = np.mean(comp_distances), np.std(comp_distances)
                self.comp_stats[comp_idx].mean_edge_len = mean_dist
                self.comp_stats[comp_idx].std_edge_len = std_dist
                upper_bound = mean_dist + N * std_dist
                comp_distance_logger[comp_idx] = [upper_bound]
                considered_comp_idx_list.append(comp_idx)
                logger.debug(
                    f"Component {comp_idx} distance interval: [0, {upper_bound}]"
                )
        comp_distances_df = pd.DataFrame.from_dict(comp_distance_logger).T.rename(
            columns={0: "mean_plus_std"}
        )  # index
        return comp_distances_df, considered_comp_idx_list

    def save_stats(self):
        """
        Saves global and local evaluation scores.
        :return: dictionary with dca scores.
        """
        output_json = {}
        for stat, stat_value in self.network_stats.__dict__.items():
            result_logger.debug(f"{stat}: {stat_value}")
            output_json[stat] = stat_value

        for component_id, component in enumerate(self.comp_stats):
            output_json[component_id] = {}
            result_logger.debug(
                "c(G{0}): {1:.2f}, q(G{0}): {2:.2f}, |G{0}^R|_v: {3:<4}, |G{0}^E|_v: {4:<4}, |G{0}|_v: {5:<4}".format(
                    component_id,
                    round(component.comp_consistency, 2),
                    round(component.comp_quality, 2),
                    len(component.Ridx),
                    len(component.Eidx),
                    len(component.Ridx) + len(component.Eidx),
                )
            )
            output_json[component_id]["comp_consistency"] = component.comp_consistency
            output_json[component_id]["comp_quality"] = component.comp_quality
            output_json[component_id]["num_R"] = len(component.Ridx)
            output_json[component_id]["num_E"] = len(component.Eidx)
            output_json[component_id]["num_RE"] = len(component.Ridx) + len(
                component.Eidx
            )
            if component.mean_edge_len is not None:
                output_json[component_id]["mean_edge_len"] = component.mean_edge_len
                output_json[component_id]["std_edge_len"] = component.std_edge_len
        return output_json
