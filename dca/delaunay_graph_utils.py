import networkx as nx
import numpy as np
import os
import pickle
import hdbscan
from dca.schemes import (
    DelaunayGraphParams,
    HDBSCANParams,
)
from hdbscan._hdbscan_linkage import label
import logging
from dca.loggers import logger_time
import gc


# -------------------------------------------------------------------------- #
# Logging settings
# -------------------------------------------------------------------------- #
logger = logging.getLogger("DCA_info_logger")
result_logger = logging.getLogger("DCA_result_logger")
time_logger = logging.getLogger("DCA_time_logger")
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #


@logger_time
def _approximate_Delaunay_edges(
    root: str, input_array_filepath: str, graph_params: DelaunayGraphParams
):
    """
    Phase 1
    Approximates the Delaunay graph.
    :param root: root directory of the experiment.
    :param input_array_filepath: path to R and E input array.
    :param graph_params: Delaunay graph parameters.
    """
    # Load unfiltered Delaunay graph if exists
    unfiltered_Delaunay_edges_filepath = os.path.join(
        root, graph_params.unfiltered_edges_filepath
    )
    if not os.path.isfile(unfiltered_Delaunay_edges_filepath):
        os_output = os.system(
            "./dca/approximate_Delaunay_graph {0} --nrays {1} --out {2} --out_dist {3}".format(
                os.path.join(root, input_array_filepath),
                graph_params.T,
                unfiltered_Delaunay_edges_filepath,
                os.path.join(root, graph_params.unfiltered_edges_len_filepath),
            )
        )
        if os_output != 0:
            raise ValueError(
                "Executable file dca/approximate_Delaunay_graph not working. Forgotten to change ownership?"
            )


@logger_time
def _filter_Delaunay_edges(
    unfiltered_Delaunay_edges_filepath: str,
    unfiltered_Delaunay_edges_len_filepath: str,
    graph_params: DelaunayGraphParams,
    filtered_Delaunay_edges_filepath: str,
    filtered_Delaunay_edges_len_filepath: str,
    n_points: int,
    n_query_points: int = 0,
):
    """
    Filters the approximated Delaunay graph with given sphere coverage parameter B as
    explained in Section 3.2.
    :param unfiltered_Delaunay_edges_filepath: path to unfiltered approximated Delaunay graph.
    :param unfiltered_Delaunay_edges_len_filepath: path to unfiltered approximated Delaunay edge lengths.
    :param graph_params: Delaunay graph parameters.
    :param filtered_Delaunay_edges_filepath: path to filtered approximated Delaunay graph.
    :param filtered_Delaunay_edges_len_filepath: path to filtered approximated Delaunay edge lengths.
    :param n_points: total number of points in R and E.
    :param n_query_points: total number of query points in Q.
    :return: shape of the unfilitered Delaunay edges.
    """
    if not os.path.isfile(filtered_Delaunay_edges_filepath):
        unfiltered_Delaunay_edges = np.load(unfiltered_Delaunay_edges_filepath)
        unfiltered_Delaunay_edges_len = np.load(unfiltered_Delaunay_edges_len_filepath)
        logger.debug("Unfiltered edges loaded")
        logger.debug("Chosen sphere coverage: %s", graph_params.sphere_coverage)
        logger.debug(
            "Unfiltered Delaunay edges shape: %s", unfiltered_Delaunay_edges.shape
        )
        unfiltered_Delaunay_edges_len_shape = unfiltered_Delaunay_edges_len.shape

        vertex_stats = {i: [] for i in range(n_points + n_query_points)}
        for edge_idx in range(len(unfiltered_Delaunay_edges)):
            (v1, v2, s) = unfiltered_Delaunay_edges[edge_idx]
            d = unfiltered_Delaunay_edges_len[edge_idx]
            vertex_stats[v1].append([v2, s, d, edge_idx])
            vertex_stats[v2].append([v1, s, d, edge_idx])
        logger.debug("Vertex stats dict created")

        # Needed when working with large datasets
        if n_points > 20000:
            del unfiltered_Delaunay_edges
            del unfiltered_Delaunay_edges_len
            gc.collect()
            logger.debug("Unfiltered edges deleted")

        significant_edge_idxs = []
        for vtx_idx, vtx_data in vertex_stats.items():
            vtx_data = np.array(vtx_data)
            if len(vtx_data) != 0:
                vtx_data = vtx_data[(-vtx_data[:, 2]).argsort()]  # sort by distance
                vtx_data[:, 1] = np.divide(
                    vtx_data[:, 1], graph_params.T * 2
                )  # normalize significance
                idx = -1
                cur_coverage = 0
                while (
                    -idx <= len(vtx_data)
                    and cur_coverage < graph_params.sphere_coverage
                ):
                    significant_edge_idxs.append(int(vtx_data[idx, 3]))
                    cur_coverage += vtx_data[idx, 1]
                    idx += -1
        logger.debug("Significant edges extracted")

        # Needed when working with large datasets
        if n_points > 20000:
            del vertex_stats
            gc.collect()
            logger.debug("Vertex stats dict deleted")

        filtered_significant_edge_idxs, _ = np.unique(
            significant_edge_idxs, return_counts=True
        )
        logger.debug("Unique significant edges extracted")

        # Needed when working with large datasets
        if n_points > 20000:
            del significant_edge_idxs
            gc.collect()
            logger.debug("Duplicate significant edges deleted")
            unfiltered_Delaunay_edges = np.load(unfiltered_Delaunay_edges_filepath)
            unfiltered_Delaunay_edges_len = np.load(
                unfiltered_Delaunay_edges_len_filepath
            )
            logger.debug("Unfiltered edges loaded again")

        filtered_Delaunay_edges = unfiltered_Delaunay_edges[
            filtered_significant_edge_idxs, :2
        ]
        filtered_Delaunay_edges_len = unfiltered_Delaunay_edges_len[
            filtered_significant_edge_idxs
        ]
        logger.debug("Filtered Delaunay edges shape: %s", filtered_Delaunay_edges.shape)

        # Needed when working with large datasets
        if n_points > 20000:
            del unfiltered_Delaunay_edges
            del unfiltered_Delaunay_edges_len
            gc.collect()
            logger.debug("Unfiltered edges deleted")

        # Save filtered edges and their length
        np.save(filtered_Delaunay_edges_filepath, filtered_Delaunay_edges)
        np.save(filtered_Delaunay_edges_len_filepath, filtered_Delaunay_edges_len)
        return unfiltered_Delaunay_edges_len_shape


@logger_time
def _distil_Delaunay_connected_components(
    root: str, params: HDBSCANParams, graph: nx.Graph
):
    """
    Phase 2
    Distils the Delaunay graph into connected components using HDBSCAN as
    explained in Section 3.
    :param root: root directory of the experiment.
    :param params: HDBSCAN hyperparameters.
    :param graph: approximated Delaunay graph obtained in Phase 1.
    :return: array of labels corresponding to each point in R U E.
    """
    clusterer_path = os.path.join(root, params.clusterer_filepath)
    if os.path.isfile(clusterer_path):
        with open(clusterer_path, "rb") as f:
            clusterer = pickle.load(f)
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params.min_cluster_size,
            min_samples=params.min_samples,
            metric="precomputed",
            cluster_selection_epsilon=params.cluster_selection_epsilon,
        )

        min_spanning_tree = nx.to_pandas_edgelist(
            nx.minimum_spanning_tree(graph)
        ).to_numpy()
        logger.debug("Minimum spanning tree extracted")

        del graph
        gc.collect()
        logger.debug("Graph deleted")

        min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]

        # Convert edge list into standard hierarchical clustering format
        single_linkage_tree = label(min_spanning_tree)
        logger.debug("Single linkage tree extracted")

        (
            clusterer.labels_,
            clusterer.probabilities_,
            clusterer.cluster_persistence_,
            clusterer._condensed_tree,
            clusterer._single_linkage_tree,
        ) = hdbscan.hdbscan_._tree_to_labels(
            None,
            single_linkage_tree,
            params.min_cluster_size,
            cluster_selection_epsilon=params.cluster_selection_epsilon,
        )

        clusterer._min_spanning_tree = min_spanning_tree

        with open(clusterer_path, "wb") as f:
            pickle.dump(clusterer, f)
        logger.debug("HDBSCAN clusterer saved")

    labels = clusterer.labels_
    del clusterer
    gc.collect()

    return labels


@logger_time
def _sort_Delaunay_connected_components(
    root: str,
    input_data_labels: np.ndarray,
    clustering_params: HDBSCANParams,
    n_points: int,
):
    """
    Sorts the connected components obtained from HDBSCAN in decreasing order by size.
    :param root: root directory of the experiment.
    :param input_data_labels: labels of R and E points given by HDBSCAN.
    :param clustering_params: HDBSCAN hyperparameters.
    :param n_points: total number of points in R and E.
    :return: dict with keys representing component indices and arrays of
    corresponding vertices containined in each component as values;
    index of the first trivial component.
    """
    # Relabel outliers
    n_outliers = np.count_nonzero(input_data_labels == -1)
    per_outliers = n_outliers / n_points
    if per_outliers > 0.2:
        logger.warning(
            f"Warning: DCA obtained {per_outliers}% of outliers. Consider modifying HDBSCAN parameters."
        )
    outlier_idxs = np.arange(
        max(input_data_labels) + 1,
        max(input_data_labels) + 1 + n_outliers,
    )
    first_trivial_component_idx = max(input_data_labels) + 1
    input_data_labels[input_data_labels == -1] = outlier_idxs
    logger.debug("Outliers relabeled")
    logger.debug("Number of outliers: {0}".format(n_outliers))
    logger.debug(
        "Number of all connected components: {0}".format(
            n_outliers + first_trivial_component_idx
        )
    )

    # Relabel compoments by size
    unique_idx, counts = np.unique(input_data_labels, return_counts=True)
    relabel = np.argsort(counts.argsort()[::-1])[input_data_labels]
    np.save(
        os.path.join(root, clustering_params.input_array_labels_filepath),
        relabel,
    )
    logger.debug("Connected components relabeled by size")

    del input_data_labels
    gc.collect()
    logger.debug("Input data labels saved and removed from memory")

    component_vertex_idx_mapping = {idx: [] for idx in unique_idx}
    for label_idx, label in enumerate(relabel):
        component_vertex_idx_mapping[label].append(label_idx)
    logger.debug("Component vertex id mapping created")

    return component_vertex_idx_mapping, first_trivial_component_idx


def _extract_RE_subgraphs(graph: nx.Graph, num_R: int, num_E: int):
    """
    Extracts graph restrictions H^R and H^E from a given graph H.
    :param graph: Delaunay graph (or component) to extract restrictions from.
    :param num_R: total number of R points.
    :param num_E: total number of E points.
    :return: graph resticted to R; graph resticted to E.
    """
    R_graph = graph.subgraph(np.arange(num_R))
    E_graph = graph.subgraph(np.arange(num_R, num_R + num_E))
    return R_graph, E_graph


@logger_time
def _evaluate_Delaunay_component(
    subgraph_RE_comp: nx.Graph,
    subgraph_R_comp: nx.Graph,
    subgraph_E_comp: nx.Graph,
    num_R: int,
):
    """
    Phase 3
    Evaluates given graph-connected component.
    :param subgraph_RE_comp: Delaunay graph of the component containing R and E points..
    :param subgraph_R_comp: Delaunay graph of component restricted to R.
    :param subgraph_E_comp: Delaunay graph of component restricted to E.
    :param num_R: total number of R points.
    :return: indices of R points in the component; indices of E points in the component;
    component consistency score; component quality score; number of heteroeneous edges
    in the component; total number of edges in the component
    """

    comp_R_idxs = np.array(list(subgraph_R_comp.nodes()))
    comp_E_idxs = np.array(list(subgraph_E_comp.nodes())) - num_R
    comp_consistency = _get_graph_consistency(len(comp_R_idxs), len(comp_E_idxs))
    comp_quality, num_comp_RE_edges, num_total_comp_edges = _get_graph_quality(
        subgraph_RE_comp, subgraph_R_comp, subgraph_E_comp
    )
    return (
        comp_R_idxs,
        comp_E_idxs,
        comp_consistency,
        comp_quality,
        num_comp_RE_edges,
        num_total_comp_edges,
    )


@logger_time
def _get_graph_consistency(num_R_vertices: int, num_E_vertices: int):
    """
    Calculates the consistency of the given graph as in Definition 3.2.
    :param num_R_vertices: number of R vertices contained in a graph.
    :param num_E_vertices: number of E vertices contained in a graph.
    :return: consistency score of the graph.
    """
    return 1 - abs(num_R_vertices - num_E_vertices) / (num_R_vertices + num_E_vertices)


@logger_time
def _get_graph_quality(RE_graph: nx.Graph, R_graph: nx.Graph, E_graph: nx.Graph):
    """
    Calculates the quality of the given graph as in Definition 3.2.
    :param RE_graph: Delaunay graph (or component) containing both R and E points.
    :param R_graph: Delaunay graph (or component) restricted to R.
    :param E_graph: Delaunay graph (or component) restricted to E.
    :return: quality score of the graph; number of heterogeneous edges; total number of edges
    """
    num_R_and_E_edges = R_graph.number_of_edges() + E_graph.number_of_edges()
    num_total_edges = RE_graph.number_of_edges()
    num_RE_edges = num_total_edges - num_R_and_E_edges
    graph_quality = num_RE_edges / num_total_edges if num_total_edges != 0 else 0
    return graph_quality, num_RE_edges, num_total_edges


@logger_time
def _approximate_query_Delaunay_edges(
    exp_log_dir: str,
    input_array_filepath: str,
    query_input_array_filepath: str,
    graph_params: DelaunayGraphParams,
):
    """
    q-DCA
    Approximates edges to query points as in Secion 3.2,
    :param exp_log_dir: export directory to save results to.
    :param input_array_filepath: path to R and E input array.
    :param query_input_array_filepath: path to query points array.
    :param graph_params: Delaunay graph parameters.
    """
    query_Delaunay_edges_filepath = os.path.join(
        exp_log_dir, graph_params.query_unfiltered_edges_filename
    )
    # Load unfiltered Delaunay graph if exists
    if not os.path.isfile(query_Delaunay_edges_filepath):
        os_output = os.system(
            "./dca/approximate_Delaunay_graph {0} --queries {1} --nrays {2} --out {3} --out_dist {4}".format(
                input_array_filepath,
                query_input_array_filepath,
                graph_params.T * 2,
                query_Delaunay_edges_filepath,
                os.path.join(
                    exp_log_dir, graph_params.query_unfiltered_edges_len_filename
                ),
            )
        )
        if os_output != 0:
            raise ValueError(
                "Executable file dca/approximate_Delaunay_graph not working. Forgotten to change ownership?"
            )
