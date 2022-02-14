import gc
import networkx as nx
import numpy as np
import os
import logging
import pickle
from dca.schemes import (
    DCALoggers,
    DelaunayGraphVisualizer,
    REData,
    ExperimentDirs,
    DelaunayGraphParams,
    HDBSCANParams,
    GeomCAParams,
    QueryData,
)
from dca.delaunay_graph import DelaunayGraph
from typing import Optional, List
import pandas as pd
import dca.delaunay_graph_utils as graph_utils
import dca.visualization as visualization
import logging.config
from dca.loggers import logger_time, get_parameters
import json

# -------------------------------------------------------------------------- #
# Logging settings
# -------------------------------------------------------------------------- #
logger = logging.getLogger("DCA_info_logger")
result_logger = logging.getLogger("DCA_result_logger")
time_logger = logging.getLogger("DCA_time_logger")
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #


class DCA:
    def __init__(
        self,
        dirs: ExperimentDirs,
        Delaunay_graph_params: DelaunayGraphParams,
        clustering_params: HDBSCANParams,
        GeomCA_params: GeomCAParams,
        loggers: DCALoggers,
        random_seed: int = 1111,
    ):
        np.random.seed(random_seed)

        # Paths
        self.root = dirs.experiment_dir
        self.precomputed_dir = dirs.precomputed_dir
        self.DCA_dir = dirs.DCA_dir

        self.visualization_dir = dirs.visualization_dir
        self.results_dir = dirs.results_dir
        self.logs_dir = dirs.logs_dir

        # Initialize the loggers
        self.loggers = loggers
        logging.config.dictConfig(loggers.loggers)

        # Parameters
        self.GeomCA_params = GeomCA_params
        self.graph_params = Delaunay_graph_params
        self.clustering_params = clustering_params
        self.save_parameters(
            [dirs, Delaunay_graph_params, clustering_params, GeomCA_params],
            random_seed,
            loggers.version,
        )

        # Visualisation
        self.visualize_Delaunay_graph = False

    # -------------------------------------------------------------------------- #
    # Prepocess data and save parameters
    # -------------------------------------------------------------------------- #
    def preprocess_data(self, data: REData):
        """
        Prepares the input array for Delaunay approximation.
        :param data: R and E data parameters.
        :return: DelaunayGraphVisualizer object.
        """
        input_array_filepath = os.path.join(self.root, data.input_array_filepath)
        if not os.path.isfile(input_array_filepath):
            input_array = np.concatenate([data.R, data.E]).astype(np.float32)
            np.save(input_array_filepath, input_array)
        if data.visualize:
            G_visualizer = DelaunayGraphVisualizer(
                os.path.join(self.root, data.input_array_filepath),
                data.num_R,
                data.num_E,
            )
            logger.debug("DelaunayGraphVisualizer initialized")
            return G_visualizer

    def preprocess_query_data(self, query_data: QueryData):
        """
        Prepares the input array of query points for query point Delaunay approximation.
        :param query_data: query data parameters.
        """
        query_input_array_filepath = os.path.join(
            self.root, query_data.query_input_array_filepath
        )
        if not os.path.isfile(query_input_array_filepath):
            input_array = query_data.Q.astype(np.float32)
            np.save(query_input_array_filepath, input_array)

    def save_parameters(
        self,
        params_list: List,
        random_seed: int = 1111,
        version: int = 0,
    ):
        """
        Saves input parameters.
        :param params_list: list of input parameters.
        :param random_seed:
        :param version: experiment version index.
        """
        params_dict = {"random_seed": random_seed}
        for params in params_list:
            dict = get_parameters(params)
            params_dict = {**params_dict, **dict}

        with open(
            os.path.join(self.logs_dir, f"version{version}_input.json"),
            "w",
        ) as f:
            json.dump(params_dict, f, indent=4)

    # -------------------------------------------------------------------------- #
    # DCA: R and E
    # -------------------------------------------------------------------------- #
    @logger_time
    def fit(self, data: REData):
        """
        DCA
        Runs DCA algorithm on the given sets of representations R and E.
        :param data: R and E data parameters.
        :return: DCA local and global evaluation scores.
        """
        print("Starting to run DCA...")

        # Preprocess input data
        G_visualizer = self.preprocess_data(data)
        logger.debug("Input data saved")

        # Get Delaunay graph
        G = self.get_Delaunay_graph(data, G_visualizer)
        logger.debug("DelaunayGraph initialized")
        print("- Delaunay graph approximated.")

        n_points = data.num_R + data.num_E
        del data
        gc.collect()

        # Get Delaunay connected components
        (
            component_vertex_idx_mapping,
            first_non_trivial_component,
        ) = self.get_Delaunay_connected_components(n_points, G.graph, G_visualizer)
        logger.debug("Delaunay connected components obtained")

        G.set_first_trivial_component_idx(first_non_trivial_component.item())
        logger.debug(
            "Delaunay first non trivial component set to: %s",
            first_non_trivial_component,
        )
        print("- Distilled Delaunay graph built.")

        # Analyse Delaunay connected components
        self.analyse_Delaunay_connected_components(
            component_vertex_idx_mapping, G, G_visualizer
        )
        logger.debug("Delaunay connected components analysed")
        print("- Distilled Delaunay graph analysed.")

        # Save results
        output = self.save_DCA_logs(G)
        logger.debug("- DCA results saved.")

        # Plot results
        visualization._plot_RE_components_consistency(
            G,
            self.visualization_dir,
            min_comp_size=2,
            annotate_largest=True,
            display_smaller=False,
        )
        visualization._plot_RE_components_quality(
            G,
            self.visualization_dir,
            min_comp_size=2,
            annotate_largest=True,
            display_smaller=False,
        )
        logger.debug("DCA results visualized")
        print("- DCA executed, results saved to: {0}.".format(self.DCA_dir))
        return output

    @logger_time
    def get_Delaunay_graph(
        self, data: REData, visualizer: Optional[DelaunayGraphVisualizer] = None
    ):
        """
        Phase 1
        Approximates and filters Delunay graph on the given sets of representations R and E.
        :param data: R and E data parameters.
        :param visualizer: DelaunayGraphVisualizer object.
        :return: approximated and filtered Delaunay graph.
        """
        # Build Delaunay edges if it does not exists
        graph_utils._approximate_Delaunay_edges(
            self.root, data.input_array_filepath, self.graph_params
        )
        logger.debug(
            "Delaunay graph {0} created with parameter nrays={1}.".format(
                os.path.join(self.root, self.graph_params.unfiltered_edges_filepath),
                self.graph_params.T,
            )
        )

        # Filter Delaunay edges if specified
        if self.graph_params.sphere_coverage == 1.0:
            Delaunay_edges = np.load(
                os.path.join(self.root, self.graph_params.unfiltered_edges_filepath)
            )[:, :2]
            Delaunay_edges_len = np.load(
                os.path.join(self.root, self.graph_params.unfiltered_edges_len_filepath)
            )
            logger.debug("Unfiltered Delaunay edges of shape: %s", Delaunay_edges.shape)

        else:
            logger.debug(
                "Chosen sphere coverage: %s", self.graph_params.sphere_coverage
            )
            unfiltered_Delaunay_edges_shape = graph_utils._filter_Delaunay_edges(
                os.path.join(self.root, self.graph_params.unfiltered_edges_filepath),
                os.path.join(
                    self.root, self.graph_params.unfiltered_edges_len_filepath
                ),
                self.graph_params,
                os.path.join(self.root, self.graph_params.filtered_edges_filepath),
                os.path.join(self.root, self.graph_params.filtered_edges_len_filepath),
                data.num_R + data.num_E,
            )

            Delaunay_edges = np.load(
                os.path.join(self.root, self.graph_params.filtered_edges_filepath)
            )
            Delaunay_edges_len = np.load(
                os.path.join(self.root, self.graph_params.filtered_edges_len_filepath)
            )

            logger.debug(
                "Unfiltered Delaunay graph shape: %s", unfiltered_Delaunay_edges_shape
            )
            logger.debug("Filtered Delaunay graph shape: %s", Delaunay_edges.shape)
        logger.debug("Delaunay edges extracted")

        # Init DelaunayGraph
        G = DelaunayGraph(data.num_R, data.num_E)
        G.init_Delaunay_graph(Delaunay_edges, Delaunay_edges_len)

        if visualizer is not None:
            visualization._plot_Delaunay_graph(
                visualizer,
                edges=Delaunay_edges,
                filename="approximated_Delaunay_graph",
                root=self.visualization_dir,
            )
            logger.debug("Delaunay edges visualized")
        return G

    @logger_time
    def get_Delaunay_connected_components(
        self,
        n_points: int,
        graph: nx.Graph,
        visualizer: Optional[DelaunayGraphVisualizer] = None,
    ):
        """
        Phase 2
        Distilles the approximated Delunay graph into connected components.
        :param n_points: total number of points in R and E.
        :param graph: approximated Delaunay graph.
        :param visualizer: DelaunayGraphVisualizer object.
        :return: dict with keys representing component indices and arrays of
        corresponding vertices containined in each component as values;
        index of the first non trivial component.
        """
        # Perform HDBSCAN clustering
        input_array_labels = graph_utils._distil_Delaunay_connected_components(
            self.root, self.clustering_params, graph
        )
        logger.debug("HDBSCAN executed")
        logger.debug(
            "Number of significant connected components: %s",
            len(np.unique(input_array_labels)),
        )

        if visualizer is not None:
            xlim, ylim = visualization._plot_Delaunay_graph(
                visualizer,
                graph.edges,
                filename="Delaunay_components",
                root=self.visualization_dir,
                labels=input_array_labels,
            )
            logger.debug("Delaunay connected components visualized")
            visualizer.xlim = xlim
            visualizer.ylim = ylim
            logger.debug(f"DelaunayGraphVisualizer updated xlim={xlim} and ylim={ylim}")

        # Extract components sorted by their vertex size
        (
            component_vertex_idx_mapping,
            first_non_trivial_component,
        ) = graph_utils._sort_Delaunay_connected_components(
            self.root,
            input_array_labels,
            self.clustering_params,
            n_points,
        )
        logger.debug("Delaunay connected components extracted")
        gc.collect()
        return (component_vertex_idx_mapping, first_non_trivial_component)

    @logger_time
    def analyse_Delaunay_connected_components(
        self,
        component_vertex_idx_mapping: dict,
        G: DelaunayGraph,
        visualizer: Optional[DelaunayGraphVisualizer] = None,
        discard_component_graph: Optional[bool] = True,
    ):
        """
        Phase 3
        Analyses the connected components of the distilled Delunay graph.
        :param component_vertex_idx_mapping: dictionary of vertex indices contained in each component.
        :param G: distilled Delaunay graph.
        :param visualizer: DelaunayGraphVisualizer object.
        :param discard_component_graph: whether to discard the component nx.Graph object (storage heavy).
        """

        for comp_idx, comp_vertices in component_vertex_idx_mapping.items():
            subgraph_RE_comp = G.graph.subgraph(comp_vertices)
            if nx.is_empty(subgraph_RE_comp):
                subgraph_RE_comp = nx.Graph()
                subgraph_RE_comp.add_nodes_from(comp_vertices)

            if visualizer is not None and comp_idx < G.first_trivial_component_idx:
                visualization._plot_Delaunay_graph(
                    visualizer,
                    edges=subgraph_RE_comp.edges,
                    filename=f"component_{comp_idx}_Delaunay",
                    root=self.visualization_dir,
                    vertices=np.array(comp_vertices),
                    keep_range=True,
                )
                logger.debug(f"Delaunay connected component {comp_idx} visualized")

            subgraph_R_comp, subgraph_E_comp = graph_utils._extract_RE_subgraphs(
                subgraph_RE_comp, G.num_R, G.num_E
            )
            (
                comp_R_idxs,
                comp_E_idxs,
                comp_consistency,
                comp_quality,
                num_comp_RE_edges,
                num_total_comp_edges,
            ) = graph_utils._evaluate_Delaunay_component(
                subgraph_RE_comp, subgraph_R_comp, subgraph_E_comp, G.num_R
            )
            logger.debug(f"Delaunay connected component {comp_idx} analyzed")

            G.update_local_stats(
                comp_R_idxs,
                comp_E_idxs,
                comp_consistency,
                comp_quality,
                num_comp_RE_edges,
                num_total_comp_edges,
                self.GeomCA_params.comp_consistency_threshold,
                self.GeomCA_params.comp_quality_threshold,
                None if not discard_component_graph else subgraph_RE_comp,
            )
            logger.debug(f"DelaunayGraph updated local stats with component {comp_idx}")

        if visualizer is not None:
            visualization._plot_isolated_components(
                G, visualizer, self.visualization_dir
            )
            logger.debug("Isolated Delaunay connected components")

            visualization._plot_Delaunay_graph(
                visualizer,
                edges=G.distil_edges(),
                filename="distilled_Delaunay_graph",
                root=self.visualization_dir,
            )
            logger.debug("distilled Delaunay edges visualized")

        G.update_global_stats()
        logger.debug(f"DelaunayGraph updated global stats")

    def save_DCA_logs(self, G: DelaunayGraph):
        """
        Saves DCA scores to files.
        :param G: distilled Delaunay graph with local and global evaluation scores.
        """
        path = os.path.join(self.results_dir, "network_stats.pkl")
        with open(path, "wb") as f:
            pickle.dump(G.network_stats, f)
        logger.debug(f"DelaunayGraph network_stats saved")

        path = os.path.join(self.results_dir, "components_stats.pkl")
        with open(path, "wb") as f:
            pickle.dump(G.comp_stats, f)
        logger.debug(f"DelaunayGraph components_stats saved")

        output = G.save_stats()
        with open(os.path.join(self.DCA_dir, "output.json"), "w") as fp:
            json.dump(output, fp, indent=4)

    def cleanup(self, remove_visualizations: bool = True, remove_logs: bool = True):
        """
        Removes the DCA files in the experiment folder. Default removes all files except for the output scores.
        :param remove_visualizations: whether to remove the visualizations.
        :param remove_logs: whether to remove the logging files.
        """
        # Remove precomputed folder
        os.system(f"rm -r {self.precomputed_dir}")

        # Remove DCA dir
        os.system((f"rm -r {self.results_dir}"))

        # Remove logs
        if remove_logs:
            os.system(f"rm -r {self.logs_dir}")
        else:  # Remove all non-log files, eg npy from qDCA
            for file in os.listdir(str(self.logs_dir)):
                if not file.endswith(".logs"):
                    os.system(f"rm {file}")

        # Remove logs
        if remove_visualizations:
            os.system(f"rm -r {self.visualization_dir}")
        print("- Cleanup completed.")

    # -------------------------------------------------------------------------- #
    # qDCA: query point processing
    # -------------------------------------------------------------------------- #
    @logger_time
    def process_query_points(
        self,
        init_data: REData,
        query_data: QueryData,
        assign_to_comp: bool = False,
        consider_several_assignments: bool = False,
        assign_to_R: bool = False,
        assign_to_RE: bool = False,
        return_len: bool = False,
    ):
        """
        query point Delaunay Component Analysis (q-DCA).
        :param init_data: R and E data parameters.
        :param query_data: query data parameters.
        :param assign_to_comp: whether to assign query points to fundamental components.
        :param consider_several_assignments: whether to consider fliexible assignment.
        :param assign_to_R: whether to assign query points to R points only.
        :param assign_to_RE: whether to assign query points to R and E points.
        :param return_len: whether to return the length of the shortest edges.
        :return: dataframe of query point indices and the associated assignments.
        """
        self.loggers.qdca_flag = True
        G = DelaunayGraph(init_data.num_R, init_data.num_E)
        G.load_existing(self.results_dir)
        logger.debug("Loaded existing DelaunayGraph")
        self.preprocess_query_data(query_data)

        if assign_to_comp:
            (
                query_points_comp_labels,
                considered_comp_idx_list,
            ) = self.assign_query_points_to_components(
                init_data,
                query_data,
                G,
                consider_several_assignments=consider_several_assignments,
            )
            logger.debug("Query points assigned to connected components")
            print(
                "- qDCA assignment to components executed, results saved to: {0}.".format(
                    self.results_dir
                )
            )
            return query_points_comp_labels, considered_comp_idx_list

        elif assign_to_RE:
            query_points_nclosest_init_point_idxs = (
                self.assign_query_points_to_closest_init_point(
                    init_data, query_data, return_len=return_len
                )
            )
            logger.debug("Query points assigned to closest RE point")
            print(
                "- qDCA assignment to closest RE executed, results saved to: {0}.".format(
                    self.results_dir
                )
            )
            return query_points_nclosest_init_point_idxs

        elif assign_to_R:
            query_points_nclosest_init_point_idxs = (
                self.assign_query_points_to_closest_init_point(
                    init_data, query_data, assign_to_R=True
                )
            )
            logger.debug("Query points assigned to closest R point")
            print(
                "- qDCA assignment to closest R executed, results saved to: {0}.".format(
                    self.results_dir
                )
            )
            return query_points_nclosest_init_point_idxs

        else:
            raise ValueError(
                "Query pont processing format not specified, choose one option."
            )

    @logger_time
    def assign_query_points_to_components(
        self,
        init_data: REData,
        query_data: QueryData,
        G: DelaunayGraph,
        consider_several_assignments: bool = False,
    ):
        """
        Assigns query points to fundamental components.
        :param init_data: R and E data parameters.
        :param query_data: query data parameters.
        :param G: existing distilled Delaunay graph.
        :param consider_several_assignments: whether to consider fliexible assignment.
        :return: dataframe of query point indices and the associated assignments;
        indices of fundamental components.
        """
        # Compute average edge length of each component
        comp_distances_df, considered_comp_idx_list = G.get_component_edge_len(
            self.GeomCA_params.comp_consistency_threshold,
            self.GeomCA_params.comp_quality_threshold,
        )
        self.save_DCA_logs(G)
        logger.debug("Average edge length per (non-trivial) component extracted")
        print("- Delaunay edges to query points obtained.")

        # Remove query edges connecting to outliers defined by HDBSCAN
        query_edges_df = self.get_query_Delaunay_edges_stats(init_data, query_data)
        query_edges_to_components_df = query_edges_df[
            query_edges_df["label"] < G.first_trivial_component_idx
        ]
        logger.debug("Average edge length per (non-trivial) component extracted")

        # For each query point and component it is connected to:
        # get the shortest edge length to each component and number of edges connecting
        # to that component
        df = (
            query_edges_to_components_df.groupby(["query_idx", "label"])[["len"]]
            .agg(["min", "count"])
            .reset_index()
        )
        df.columns = df.columns.droplevel(0)
        df.reset_index()
        df.columns = ["query_idx", "label", "len", "init_idx_count"]

        # Merge with the range of component edges from the distilled Delaunay graph
        df = df.join(comp_distances_df, on="label")

        # Extract query points whose shortest edges fall within the edge length range
        # of the corresponding component
        df_component_assignment = df[df.len <= df.mean_plus_std]
        num_comp_assignments = (
            df_component_assignment.groupby(["query_idx"])["label"]
            .count()
            .to_frame()
            .rename(columns={"label": "num_comp_assignments"})
        )
        df_component_assignment = df_component_assignment.join(
            num_comp_assignments, on="query_idx"
        )

        # Conservative assignment:
        # Extract only those points that belong to one component
        num_Q = query_data.num_Q
        query_points_comp_assignment = (
            df_component_assignment[df_component_assignment.num_comp_assignments == 1][
                ["query_idx", "label"]
            ]
            .to_numpy()
            .astype(int)
        )

        # Flexible assignment:
        # Extract those points that are assigned to more components
        if consider_several_assignments:
            two_assignments = df_component_assignment[
                df_component_assignment.num_comp_assignments >= 2
            ]
            two_assignments_sorted_by_len = two_assignments.sort_values(
                by=["query_idx", "len"]
            )
            extra_assignments_by_len = two_assignments_sorted_by_len.drop_duplicates(
                subset=["query_idx"], keep="first"
            )

            two_assignments_sorted_by_num_edges = two_assignments.sort_values(
                by=["query_idx", "init_idx_count"]
            )
            extra_assignments_by_num_edges = (
                two_assignments_sorted_by_num_edges.drop_duplicates(
                    subset=["query_idx"], keep="last"
                )
            )

            extras = pd.merge(
                extra_assignments_by_len,
                extra_assignments_by_num_edges,
                how="inner",
                on=list(extra_assignments_by_num_edges),
            )
            assert (
                np.intersect1d(
                    extras["query_idx"], query_points_comp_assignment[:, 0]
                ).size
                == 0
            )
            extra_labels = extras[["query_idx", "label"]].astype(int).to_numpy()
            query_points_comp_assignment = np.concatenate(
                [query_points_comp_assignment, extra_labels]
            )

        query_points_comp_assignment[:, 0] -= init_data.num_R + init_data.num_E

        not_assigned_idx = np.setdiff1d(
            np.arange(num_Q), query_points_comp_assignment[:, 0]
        )
        not_assigned = np.array(
            [not_assigned_idx, np.repeat(-1, len(not_assigned_idx))]
        ).T

        query_points_comp_labels = np.concatenate(
            [query_points_comp_assignment, not_assigned]
        )
        query_points_comp_labels = query_points_comp_labels[
            np.argsort(query_points_comp_labels[:, 0])
        ]

        # Save the assignment results
        np.save(
            os.path.join(
                self.results_dir, query_data.query_input_array_comp_assignment_filename
            ),
            query_points_comp_labels,
        )
        np.save(
            os.path.join(
                self.results_dir,
                query_data.query_input_array_considered_comp_list_filename,
            ),
            np.array(considered_comp_idx_list),
        )
        return query_points_comp_labels, considered_comp_idx_list

    @logger_time
    def assign_query_points_to_closest_init_point(
        self,
        init_data: REData,
        query_data: QueryData,
        n_closest: int = 1,
        assign_to_R: bool = False,
        return_len: bool = False,
    ):
        """
        Assigns query points to closest R (and E) point.
        :param init_data: R and E data parameters.
        :param query_data: query data parameters.
        :param n_closest: number of closest neighbours to consider.
        :param assign_to_R: whether to assign query points to R points only.
        :param return_len: whether to return the length of the shortest edges.
        :return: dataframe of query point indices and the associated assignments.
        """
        query_edges_df = self.get_query_Delaunay_edges_stats(init_data, query_data)
        query_edges_df.query_idx -= init_data.num_R + init_data.num_E

        # Whether to consider edges to E points or not
        if assign_to_R:
            query_edges_df = query_edges_df[query_edges_df.init_idx < init_data.num_R]

        if n_closest > 1:
            nclosest_init_point_list = []
            for query_idx in query_edges_df.query_idx.unique():
                nclosest_init_idxs = (
                    query_edges_df[query_edges_df.query_idx == query_idx]
                    .nsmallest(n_closest, "len")["init_idx"]
                    .to_numpy()
                    .astype(int)
                )
                nclosest_init_idxs = np.insert(nclosest_init_idxs, 0, int(query_idx))
                # Pad if not enough neighbours
                nclosest_init_idxs = np.pad(
                    nclosest_init_idxs,
                    (0, max(n_closest + 1 - len(nclosest_init_idxs), 0)),
                    mode="constant",
                    constant_values=-1,
                )
                nclosest_init_point_list.append(nclosest_init_idxs)

            query_points_nclosest_init_point_idxs = np.stack(nclosest_init_point_list)
            np.save(
                os.path.join(
                    self.results_dir,
                    query_data.query_input_array_point_assignment_filename,
                ),
                query_points_nclosest_init_point_idxs,
            )
            return query_points_nclosest_init_point_idxs

        else:
            # Find closest
            df = query_edges_df.loc[
                query_edges_df.groupby(["query_idx"])["len"].idxmin()
            ]
            if return_len:
                query_points_closest_init_point_idxs = (
                    df[["query_idx", "init_idx", "len"]].to_numpy().astype(float)
                )
            else:
                query_points_closest_init_point_idxs = (
                    df[["query_idx", "init_idx"]].to_numpy().astype(int)
                )
            np.save(
                os.path.join(
                    self.results_dir,
                    query_data.query_input_array_point_assignment_filename,
                ),
                query_points_closest_init_point_idxs,
            )
            return query_points_closest_init_point_idxs

    @logger_time
    def get_query_Delaunay_edges_stats(self, init_data: REData, query_data: QueryData):
        """
        Extracts graph neighbourhood of each query point.
        :param init_data: R and E data parameters.
        :param query_data: query data parameters.
        :return: dataframe of query point indices and the associated neughbourhood.
        """
        Delaunay_edges_stats_filepath = os.path.join(
            self.logs_dir, query_data.query_input_array_edges_stats_filename
        )
        try:
            query_edges_df = pd.read_pickle(Delaunay_edges_stats_filepath)
        except:
            # Extract query Delaunay edges
            graph_utils._approximate_query_Delaunay_edges(
                self.logs_dir,
                os.path.join(self.root, init_data.input_array_filepath),
                os.path.join(self.root, query_data.query_input_array_filepath),
                self.graph_params,
            )
            logger.debug("Unfiltered query Delaunay edges extracted")

            # Filter query edges as in the initial approximated Delaunay graph
            if self.graph_params.sphere_coverage == 1.0:
                query_Delaunay_edges = np.load(
                    os.path.join(
                        self.logs_dir, self.graph_params.query_unfiltered_edges_filename
                    )
                )[:, :2]
                query_Delaunay_edges_len = np.load(
                    os.path.join(
                        self.logs_dir,
                        self.graph_params.query_unfiltered_edges_len_filename,
                    )
                )
            else:
                graph_utils._filter_Delaunay_edges(
                    os.path.join(
                        self.logs_dir, self.graph_params.query_unfiltered_edges_filename
                    ),
                    os.path.join(
                        self.logs_dir,
                        self.graph_params.query_unfiltered_edges_len_filename,
                    ),
                    self.graph_params,
                    os.path.join(
                        self.logs_dir, self.graph_params.query_filtered_edges_filename
                    ),
                    os.path.join(
                        self.logs_dir,
                        self.graph_params.query_filtered_edges_len_filename,
                    ),
                    n_points=init_data.num_R + init_data.num_E,
                    n_query_points=query_data.num_Q,
                )

                query_Delaunay_edges = np.load(
                    os.path.join(
                        self.logs_dir, self.graph_params.query_filtered_edges_filename
                    )
                )
                query_Delaunay_edges_len = np.load(
                    os.path.join(
                        self.logs_dir,
                        self.graph_params.query_filtered_edges_len_filename,
                    )
                )
            logger.debug("query Delaunay edges extracted")

            # Get component idx (label) that each query edge connects to and its length
            input_array_comp_labels = np.load(
                os.path.join(
                    self.root, self.clustering_params.input_array_labels_filepath
                )
            )
            query_edges_array = np.stack(
                [
                    query_Delaunay_edges[:, 0],
                    query_Delaunay_edges[:, 1],
                    input_array_comp_labels[query_Delaunay_edges[:, 1]],
                    query_Delaunay_edges_len.squeeze(),
                ],
                axis=1,
            ).astype(np.float32)
            query_edges_df = pd.DataFrame(
                data=query_edges_array,
                columns=["query_idx", "init_idx", "label", "len"],
            ).astype(float)

            query_edges_df.to_pickle(
                os.path.join(
                    self.logs_dir, query_data.query_input_array_edges_stats_filename
                )
            )
        return query_edges_df
