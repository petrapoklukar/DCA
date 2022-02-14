import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def get_representations_by_class(
    representations: np.ndarray,
    labels: np.ndarray,
    Rclasses: List,
    Eclasses: Optional[List] = None,
):
    """
    Extracts representations of a particular class from a given array.
    :param representations: array of representations.
    :param labels: array of labels corresponding to the representations.
    :param Rclasses: list of R classes to extract.
    :param Eclasses: list of E classes to extract.
    :return: array of filtered representations.
    """
    R = []
    for Rc in Rclasses:
        idxs = np.where(labels == Rc)[0]
        R.append(representations[idxs])

    if Eclasses is not None:
        E = []
        for Ec in Eclasses:
            idxs = np.where(labels == Ec)[0]
            E.append(representations[idxs])
        return np.vstack(R), np.vstack(E)
    else:
        return np.vstack(R)


def get_pruned_representation_idxs_by_class(
    labels: np.ndarray, Rclasses: List, classes_to_prune: List, perc_to_discard: float
):
    """
    Discards the specified percentage of points corresponding to the selected classes.
    :param labels: array of representations.
    :param Rclasses: list of R classes to extract.
    :param classes_to_prune: list of classes to prune.
    :param perc_to_discard: percentage of points to discard.
    :return: Pruned array of indices and labels.
    """
    R_idxs = []
    R_labels = []
    for Rc in Rclasses:
        idxs = np.where(labels == Rc)[0]
        if Rc in classes_to_prune:
            to_discard = np.random.choice(
                idxs, size=int(idxs.shape[0] * perc_to_discard), replace=False
            )
            to_keep = np.setdiff1d(idxs, to_discard)
            R_idxs.append(to_keep)
            R_labels.append(np.repeat(Rc, to_keep.shape[0]))
        else:
            R_idxs.append(idxs)
            R_labels.append(np.repeat(Rc, idxs.shape[0]))

    return np.concatenate(R_idxs).astype(int), np.concatenate(R_labels).astype(int)


def get_representation_idxs_by_class(labels: np.ndarray, Rclasses: List):
    """
    Extracts indices of representations of a particular class from given array of labels.
    :param labels: array of labels.
    :param Rclasses: list of classes to extract.
    :return: array of indices corresponding to the specified classes.
    """
    R_idxs = []
    for Rc in Rclasses:
        idxs = np.where(labels == Rc)[0]
        R_idxs.append(idxs)
    return np.concatenate(R_idxs).astype(int)


def plot_query_points(
    query_points_comp_assignment: np.ndarray,
    considered_comp_list: list,
    query_data: dict,
    query_data_name: str,
    input_data_name: str,
    path_to_input_array_labels: str,
    path_to_DCA_comp_stats_logs: str,
    path_to_save: str,
    n_pts_to_plot: int = 5,
):
    """
    Extracts images corresponding to the qDCA analysis.
    :param query_points_comp_assignment: array of component assignments of query points.
    :param considered_comp_list: list of fundamental component indices.
    :param query_data: original query data.
    :param query_data_name: name of the query dataset.
    :param input_data_name: name of the dataset used for DCA.
    :param path_to_input_array_labels: path to component labels of R (and E) points.
    :param path_to_DCA_comp_stats_logs: path to DCA components logs.
    :param path_to_save: path to save results to.
    :param n_pts_to_plot: number of images to plot.
    """
    unique_component_assignments, unique_component_assignments_count = np.unique(
        query_points_comp_assignment[:, 1], return_counts=True
    )

    path_to_dataset = (
        f"representations/contrastive_learning/{input_data_name}_train.pkl"
    )
    with open(os.path.join(path_to_dataset), "rb") as f:
        dataR = pickle.load(f)
        Rrepresentations, Rlabels = dataR["R"], dataR["class_labels"]
        R = get_representations_by_class(
            Rrepresentations, Rlabels, [0, 1, 2, 3, 4, 5, 6]
        )
        trueRidxs = get_representation_idxs_by_class(Rlabels, [0, 1, 2, 3, 4, 5, 6])
        Rimages = dataR["images"][trueRidxs]

    path_to_dataset = (
        f"representations/contrastive_learning/remaining_{input_data_name}_holdout.pkl"
    )
    with open(path_to_dataset, "rb") as f:
        dataE = pickle.load(f)
        Erepresentations, Elabels = dataE["E"], dataE["class_labels"]
        E = get_representations_by_class(
            Erepresentations, Elabels, [0, 1, 2, 3, 4, 5, 6]
        )
        Eimages = dataE["images"]

    # Get the ground truth component labels
    with open(
        os.path.join(path_to_DCA_comp_stats_logs, "components_stats.pkl"), "rb",
    ) as f:
        comp_stats = pickle.load(f)

    comp_label_to_class_label_mapping = {}
    for comp_idx in considered_comp_list:
        R_labels = dataR["class_labels"][comp_stats[comp_idx].Ridx]
        E_labels = dataE["class_labels"][comp_stats[comp_idx].Eidx]
        comp_class_labels, comp_class_labels_counts = np.unique(
            np.concatenate([R_labels, E_labels]), return_counts=True
        )
        # print(comp_idx, comp_class_labels, comp_class_labels_counts)
        comp_class_label = comp_class_labels[np.argmax(comp_class_labels_counts)]
        comp_label_to_class_label_mapping[comp_idx] = comp_class_label

    def comp_to_class_label(comp_class):
        if comp_class in comp_label_to_class_label_mapping.keys():
            return comp_label_to_class_label_mapping[comp_class]
        else:
            return -1

    # Get ground truth class labels of query points
    class_labels_of_query_points = query_data["class_labels"][
        query_points_comp_assignment[:, 0]
    ]
    # Get the assigned component labels of query points
    comp_labels_of_query_points = query_points_comp_assignment[:, 1]
    # Get the ground truth class label of the corresponding assigned component, i.e.,
    # dominating class label of RE points in that component
    comp_class_labels_of_query_points = np.vectorize(comp_to_class_label)(
        comp_labels_of_query_points
    )

    # Out of assigned ones, check how many are assigned to correct components
    assigned_query_points_idxs = np.where(query_points_comp_assignment[:, 1] != -1)[0]
    correctly_assigned_out_of_all_assigned = (
        class_labels_of_query_points[assigned_query_points_idxs]
        == comp_class_labels_of_query_points[assigned_query_points_idxs]
    ).sum()
    perc_correctly_assigned_out_of_all_assigned = (
        correctly_assigned_out_of_all_assigned / len(assigned_query_points_idxs)
    )

    # Out of the ones that should (or not) be assigned, check how many are correctly assigned (or not)
    # For this, transform the class labels of query points that were not in RE to -1 == these should not be assigned
    # This includes assignments to correct components too
    underlying_class_labels_of_query_points = np.vectorize(
        lambda p_class: p_class if p_class in considered_comp_list else -1
    )(class_labels_of_query_points)
    correctly_assigned_out_of_all_that_should_be_assigned = (
        underlying_class_labels_of_query_points == comp_class_labels_of_query_points
    ).sum()
    perc_correctly_assigned_out_of_all_that_should_be_assigned = (
        correctly_assigned_out_of_all_that_should_be_assigned
        / len(query_points_comp_assignment)
    )

    (
        unique_class_labels_of_assigned_query_points,
        class_labels_of_assigned_query_points_counts,
    ) = np.unique(
        class_labels_of_query_points[assigned_query_points_idxs], return_counts=True
    )

    query_points_stats = {
        "unique_component_assignments": unique_component_assignments,
        "unique_component_assignments_count": unique_component_assignments_count,
        "assigned_query_points_idxs": assigned_query_points_idxs,
        "percentage_of_assigned_query_points": len(assigned_query_points_idxs)
        / len(query_points_comp_assignment),
        "class_labels_of_assigned_query_points": class_labels_of_query_points[
            assigned_query_points_idxs
        ],
        "unique_class_labels_of_assigned_query_points": unique_class_labels_of_assigned_query_points,
        "class_labels_of_assigned_query_points_counts": class_labels_of_assigned_query_points_counts,
        "correctly_assigned_out_of_all_that_should_be_assigned": correctly_assigned_out_of_all_that_should_be_assigned,
        "perc_correctly_assigned_out_of_all_that_should_be_assigned": perc_correctly_assigned_out_of_all_that_should_be_assigned,
        "correctly_assigned_out_of_all_assigned": correctly_assigned_out_of_all_assigned,
        "perc_correctly_assigned_out_of_all_assigned": perc_correctly_assigned_out_of_all_assigned,
    }

    with open(
        os.path.join(path_to_save, "logs", f"qDCA_{query_data_name}_stats.pkl",), "wb",
    ) as f:
        pickle.dump(query_points_stats, f)

    input_array_labels = np.load(path_to_input_array_labels)
    R_comp_labels = input_array_labels[: R.shape[0]]
    E_comp_labels = input_array_labels[R.shape[0] :]

    for id in range(1, len(unique_component_assignments)):
        # Plot R examples
        n_pts_to_plot = min(n_pts_to_plot, unique_component_assignments_count[id])
        Ridxs = np.where(R_comp_labels == unique_component_assignments[id])[0]
        Ridxs_sampled = np.random.choice(
            Ridxs, n_pts_to_plot, replace=False if len(Ridxs) >= n_pts_to_plot else True
        )

        Eidxs = np.where(E_comp_labels == unique_component_assignments[id])[0]
        Eidxs_sampled = np.random.choice(
            Eidxs, n_pts_to_plot, replace=False if len(Eidxs) >= n_pts_to_plot else True
        )

        Qidxs = np.where(
            query_points_comp_assignment[:, 1] == unique_component_assignments[id]
        )[0]
        Qidxs_sampled = np.random.choice(
            Qidxs,
            n_pts_to_plot,
            replace=False if len(Qidxs) >= n_pts_to_plot else True,
        )

        plt.figure(3)
        plt.suptitle("R, E, and Q examples")
        for i in range(n_pts_to_plot):
            plt.subplot(3, n_pts_to_plot, i + 1)
            plt.imshow(Rimages[Ridxs_sampled][i])
            plt.axis("off")

            plt.subplot(3, n_pts_to_plot, n_pts_to_plot + i + 1)
            plt.imshow(Eimages[Eidxs_sampled][i])
            plt.axis("off")

            plt.subplot(3, n_pts_to_plot, 2 * n_pts_to_plot + i + 1)
            plt.imshow(query_data["images"][Qidxs_sampled][i])
            plt.axis("off")

        plt.savefig(
            os.path.join(
                path_to_save, "visualization", f"qDCA_{query_data_name}_R_E_Q_comp{id}",
            )
        )
        with open(
            os.path.join(
                path_to_save, "logs", f"qDCA_{query_data_name}_examples_comp{id}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(
                {
                    "Rimages": Rimages[Ridxs_sampled],
                    "Ridx_sampled": Ridxs_sampled,
                    "Eimages": Eimages[Eidxs_sampled],
                    "Eidx_sampled": Eidxs_sampled,
                    "queryimages": query_data["images"][Qidxs_sampled],
                    "queryidx_sampled": Qidxs_sampled,
                },
                f,
            )
