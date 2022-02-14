import pickle
import numpy as np
import os


def _analyze_query_point_assignment(
    query_data_dict: dict,
    init_Rdata_dict: dict,
    init_Edata_dict: dict,
    num_R: int,
    query_point_assignment_array: np.ndarray,
    root: str,
    n_points_to_copy=50,
):
    """
    Analyzes and visualizes qDCA results.
    :param query_data_dict: raw query data.
    :param init_Rdata_dict: raw R data.
    :param init_Edata_dict: raw E data.
    :param num_R: total number of R points.
    :param query_point_assignment_array: query point assignments results.
    :param root: root directory of the experiment.
    :param n_points_to_copy: number of images to save.
    :return: accuracy of qDCA assignments; list of (R, query) points with same label;
    list of (R, query) points with different label
    """
    true_query_data_labels = query_data_dict["labels"]
    assigned_R = query_point_assignment_array[
        query_point_assignment_array[:, 1] < num_R, 1
    ]
    assigned_E = query_point_assignment_array[
        query_point_assignment_array[:, 1] >= num_R, 1
    ]
    assigned_R_labels = init_Rdata_dict["labels"][assigned_R]
    assigned_E_labels = init_Edata_dict["labels"][assigned_E - num_R]

    assigned_query_data_labels = np.empty(
        shape=query_point_assignment_array.shape[0]
    ).astype(np.int32)
    assigned_query_data_labels[
        query_point_assignment_array[:, 1] < num_R
    ] = assigned_R_labels
    assigned_query_data_labels[
        query_point_assignment_array[:, 1] >= num_R
    ] = assigned_E_labels

    accuracy = (
        true_query_data_labels == assigned_query_data_labels
    ).sum() / assigned_query_data_labels.shape[0]

    same_label_idx = np.where(true_query_data_labels == assigned_query_data_labels)[0]
    wrong_label_idx = np.where(true_query_data_labels != assigned_query_data_labels)[0]

    correct_pairs = []
    for i in query_point_assignment_array[same_label_idx]:
        query_idx, init_idx = i
        if init_idx < num_R:
            correct_pairs.append(
                [
                    query_data_dict["paths"].astype(object)[query_idx],
                    init_Rdata_dict["paths"].astype(object)[init_idx],
                    query_data_dict["labels"][query_idx],
                    init_Rdata_dict["labels"][init_idx],
                ]
            )
        else:
            correct_pairs.append(
                [
                    query_data_dict["paths"].astype(object)[query_idx],
                    init_Edata_dict["paths"].astype(object)[init_idx - num_R],
                    query_data_dict["labels"][query_idx],
                    init_Edata_dict["labels"][init_idx - num_R],
                ]
            )

    wrong_pairs = []
    for i in query_point_assignment_array[wrong_label_idx]:
        query_idx, init_idx = i
        if init_idx < num_R:
            wrong_pairs.append(
                [
                    query_data_dict["paths"].astype(object)[query_idx],
                    init_Rdata_dict["paths"].astype(object)[init_idx],
                    query_data_dict["labels"][query_idx],
                    init_Rdata_dict["labels"][init_idx],
                ]
            )
        else:
            wrong_pairs.append(
                [
                    query_data_dict["paths"].astype(object)[query_idx],
                    init_Edata_dict["paths"].astype(object)[init_idx - num_R],
                    query_data_dict["labels"][query_idx],
                    init_Edata_dict["labels"][init_idx - num_R],
                ]
            )

    with open(
        os.path.join(root, "logs", "analyzed_query_point_assignments.pkl"), "wb"
    ) as f:
        pickle.dump(
            {
                "accuracy": accuracy,
                "same_label_idx": same_label_idx,
                "wrong_label_idx": wrong_label_idx,
                "correct_pairs": correct_pairs,
                "wrong_pairs": wrong_pairs,
                "query_point_assignment_array": query_point_assignment_array,
            },
            f,
        )

    same_label_image_path = os.path.join(root, "visualization", "same_label_images")
    wrong_label_image_path = os.path.join(root, "visualization", "wrong_label_images")
    if not os.path.exists(wrong_label_image_path):
        os.mkdir(wrong_label_image_path)

    if not os.path.exists(same_label_image_path):
        os.mkdir(same_label_image_path)

    for i in range(n_points_to_copy):
        query_image_path, init_image_path, query_label, init_label = correct_pairs[i]
        path_to_copy = os.path.join(
            same_label_image_path,
            "i{0}_init_image_querylabel{1}_initlabel{2}.png".format(
                str(i), str(query_label), str(init_label)
            ),
        )

        os.system("cp {0} {1}".format(init_image_path, path_to_copy))

        path_to_copy2 = os.path.join(
            same_label_image_path,
            "i{0}_query_image_querylabel{1}_initlabel{2}.png".format(
                str(i), str(query_label), str(init_label)
            ),
        )
        os.system("cp {0} {1}".format(query_image_path, path_to_copy2))

        (
            w_query_image_path,
            w_init_image_path,
            w_query_label,
            w_init_label,
        ) = wrong_pairs[i]
        path_to_copy_w = os.path.join(
            wrong_label_image_path,
            "i{0}_init_image_querylabel{1}_initlabel{2}.png".format(
                str(i), str(w_query_label), str(w_init_label)
            ),
        )
        os.system("cp {0} {1}".format(w_init_image_path, path_to_copy_w))

        path_to_copy_w2 = os.path.join(
            wrong_label_image_path,
            "i{0}_query_image_querylabel{1}_initlabel{2}.png".format(
                i, w_query_label, w_init_label
            ),
        )
        os.system("cp {0} {1}".format(w_query_image_path, path_to_copy_w2))

    return accuracy, correct_pairs, wrong_pairs


def _generate_query_sets(version: str, N: int = 5000):
    """
    Generates query sets for qDCA experiment in Section 4.3.
    :param version: either version1 (dogs vs kitchen utils) or version2 (random).
    :param N: number of points to sample for R used in DCA.
    """
    with open(f"representations/vgg16/{version}/Rfeatures.pkl", "rb") as f:
        Rdata_v1 = pickle.load(f)

    with open(f"representations/vgg16/{version}/Efeatures.pkl", "rb") as f:
        Edata_v1 = pickle.load(f)

    init_Ridxs = np.random.choice(
        np.arange(len(Rdata_v1["feat_lin1"])), size=N, replace=False
    )
    query_Ridxs = np.setdiff1d(np.arange(len(Rdata_v1["feat_lin1"])), init_Ridxs)

    init_Eidxs = np.random.choice(
        np.arange(len(Edata_v1["feat_lin1"])), size=N, replace=False
    )
    query_Eidxs = np.setdiff1d(np.arange(len(Edata_v1["feat_lin1"])), init_Eidxs)

    with open(f"representations/vgg16/{version}/sampled_Rfeatures.pkl", "wb") as f:
        pickle.dump(
            {
                "feat_lin1": Rdata_v1["feat_lin1"][init_Ridxs],
                "feat_lin2": Rdata_v1["feat_lin2"][init_Ridxs],
                "labels": Rdata_v1["labels"][init_Ridxs],
                "paths": np.array(Rdata_v1["paths"])[init_Ridxs],
                "init_Ridx": init_Ridxs,
                "query_Ridx": query_Ridxs,
            },
            f,
        )

    with open(f"representations/vgg16/{version}/sampled_Efeatures.pkl", "wb") as f:
        pickle.dump(
            {
                "feat_lin1": Edata_v1["feat_lin1"][init_Eidxs],
                "feat_lin2": Edata_v1["feat_lin2"][init_Eidxs],
                "labels": Edata_v1["labels"][init_Eidxs],
                "paths": np.array(Edata_v1["paths"])[init_Eidxs],
                "init_Eidx": init_Eidxs,
                "query_Eidx": query_Eidxs,
            },
            f,
        )

    with open(f"representations/vgg16/{version}/query_features.pkl", "wb") as f:
        pickle.dump(
            {
                "feat_lin1": np.concatenate(
                    [
                        Rdata_v1["feat_lin1"][query_Ridxs],
                        Edata_v1["feat_lin1"][query_Eidxs],
                    ]
                ),
                "feat_lin2": np.concatenate(
                    [
                        Rdata_v1["feat_lin2"][query_Ridxs],
                        Edata_v1["feat_lin2"][query_Eidxs],
                    ]
                ),
                "labels": np.concatenate(
                    [Rdata_v1["labels"][query_Ridxs], Edata_v1["labels"][query_Eidxs]]
                ),
                "paths": np.concatenate(
                    [
                        np.array(Rdata_v1["paths"])[query_Ridxs],
                        np.array(Edata_v1["paths"])[query_Eidxs],
                    ]
                ),
                "init_Eidxs": init_Eidxs,
                "query_Eidxs": query_Eidxs,
                "init_Ridxs": init_Ridxs,
                "query_Ridxs": query_Ridxs,
            },
            f,
        )
