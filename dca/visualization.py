from dca.DCA import DelaunayGraph
from dca.schemes import DelaunayGraphVisualizer
import numpy as np
import os
import matplotlib as mpl

if not "DISPLAY" in os.environ:
    print("no display found. Using non-interactive Agg backend")
    mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from typing import Optional


# -------------------------------------------------------------------------- #
# Matplotlib settings
# -------------------------------------------------------------------------- #
SMALL_SIZE = 12
MEDIUM_SIZE = 15

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=SMALL_SIZE)  # fontsize of the figure title

R_color, E_color = "C0", "C1"

# -------------------------------------------------------------------------- #
# DelaunayGeomCA: visualization
# -------------------------------------------------------------------------- #
def get_color(edge, num_R):
    """
    Gets the color of the edge.
    :param edge: edge given as a list of two indices.
    :param num_R: number of R points in the graph.
    :return: color of the edge and its zorder
    """
    R_color, E_color = "C0", "C1"
    edge = sorted(edge)
    if edge[0] < num_R:
        if edge[1] >= num_R:
            comp_color = "gray"
            zorder = 10
        else:
            comp_color = R_color
            zorder = 5
    else:
        comp_color = E_color
        zorder = 5
    return comp_color, zorder


def _plot_Delaunay_graph(
    G_visualizer: DelaunayGraphVisualizer,
    edges: np.ndarray,
    filename: str,
    root: str,
    vertices: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    figsize: tuple = (5, 5),
    keep_range: bool = False,
):
    """
    Plots a Delaunay graph.
    :param G_visualizer: DelaunayGraphVisualizer object.
    :param edges: array of edges to plot.
    :param filename: filename of the image.
    :param root: root directory of the experiment.
    :param vertices: array of vertices to plot.
    :param labels: array of vertex labels.
    :param figsize: size of the figure.
    :param keep_range: whether to remember current xlim and ylim.
    :return: xlim and ylim if keep_range else None
    """
    input_data = G_visualizer.get_input_array_data()
    Rplot_kwds = {"alpha": 0.7, "s": 50, "linewidths": 0}
    Eplot_kwds = {"alpha": 0.7, "s": 50, "linewidths": 0, "marker": "X"}
    Rvertices, Evertices = (
        input_data[: G_visualizer.num_R],
        input_data[G_visualizer.num_R :],
    )

    if vertices is not None:
        Rcolors = np.empty(shape=G_visualizer.num_R).astype(str)
        Rcolors[vertices[vertices < G_visualizer.num_R].astype(int)] = R_color
        Rcolors[
            np.setdiff1d(
                np.arange(G_visualizer.num_R), vertices[vertices < G_visualizer.num_R]
            )
        ] = "gray"

        Ecolors = np.empty(shape=G_visualizer.num_E).astype(str)
        Ecolors[
            vertices[vertices >= G_visualizer.num_R].astype(int) - G_visualizer.num_R
        ] = E_color
        Ecolors[
            np.setdiff1d(
                np.arange(G_visualizer.num_E).astype(int),
                vertices[vertices >= G_visualizer.num_R].astype(int)
                - G_visualizer.num_R,
            )
        ] = "gray"

        if labels is not None:
            labels = labels[vertices]
    else:
        Rcolors = np.repeat(R_color, G_visualizer.num_R).astype(str)
        Ecolors = np.repeat(E_color, G_visualizer.num_E).astype(str)

    plt.figure(figsize=figsize)
    plt.clf()
    # Plot vertices
    if labels is not None:
        plt.scatter(
            Rvertices.T[0], Rvertices.T[1], c=labels[: G_visualizer.num_R], **Rplot_kwds
        )
        plt.scatter(
            Evertices.T[0], Evertices.T[1], c=labels[G_visualizer.num_R :], **Eplot_kwds
        )

    else:
        plt.scatter(Rvertices.T[0], Rvertices.T[1], color=Rcolors, **Rplot_kwds)
        plt.scatter(Evertices.T[0], Evertices.T[1], color=Ecolors, **Eplot_kwds)

    # Plot edges
    # draw edges of correct color
    for e in edges:
        e0, e1 = int(e[0]), int(e[1])
        start = (
            Rvertices[e0]
            if e0 < G_visualizer.num_R
            else Evertices[e0 - G_visualizer.num_R]
        )
        end = (
            Rvertices[e1]
            if e1 < G_visualizer.num_R
            else Evertices[e1 - G_visualizer.num_R]
        )
        color, zorder = get_color(e, G_visualizer.num_R)
        plt.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            "-",
            linewidth=1.0,
            color=color,
            zorder=zorder,
        )
    plt.axis("off")
    plt.tight_layout()
    save_path = os.path.join(root, filename)
    if keep_range:
        assert G_visualizer.xlim is not None and G_visualizer.ylim is not None
        plt.xlim(*G_visualizer.xlim)
        plt.ylim(*G_visualizer.ylim)
    current_xlim = plt.xlim()
    current_ylim = plt.ylim()
    plt.savefig(save_path)
    plt.clf()
    plt.close()
    return current_xlim, current_ylim


def _plot_Delaunay_graph_colorbar(
    G_visualizer: DelaunayGraphVisualizer,
    edges: np.ndarray,
    distances: np.ndarray,
    filename: str,
    root: str,
    figsize: tuple = (5, 5),
):
    """
    Plots a Delaunay graph with colored edge lengths.
    :param G_visualizer: DelaunayGraphVisualizer object.
    :param edges: array of edges to plot.
    :param distances: array of edge lengths.
    :param filename: filename of the image.
    :param root: root directory of the experiment.
    :param figsize: size of the figure.
    """
    input_data = G_visualizer.get_input_array_data()
    Rplot_kwds = {"alpha": 0.7, "s": 50, "linewidths": 0}
    Eplot_kwds = {"alpha": 0.7, "s": 50, "linewidths": 0, "marker": "X"}
    Rvertices, Evertices = (
        input_data[: G_visualizer.num_R],
        input_data[G_visualizer.num_R :],
    )

    # Plot vertices
    # Plot edges, color = distance
    plt.figure(figsize=figsize)
    axis = plt.gca()
    segments, colors = [], []
    for e in edges:
        start = (
            Rvertices[e[0]]
            if e[0] < G_visualizer.num_R
            else Evertices[e[0] - G_visualizer.num_R]
        )
        end = (
            Rvertices[e[1]]
            if e[1] < G_visualizer.num_R
            else Evertices[e[1] - G_visualizer.num_R]
        )
        segments.append([start, end])
        colors.append(distances[e[0], e[1]])
    colors = np.array(colors)
    lc = LineCollection(segments, cmap="viridis_r")
    lc.set_array(colors)
    axis.add_artist(lc)
    cb = plt.colorbar(lc, ax=axis)
    cb.ax.set_ylabel(filename)

    Rcolors = np.repeat(R_color, G_visualizer.num_R).astype(str)
    Ecolors = np.repeat(E_color, G_visualizer.num_E).astype(str)
    axis.scatter(Rvertices.T[0], Rvertices.T[1], color=Rcolors, **Rplot_kwds)
    axis.scatter(Evertices.T[0], Evertices.T[1], color=Ecolors, **Eplot_kwds)
    axis.set_xlim(np.min(input_data[:, 0]) - 1, np.max(input_data[:, 0]) + 1)
    axis.set_ylim(np.min(input_data[:, 1]) - 1, np.max(input_data[:, 1]) + 1)
    save_path = os.path.join(root, filename)
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def _plot_isolated_components(
    G: DelaunayGraph, G_visualizer: DelaunayGraphVisualizer, root: str
):
    """
    Plots outliers in a distilled Delaunay graph.
    :param G: Delaunay graph.
    :param G_visualizer: DelaunayGraphVisualizer object.
    :param root: root directory of the experiment.
    """
    # Get outliers
    n_components = len(G.comp_stats)
    R_outliers, E_outliers = [], []
    for i in range(G.first_trivial_component_idx, n_components):
        if G.comp_stats[i].Ridx.size == 1:
            R_outliers.append(G.comp_stats[i].Ridx.item())
        if G.comp_stats[i].Eidx.size == 1:
            E_outliers.append(G.comp_stats[i].Eidx.item())
    vertices = np.concatenate([R_outliers, np.array(E_outliers) + G.num_R])
    _plot_Delaunay_graph(
        G_visualizer,
        edges=np.array([]),
        filename="components_isolated",
        root=root,
        vertices=vertices,
        keep_range=True,
    )


def _plot_RE_components_quality(
    G: DelaunayGraph,
    root: str,
    annotate_largest: bool = True,
    min_comp_size: int = 0,
    display_smaller: bool = False,
    figsize: tuple = (10, 5),
):
    """
    Visualizes components quality as a scatter plot.
    :param G: Delaunay graph.
    :param root: root directory of the experiment.
    :param annotate_largest: if annotate the size (in percentage) of the largest component.
    :param min_comp_size: minimum size (number of vertices) of the components to visualize.
    :param display_smaller: if display aggregated components with size smaller than min_comp_size.
    :param figsize: size of the plot.
    """
    n_comp = len(G.comp_stats)
    total_n_pts = G.num_R + G.num_E
    max_score, last_display_comp = 0, 0
    small_R_comp, small_E_comp, small_RE_comp = 0, 0, 0
    quality_scores, ticks_labels = [], []
    fig, ax = plt.subplots(figsize=figsize)
    for comp_id in range(n_comp):
        compR = G.comp_stats[comp_id].Ridx
        compE = G.comp_stats[comp_id].Eidx
        comp_n_points = len(compR) + len(compE)
        if comp_n_points >= min_comp_size:
            comp_quality = np.round(G.comp_stats[comp_id].comp_quality, 2)
            max_score = max(max_score, comp_quality)
            last_display_comp = comp_id + 1
            quality_scores.append(comp_quality)
            if len(compR) != 0:
                if len(compE) != 0:
                    comp_color = "gray"
                else:
                    comp_color = R_color
            else:
                comp_color = E_color

            ax.scatter(
                comp_id,
                comp_quality,
                c=comp_color,
                linestyle="--",
                s=1000 * (comp_n_points) / total_n_pts,
                alpha=0.8,
                zorder=10,
            )
        else:
            if len(compR) != 0:
                if len(compE) != 0:
                    small_RE_comp += 1
                else:
                    small_R_comp += 1
            else:
                small_E_comp += 1

    if min_comp_size > 0 and display_smaller:
        if small_RE_comp + small_R_comp + small_E_comp > 0:
            ticks_labels = [last_display_comp]

        if small_RE_comp > 0:
            r = last_display_comp + 2 * len(ticks_labels)
            ticks_labels.append(ticks_labels[-1] + small_RE_comp)
            ax.axvspan(r - 2, r, alpha=0.5, color="gray")

        if small_R_comp > 0:
            r = last_display_comp + 2 * len(ticks_labels)
            ticks_labels.append(ticks_labels[-1] + small_R_comp)
            ax.axvspan(r - 2, r, alpha=0.5, color=R_color)

        if small_E_comp > 0:
            r = last_display_comp + 2 * len(ticks_labels)
            ticks_labels.append(ticks_labels[-1] + small_E_comp)
            ax.axvspan(r - 2, r, alpha=0.5, color=E_color)

    # Annotate the largest component
    if annotate_largest:
        largest_comp_size = len(G.comp_stats[0].Ridx) + len(G.comp_stats[0].Eidx)
        ax.annotate(
            round(largest_comp_size / total_n_pts, 2),
            xy=(0, G.comp_stats[0].comp_quality + 0.03),
            ha="center",
            va="bottom",
            color="k",
        )
        if max_score == 0:
            ax.plot(0, G.comp_stats[0].comp_quality, "kX")

    ax.plot(
        np.arange(last_display_comp),
        quality_scores,
        color="gray",
        linestyle="--",
        alpha=0.5,
        zorder=0,
    )
    displayed_ticks = np.arange(
        last_display_comp, step=max(int(last_display_comp / 10), 1)
    )
    if min_comp_size == 0:
        ax.set_xticks(displayed_ticks)
        ax.set_xticklabels(displayed_ticks)
    else:
        new_ticks = np.arange(
            last_display_comp, last_display_comp + len(ticks_labels) * 2, 2
        )
        ax.set_xticks(np.concatenate([displayed_ticks, new_ticks]))
        ax.set_xticklabels(list(displayed_ticks) + ticks_labels)
        max_score = 1.0 if max_score == 0 else max_score
    ax.set_ylim((-0.05, max_score + 0.1))
    ax.set_yticks(np.arange(0, max_score + 0.1, 0.1))

    # ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("component index")
    ax.set_ylabel("component quality")
    legend_elements = [
        Line2D(
            [0],
            [0],
            markerfacecolor=R_color,
            markersize=10,
            label="R",
            marker="o",
            color="w",
        ),
        Line2D(
            [0],
            [0],
            markerfacecolor=E_color,
            markersize=10,
            label="E",
            marker="o",
            color="w",
        ),
        Line2D(
            [0],
            [0],
            markerfacecolor="gray",
            markersize=10,
            label="mix",
            marker="o",
            color="w",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        ncol=len(legend_elements),
        loc="upper center",
        framealpha=0.5,
    )
    name = "component_quality_min_size{0}_annotated{1}_displaysmaller{2}".format(
        min_comp_size, int(annotate_largest), int(display_smaller)
    )
    path = os.path.join(root, name)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()


def _plot_RE_components_consistency(
    G: DelaunayGraph,
    root: str,
    annotate_largest: bool = True,
    min_comp_size: int = 0,
    display_smaller: bool = False,
    figsize: tuple = (10, 5),
):
    """
    Visualizes components consistency as a scatter plot.
    :param G: Delaunay graph.
    :param root: root directory of the experiment.
    :param annotate_largest: if annotate the size (in percentage) of the largest component.
    :param min_comp_size: minimum size (number of vertices) of the components to visualize.
    :param display_smaller: if display aggregated components with size smaller than min_comp_size.
    :param figsize: size of the plot.
    """
    n_comp = len(G.comp_stats)
    total_n_pts = G.num_R + G.num_E
    max_score, last_display_comp = 0, 0
    small_R_comp, small_E_comp, small_RE_comp = 0, 0, 0
    consistency_scores, ticks_labels = [], []

    fig, ax = plt.subplots(figsize=figsize)
    for comp_id in range(n_comp):
        compR = G.comp_stats[comp_id].Ridx
        compE = G.comp_stats[comp_id].Eidx
        comp_n_points = len(compR) + len(compE)
        if comp_n_points >= min_comp_size:
            comp_consistency = np.round(G.comp_stats[comp_id].comp_consistency, 2)
            max_score = max(max_score, comp_consistency)
            last_display_comp = comp_id + 1
            consistency_scores.append(comp_consistency)
            if len(compR) != 0:
                if len(compE) != 0:
                    comp_color = "gray"
                else:
                    comp_color = R_color
            else:
                comp_color = E_color

            ax.scatter(
                comp_id,
                comp_consistency,
                c=comp_color,
                linestyle="--",
                s=1000 * (comp_n_points) / total_n_pts,
                alpha=0.8,
                zorder=10,
            )
        else:
            if len(compR) != 0:
                if len(compE) != 0:
                    small_RE_comp += 1
                else:
                    small_R_comp += 1
            else:
                small_E_comp += 1

    if min_comp_size > 0 and display_smaller:
        if small_RE_comp + small_R_comp + small_E_comp > 0:
            ticks_labels = [last_display_comp]

        if small_RE_comp > 0:
            r = last_display_comp + 2 * len(ticks_labels)
            ticks_labels.append(ticks_labels[-1] + small_RE_comp)
            ax.axvspan(r - 2, r, alpha=0.5, color="gray")

        if small_R_comp > 0:
            r = last_display_comp + 2 * len(ticks_labels)
            ticks_labels.append(ticks_labels[-1] + small_R_comp)
            ax.axvspan(r - 2, r, alpha=0.5, color=R_color)

        if small_E_comp > 0:
            r = last_display_comp + 2 * len(ticks_labels)
            ticks_labels.append(ticks_labels[-1] + small_E_comp)
            ax.axvspan(r - 2, r, alpha=0.5, color=E_color)

    # Annotate the largest component
    if annotate_largest:
        largest_comp_size = len(G.comp_stats[0].Ridx) + len(G.comp_stats[0].Eidx)
        ax.annotate(
            round(largest_comp_size / total_n_pts, 2),
            xy=(0, G.comp_stats[0].comp_consistency + 0.03),
            ha="center",
            va="bottom",
            color="k",
        )
        if max_score == 0:
            ax.plot(0, G.comp_stats[0].comp_consistency, "kX")

    ax.plot(
        np.arange(last_display_comp),
        consistency_scores,
        color="gray",
        linestyle="--",
        alpha=0.5,
        zorder=0,
    )
    displayed_ticks = np.arange(
        last_display_comp, step=max(int(last_display_comp / 10), 1)
    )
    if min_comp_size == 0:
        ax.set_xticks(displayed_ticks)
        ax.set_xticklabels(displayed_ticks)
    else:
        new_ticks = np.arange(
            last_display_comp, last_display_comp + len(ticks_labels) * 2, 2
        )
        ax.set_xticks(np.concatenate([displayed_ticks, new_ticks]))
        ax.set_xticklabels(list(displayed_ticks) + ticks_labels)
        max_score = 1.0 if max_score == 0 else max_score
    ax.set_ylim((-0.05, max_score + 0.1))
    ax.set_yticks(np.arange(0, max_score + 0.1, 0.1))

    # ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("component index")
    ax.set_ylabel("component consistency")
    legend_elements = [
        Line2D(
            [0],
            [0],
            markerfacecolor=R_color,
            markersize=10,
            label="R",
            marker="o",
            color="w",
        ),
        Line2D(
            [0],
            [0],
            markerfacecolor=E_color,
            markersize=10,
            label="E",
            marker="o",
            color="w",
        ),
        Line2D(
            [0],
            [0],
            markerfacecolor="gray",
            markersize=10,
            label="mix",
            marker="o",
            color="w",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        ncol=len(legend_elements),
        loc="upper center",
        framealpha=0.5,
    )
    name = "component_consistency_min_size{0}_annotated{1}_displaysmaller{2}".format(
        min_comp_size, int(annotate_largest), int(display_smaller)
    )
    path = os.path.join(root, name)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()
