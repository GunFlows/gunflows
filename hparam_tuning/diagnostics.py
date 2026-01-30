#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import numpy as np
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections.abc import Callable, Sequence
import inspect
import matplotlib as mpl

from optuna._experimental import experimental_func
from optuna._imports import try_import
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite
from optuna.visualization._contour import _get_contour_subplot_info
from optuna.visualization._contour import _AxisInfo
from optuna.visualization._contour import _ContourInfo
from optuna.visualization._contour import _PlotValues
from optuna.visualization._contour import _SubContourInfo
from optuna.visualization._utils import _is_reverse_scale
from optuna.visualization.matplotlib._matplotlib_imports import _imports

with try_import() as _optuna_imports:
    import scipy

if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import Colormap
    from optuna.visualization.matplotlib._matplotlib_imports import ContourSet
    from optuna.visualization.matplotlib._matplotlib_imports import plt as _plt_backend

CONTOUR_POINT_NUM = 100

PARAM_LABELS_DEFAULT = {
    "experiment.optim.lr": "Learning rate",
    "experiment.model.nflows": "Number of flows",
}

def _get_best_points(study, sorted_params, n_best: int = 10):
    cat_maps = get_categorical_maps_from_study(study)
    trials = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    filtered = [t for t in trials if all(p in t.params for p in sorted_params)]
    filtered.sort(key=lambda t: t.value)
    best_trials = filtered[:n_best]
    best_points = []
    for t in best_trials:
        d = {}
        for p in sorted_params:
            val = t.params[p]
            if p in cat_maps:
                val = cat_maps[p].get(val, val)
            d[p] = val
        d["z"] = t.value
        d["number"] = t.number
        best_points.append(d)
    return best_points

@experimental_func("2.2.0")
def plot_contour(
    study: Study,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Validation Loss",
    labels: dict[str, str] | None = None,
    cmap: str = "viridis_r",
    fig=None,
    axs=None,
    params_to_exclude: list[str] | None = None,
    plot_title=None,
):
    _imports.check()
    if params is not None:
        completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
        for p in list(params):
            values = set(t.params.get(p) for t in completed_trials if p in t.params)
            if len(values) <= 1:
                params = [x for x in params if x != p]
    info = _get_contour_info(study, params, target, target_name, params_to_exclude=params_to_exclude)
    return _get_contour_plot(info, labels=labels, cmap=cmap, study=study, fig=fig, axs=axs, title=plot_title)

def _get_contour_plot(info: _ContourInfo, labels: dict[str, str] | None = None, cmap: str = "viridis_r", study=None, fig=None, axs=None, title=None):
    sorted_params = info.sorted_params
    sub_plot_infos = info.sub_plot_infos
    target_name = info.target_name
    if title is None:
        title = "Contour Plot"
    if len(sorted_params) <= 1:
        if fig is None or axs is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig, axs
        return fig, ax
    n_params = len(sorted_params)
    plt.style.use("ggplot")
    best_points = _get_best_points(study, sorted_params, n_best=5)
    if n_params == 2:
        if fig is None or axs is None:
            fig, axs = plt.subplots()
        axs.set_title(title)
        cmap_obj = plt.get_cmap(cmap)
        cs = _generate_contour_subplot(sub_plot_infos[0][0], axs, cmap_obj, best_points=best_points)
        if isinstance(cs, ContourSet):
            axcb = fig.colorbar(cs)
            axcb.set_label(target_name)
        if labels:
            xlab = labels.get(sorted_params[0], sorted_params[0])
            ylab = labels.get(sorted_params[1], sorted_params[1])
            axs.set_xlabel(xlab)
            axs.set_ylabel(ylab)
        return fig, axs
    else:
        if fig is None or axs is None:
            fig, axs = plt.subplots(
                n_params,
                n_params,
                figsize=(2, 2),
                gridspec_kw={
                    "right": 0.92,
                    "left": 0.06,
                    "bottom": 0.06,
                    "top": 0.96,
                    "wspace": 0.35,
                    "hspace": 0.4,
                },
            )
        assert isinstance(axs, np.ndarray)
        fig.suptitle(title)
        cmap_obj = plt.get_cmap(cmap)
        cs_list = []
        for x_i in range(len(sorted_params)):
            for y_i in range(len(sorted_params)):
                ax = axs[y_i, x_i]
                if x_i == y_i:
                    _plot_param_histogram(ax, sorted_params[x_i], x_i, sub_plot_infos, labels=labels)
                    continue
                cs = _generate_contour_subplot(sub_plot_infos[y_i][x_i], ax, cmap_obj, best_points=best_points)
                if isinstance(cs, ContourSet):
                    cs_list.append(cs)
                if labels:
                    xlab = labels.get(sorted_params[x_i], sorted_params[x_i])
                    ylab = labels.get(sorted_params[y_i], sorted_params[y_i])
                    ax.set_xlabel(xlab)
                    ax.set_ylabel(ylab)
        if cs_list:
            fig.subplots_adjust(left=0.01, right=0.88, bottom=0.01, top=0.95, wspace=0.6, hspace=0.6)
            cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
            axcb = fig.colorbar(cs_list[0], cax=cbar_ax)
            axcb.set_label(target_name)
        return fig, axs

def _set_cmap(reverse_scale: bool) -> Colormap:
    cmap = "viridis_r" if not reverse_scale else "viridis"
    return plt.get_cmap(cmap)

class _LabelEncoder:
    def __init__(self, explicit_order: list[str] | None = None) -> None:
        self.labels: list[str] = []
        self.explicit_order = explicit_order
    def fit(self, labels: list[str]) -> "_LabelEncoder":
        if self.explicit_order is not None:
            self.labels = self.explicit_order
        else:
            seen = set()
            self.labels = [str(x) for x in labels if not (str(x) in seen or seen.add(str(x)))]
        return self
    def transform(self, labels: list[str]) -> list[int]:
        indices = []
        for label in labels:
            label_str = str(label)
            if label_str not in self.labels:
                try:
                    idx = int(label)
                    if 0 <= idx < len(self.labels):
                        label_str = self.labels[idx]
                except Exception:
                    pass
            if label_str in self.labels:
                indices.append(self.labels.index(label_str))
            else:
                indices.append(-1)
        return indices
    def fit_transform(self, labels: list[str]) -> list[int]:
        return self.fit(labels).transform(labels)
    def get_labels(self) -> list[str]:
        return self.labels
    def get_indices(self) -> list[int]:
        return list(range(len(self.labels)))

def _filter_missing_values(xaxis: _AxisInfo, yaxis: _AxisInfo) -> tuple[list[str | float], list[str | float]]:
    x_values = []
    y_values = []
    for x_value, y_value in zip(xaxis.values, yaxis.values):
        if x_value is not None and y_value is not None:
            x_values.append(x_value)
            y_values.append(y_value)
    return x_values, y_values

def _calculate_axis_data(axis: _AxisInfo, values: Sequence[str | float]) -> tuple[np.ndarray, list[str], list[int], list[int | float]]:
    cat_param_labels: list[str] = []
    cat_param_pos: list[int] = []
    returned_values: Sequence[int | float]
    if axis.is_cat:
        unique_labels = []
        seen = set()
        for v in axis.values:
            if v is not None and v not in seen:
                unique_labels.append(str(v))
                seen.add(v)
        enc = _LabelEncoder(explicit_order=unique_labels)
        enc.fit(list(map(str, filter(lambda value: value is not None, axis.values))))
        returned_values = enc.transform(list(map(str, values)))
        cat_param_labels = enc.get_labels()
        cat_param_pos = enc.get_indices()
    else:
        returned_values = list(map(lambda x: float(x), values))
    if axis.is_log:
        ci = np.logspace(np.log10(axis.range[0]), np.log10(axis.range[1]), CONTOUR_POINT_NUM)
    else:
        ci = np.linspace(axis.range[0], axis.range[1], CONTOUR_POINT_NUM)
    return ci, cat_param_labels, cat_param_pos, list(returned_values)

def _calculate_griddata(info: _SubContourInfo) -> tuple[np.ndarray, _PlotValues, _PlotValues, list[tuple[float | str, float | str, float]]]:
    xaxis = info.xaxis
    yaxis = info.yaxis
    z_values_dict = info.z_values
    x_values = []
    y_values = []
    z_values = []
    for x_value, y_value in zip(xaxis.values, yaxis.values):
        if x_value is not None and y_value is not None:
            x_values.append(x_value)
            y_values.append(y_value)
            x_i = xaxis.indices.index(x_value)
            y_i = yaxis.indices.index(y_value)
            z = z_values_dict[(x_i, y_i)]
            z_values.append(z)
    if len(x_values) == 0 or len(y_values) == 0:
        return np.array([]), _PlotValues([], []), _PlotValues([], []), []
    xi, _, _, transformed_x_values = _calculate_axis_data(xaxis, x_values)
    yi, _, _, transformed_y_values = _calculate_axis_data(yaxis, y_values)
    zi: np.ndarray = np.array([])
    if xaxis.name != yaxis.name:
        zmap = _create_zmap(transformed_x_values, transformed_y_values, z_values, xi, yi)
        zi = _interpolate_zmap(zmap, CONTOUR_POINT_NUM)
    feasible = _PlotValues([], [])
    infeasible = _PlotValues([], [])
    feasible_points_with_z = []
    for x_value, y_value, z, c in zip(transformed_x_values, transformed_y_values, z_values, info.constraints):
        if c:
            feasible.x.append(x_value)
            feasible.y.append(y_value)
            feasible_points_with_z.append((x_value, y_value, z))
        else:
            infeasible.x.append(x_value)
            infeasible.y.append(y_value)
    return zi, feasible, infeasible, feasible_points_with_z

def _generate_contour_subplot(info: _SubContourInfo, ax: Axes, cmap: Colormap, best_points=None, highlight_trials_dicts=None) -> ContourSet | None:
    ax.label_outer()
    if len(info.xaxis.indices) < 2 or len(info.yaxis.indices) < 2:
        return None
    ax.set(xlabel=info.xaxis.name, ylabel=info.yaxis.name)
    ax.set_xlim(info.xaxis.range[0], info.xaxis.range[1])
    ax.set_ylim(info.yaxis.range[0], info.yaxis.range[1])
    x_values, y_values = _filter_missing_values(info.xaxis, info.yaxis)
    xi, x_cat_param_label, x_cat_param_pos, _ = _calculate_axis_data(info.xaxis, x_values)
    yi, y_cat_param_label, y_cat_param_pos, _ = _calculate_axis_data(info.yaxis, y_values)
    if info.xaxis.is_cat:
        ax.set_xticks(x_cat_param_pos)
        ax.set_xticklabels(x_cat_param_label, rotation=60, ha="right")
    else:
        ax.set_xscale("log" if info.xaxis.is_log else "linear")
    if info.yaxis.is_cat:
        ax.set_yticks(y_cat_param_pos)
        ax.set_yticklabels(y_cat_param_label)
    else:
        ax.set_yscale("log" if info.yaxis.is_log else "linear")
    if info.xaxis.name == info.yaxis.name:
        return None
    zi, feasible_plot_values, infeasible_plot_values, feasible_points_with_z = _calculate_griddata(info)
    cs = None
    if len(zi) > 0:
        ax.contour(xi, yi, zi, 15, linewidths=0.5, colors="k")
        cs = ax.contourf(xi, yi, zi, 15, cmap=cmap.reversed())
        def encode_val(val, axis):
            if axis.is_cat:
                unique_labels = []
                seen = set()
                for v in axis.values:
                    if v is not None and v not in seen:
                        unique_labels.append(str(v))
                        seen.add(v)
                enc = _LabelEncoder(explicit_order=unique_labels)
                enc.fit(list(map(str, filter(lambda value: value is not None, axis.values))))
                return enc.transform([str(val)])[0]
            else:
                return float(val)
        ax.scatter(
            [encode_val(x, info.xaxis) for x in feasible_plot_values.x],
            [encode_val(y, info.yaxis) for y in feasible_plot_values.y],
            marker="o",
            c="black",
            s=20,
            edgecolors="grey",
            linewidth=2.0,
        )
        ax.scatter(
            [encode_val(x, info.xaxis) for x in infeasible_plot_values.x],
            [encode_val(y, info.yaxis) for y in infeasible_plot_values.y],
            marker="o",
            c="#cccccc",
            s=20,
            edgecolors="grey",
            linewidth=2.0,
        )
        if best_points is not None:
            x_name = info.xaxis.name
            y_name = info.yaxis.name
            for pt in best_points:
                if x_name in pt and y_name in pt:
                    ax.scatter(
                        [encode_val(pt[x_name], info.xaxis)],
                        [encode_val(pt[y_name], info.yaxis)],
                        marker="o",
                        c="red",
                        s=40,
                        edgecolors="black",
                        linewidth=2.0,
                        zorder=10,
                    )
    return cs

def _plot_param_histogram(ax, param_name, info, sub_plot_infos, labels=None, best_values=None):
    sub_info = sub_plot_infos[0][info]
    axis_info = sub_info.xaxis
    values = [v for v in axis_info.values if v is not None]
    z_dict = {}
    n_params = len(sub_plot_infos[0])
    for i in range(n_params):
        for j in range(n_params):
            if i == j:
                continue
            if param_name == sub_plot_infos[i][j].xaxis.name:
                s_info = sub_plot_infos[i][j]
                for x_value, y_value in zip(s_info.xaxis.values, s_info.yaxis.values):
                    if x_value is not None and y_value is not None:
                        x_i = s_info.xaxis.indices.index(x_value)
                        y_i = s_info.yaxis.indices.index(y_value)
                        z = s_info.z_values.get((x_i, y_i), None)
                        if z is not None:
                            z_dict.setdefault(x_value, []).append(z)
            if param_name == sub_plot_infos[i][j].yaxis.name:
                s_info = sub_plot_infos[i][j]
                for x_value, y_value in zip(s_info.xaxis.values, s_info.yaxis.values):
                    if x_value is not None and y_value is not None:
                        x_i = s_info.xaxis.indices.index(x_value)
                        y_i = s_info.yaxis.indices.index(y_value)
                        z = s_info.z_values.get((x_i, y_i), None)
                        if z is not None:
                            z_dict.setdefault(y_value, []).append(z)
    frame = inspect.currentframe()
    while frame and "cmap" not in frame.f_locals:
        frame = frame.f_back
    cmap_name = frame.f_locals.get("cmap", "viridis_r") if frame else "viridis_r"
    cmap_obj = plt.get_cmap(cmap_name)
    if cmap_name.endswith("_r"):
        cmap_obj = cmap_obj.reversed()
    all_z = [z for zs in z_dict.values() for z in zs]
    if all_z:
        vmin = min(all_z)
        vmax = max(all_z)
    else:
        vmin = 0.0
        vmax = 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if n_params > 1:
        if param_name == "optim.weight_decay":
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")
    if axis_info.is_cat:
        unique_labels = []
        seen = set()
        for v in axis_info.values:
            if v is not None and v not in seen:
                unique_labels.append(str(v))
                seen.add(v)
        enc = _LabelEncoder(explicit_order=unique_labels).fit(list(map(str, values)))
        labels_list = enc.get_labels()
        counts = [values.count(label) for label in labels_list]
        avg_z = [np.mean(z_dict[label]) if label in z_dict else np.nan for label in labels_list]
        bar_colors = [cmap_obj(norm(z)) if not np.isnan(z) else "grey" for z in avg_z]
        ax.bar(labels_list, counts, color=bar_colors, alpha=0.7)
        if info == n_params - 1:
            xticklabels = labels_list
            new_labels = [label if i % 8 == 0 else "" for i, label in enumerate(xticklabels)]
            ax.set_xticklabels(new_labels, rotation=60, ha="right")
        else:
            ax.set_xticklabels([], rotation=60, ha="right")
    else:
        n_bins = 20
        if param_name == "optim.weight_decay":
            min_val = min(values)
            max_val = max(values)
            bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins)
            hist, bins, patches = ax.hist(values, bins=bins, alpha=0.7)
            ax.set_xscale("log")
        elif param_name == "optim.lr":
            min_val = min(values)
            max_val = max(values)
            bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins)
            hist, bins, patches = ax.hist(values, bins=bins, alpha=0.7)
            ax.set_xscale("log")
        else:
            hist, bins, patches = ax.hist(values, bins=n_bins, alpha=0.7)
        bin_avg_z = []
        for i in range(len(bins) - 1):
            bin_vals = [v for v in values if float(bins[i]) <= float(v) < float(bins[i + 1])]
            bin_zs = []
            for v in bin_vals:
                if v in z_dict:
                    bin_zs.extend(z_dict[v])
            if bin_zs:
                avgz = np.mean(bin_zs)
            else:
                avgz = np.nan
            bin_avg_z.append(avgz)
        for patch, avgz in zip(patches, bin_avg_z):
            if not np.isnan(avgz):
                patch.set_facecolor(cmap_obj(norm(avgz)))
            else:
                patch.set_facecolor("grey")
        if info == n_params - 1:
            xticklabels = [f"{b:.2g}" for b in bins]
            new_labels = [label if i % 8 == 0 else "" for i, label in enumerate(xticklabels)]
            ax.set_xticks(bins)
            ax.set_xticklabels(new_labels, rotation=30, ha="right")
        else:
            ax.set_xticklabels([])
    ax.set_title("")
    ax.set_ylabel("Trials")
    if labels and param_name in labels:
        ax.set_xlabel(labels[param_name])
    else:
        ax.set_xlabel(param_name)

def _create_zmap(x_values: Sequence[int | float], y_values: Sequence[int | float], z_values: Sequence[float], xi: np.ndarray, yi: np.ndarray) -> dict[tuple[int, int], float]:
    zmap = {}
    for x, y, z in zip(x_values, y_values, z_values):
        xindex = int(np.argmin(np.abs(xi - x)))
        yindex = int(np.argmin(np.abs(yi - y)))
        zmap[(xindex, yindex)] = z
    return zmap

def _interpolate_zmap(zmap: dict[tuple[int, int], float], contour_plot_num: int) -> np.ndarray:
    a_data = []
    a_row = []
    a_col = []
    b = np.zeros(contour_plot_num**2)
    for x in range(contour_plot_num):
        for y in range(contour_plot_num):
            grid_index = y * contour_plot_num + x
            if (x, y) in zmap:
                a_data.append(1)
                a_row.append(grid_index)
                a_col.append(grid_index)
                b[grid_index] = zmap[(x, y)]
            else:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if 0 <= x + dx < contour_plot_num and 0 <= y + dy < contour_plot_num:
                        a_data.append(1)
                        a_row.append(grid_index)
                        a_col.append(grid_index)
                        a_data.append(-1)
                        a_row.append(grid_index)
                        a_col.append(grid_index + dy * contour_plot_num + dx)
    z = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix((a_data, (a_row, a_col))), b)
    return z.reshape((contour_plot_num, contour_plot_num))

def get_min_trials_from_study(study, n_min=5):
    trials = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    trials.sort(key=lambda t: t.value)
    return [t.number for t in trials[:n_min]]

def get_categorical_maps_from_study(study):
    cat_maps = {}
    for trial in study.trials:
        if trial.state.name == "COMPLETE":
            for param, dist in trial.distributions.items():
                if hasattr(dist, "choices"):
                    cat_maps[param] = {i: v for i, v in enumerate(dist.choices)}
            break
    return cat_maps

def _get_contour_info(
    study: Study,
    params: list[str] | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
    params_to_exclude: list[str] | None = None,
) -> _ContourInfo:
    _check_plot_args(study, target, target_name)
    trials = _filter_nonfinite(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target)
    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if len(trials) == 0:
        sorted_params = []
    elif params is None:
        sorted_params = sorted(all_params)
    else:
        if len(params) <= 1:
            pass
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError(f"Parameter {input_p_name} does not exist in your study.")
        sorted_params = sorted(set(params))
    if params_to_exclude is not None:
        sorted_params = [p for p in sorted_params if p not in params_to_exclude]
    sub_plot_infos: list[list[_SubContourInfo]]
    if len(sorted_params) == 2:
        x_param = sorted_params[0]
        y_param = sorted_params[1]
        sub_plot_info = _get_contour_subplot_info(study, trials, x_param, y_param, target)
        sub_plot_infos = [[sub_plot_info]]
    else:
        sub_plot_infos = []
        for i, y_param in enumerate(sorted_params):
            sub_plot_infos.append([])
            for x_param in sorted_params:
                sub_plot_info = _get_contour_subplot_info(study, trials, x_param, y_param, target)
                sub_plot_infos[i].append(sub_plot_info)
    reverse_scale = _is_reverse_scale(study, target)
    return _ContourInfo(
        sorted_params=sorted_params,
        sub_plot_infos=sub_plot_infos,
        reverse_scale=reverse_scale,
        target_name=target_name,
    )

def _compute_param_importance_scores(study: Study) -> dict[str, float]:
    trials = _filter_nonfinite(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    if not trials:
        return {}
    param_names = {p_name for t in trials for p_name in t.params.keys()}
    scores = {}
    for param_name in param_names:
        try:
            params = [t.params[param_name] for t in trials if param_name in t.params]
            vals = [t.values[0] for t in trials if param_name in t.params and t.values is not None]
            if len(params) < 3:
                continue
            arr_x = np.array(params, dtype=float)
            arr_y = np.array(vals, dtype=float)
            if np.all(arr_x == arr_x[0]):
                continue
            if np.all(arr_y == arr_y[0]):
                continue
            R = np.corrcoef(arr_x, arr_y)
            score = float(abs(R[0, 1]))
            if not np.isnan(score):
                scores[param_name] = score
        except Exception:
            continue
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return scores

def _plot_param_importances_bar(scores: dict[str, float], param_labels: dict[str, str] | None, outpath: str) -> None:
    if not scores:
        return
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    params, importance_scores = zip(*sorted_items)
    if param_labels is not None:
        display_params = [param_labels.get(p, p) for p in params]
    else:
        display_params = list(params)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(display_params, importance_scores)
    ax.set_xlabel("Importance score (|corr|)")
    ax.set_title("Hyperparameter importances")
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def save_stage_diagnostics(study_name: str, db_path: str, stage: str, param_labels: dict[str, str] | None = None, params_to_exclude: list[str] | None = None) -> None:
    if param_labels is None:
        param_labels = PARAM_LABELS_DEFAULT
    if params_to_exclude is None:
        params_to_exclude = []
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    base_dir = os.path.dirname(db_path)
    outdir = os.path.join(base_dir, "figs")
    os.makedirs(outdir, exist_ok=True)
    if len([t for t in study.trials if t.state == TrialState.COMPLETE]) == 0:
        return
    fig, axs = plot_contour(
        study,
        labels=param_labels,
        params_to_exclude=params_to_exclude,
        plot_title=f"{study_name} – {stage}",
    )
    fig.set_size_inches(25, 20)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "contour.png"), dpi=300)
    plt.close(fig)
    scores = _compute_param_importance_scores(study)
    imp_path = os.path.join(outdir, "hp_importance.png")
    _plot_param_importances_bar(scores, param_labels, imp_path)

def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: diagnostics.py STUDY_NAME DB_PATH STAGE")
    study_name = sys.argv[1]
    db_path = sys.argv[2]
    stage = sys.argv[3]
    save_stage_diagnostics(study_name, db_path, stage)

if __name__ == "__main__":
    main()
