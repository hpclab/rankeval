"""
This package provides visualizations for several effectiveness analysis focused on assessing
the performance of the models in terms of accuracy.

"""

from __future__ import print_function
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def init_plot_style():
    """
    Initialize plot style for RankEval visualization utilities.
    Returns
    -------

    """
    plt.style.use("seaborn-notebook")
    sns.set_palette("deep")


def plot_model_performance(performance, compare="models", show_values=False):
    """
    This method plots the results obtained from the model_performance analysis.

    Parameters
    ----------
    performance: xarray
        The xarray obtained after computing model_performance.
    compare: string
        The compare parameter indicates what elements to compare between
        each other.
        Accepted values are 'models' or 'metrics'.
    show_values: bool
        If show values is True, we add numeric labels on each bar in the plot
        with the rounded value to which the bar corresponds. The default is
        False and shows no values on the bars.

    Returns
    -------
    fig_list : list
        The list of figures.
    """

    if compare not in ["models", "metrics"]:
        raise RuntimeError("Please select compare method from "
                           "['models', 'metrics']")

    fig_list = []

    for dataset in performance.coords['dataset'].values:
        fig, axes = plt.subplots(figsize=(8, 8))
        num_metrics = len(performance.coords['metric'].values)
        num_models = len(performance.coords['model'].values)
        max_width = .95

        if compare == "models":
            width = max_width / num_models
            ind = np.arange(num_metrics)

            for i, model in enumerate(performance.coords['model'].values):
                metrics = performance.sel(dataset=dataset, model=model)
                a = axes.bar(ind + (i * width), metrics.values, width,
                             align="center", zorder=3)

                # add column values on the bars
                if show_values:
                    for j, bar in enumerate(a):
                        coords = [bar.get_height(), bar.get_width()]
                        axes.text(j + (i * width), coords[0],
                                  round(metrics.values[j], 3),
                                  ha='center', va='bottom', zorder=3)
                        
            axes.set_xticks(ind - width/2. + max_width / 2.)
            axes.set_xticklabels(performance.coords['metric'].values)
            axes.legend(performance.coords['model'].values)
            
        elif compare == "metrics":
            width = max_width / num_metrics
            ind = np.arange(num_models)

            for i, metric in enumerate(performance.coords['metric'].values):
                models = performance.sel(dataset=dataset, metric=metric)
                a = axes.bar(ind + (i * width), models.values, width,
                             align="center", zorder=3)

                # add column values on the bars
                if show_values:
                    for j, bar in enumerate(a):
                        coords = [bar.get_height(), bar.get_width()]
                        axes.text(j + (i * width), coords[0],
                                  round(models.values[j],3),
                                  ha='center', va='bottom', zorder=3)
                        
            axes.set_xticks(ind - width/2. + max_width / 2.)
            axes.set_xticklabels(performance.coords['model'].values)
            axes.legend(performance.coords['metric'].values)

        axes.set_ylabel("Metric Score")
        axes.set_title(performance.name + " for " + dataset.name)
        axes.yaxis.grid(True, zorder=0, ls="--")
        
        y_max = np.ceil(performance.values.max()*1.4*10.)/10.
        axes.set_ylim([0, y_max])
        plt.tight_layout()

        fig_list.append(fig)

    return fig_list


def plot_tree_wise_performance(performance, compare="models"):
    """
    This method plots the results obtained from the tree_wise_performance
    analysis.

    Parameters
    ----------
    performance: xarray
        The xarray obtained after computing tree_wise_performance.
    compare: string
        The compare parameter indicates what elements to compare between
        each other. The default is 'models'.
        Accepted values are 'models' or 'metrics' or 'datasets'.

    Returns
    -------
    fig_list : list
        The list of figures.
    """

    if compare not in ["models", "metrics", "datasets"]:
        raise RuntimeError("Please select compare method from " +
                           "['models', 'metrics', 'datasets']")

    fig_list = []

    if compare == "metrics":
        for dataset in performance.coords['dataset'].values:
            fig, axes = plt.subplots(len(performance.coords['model'].values),
                                     sharex=True, squeeze=False, figsize=(8, 8))
            for i, model in enumerate(performance.coords['model'].values):
                for j, metric in enumerate(performance.coords['metric'].values):
                    k_values = performance.indexes['k'].values
                    metric_values = performance.sel(dataset=dataset,
                                               model=model,
                                               metric=metric)
                    axes[i, 0].plot(k_values, metric_values.values, label=metric)
                    max_k = np.nanargmax(metric_values)
                    axes[i, 0].plot(k_values[max_k], metric_values.values[max_k], "ok",
                                   fillstyle="none", label=None)
                
                axes[i, 0].plot([], [], "ok", fillstyle="none", label="Max")
                axes[i, 0].set_ylabel(model)
                axes[i, 0].yaxis.grid(True, zorder=0, ls="--")
            
            axes[i, 0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            axes[i, 0].set_xlabel("Number of trees")
            axes[i, 0].set_xlim(xmin=0)
            fig.suptitle(performance.name + " for " + dataset.name)

            fig_list.append(fig)

    elif compare == "models":
        for dataset in performance.coords['dataset'].values:
            fig, axes = plt.subplots(len(performance.coords['metric'].values),
                                     sharex=True, squeeze=False, figsize=(8, 8))
            for j, metric in enumerate(performance.coords['metric'].values):  
                for i, model in enumerate(performance.coords['model'].values):
                    k_values = performance.indexes['k'].values
                    metric_values = performance.sel(dataset=dataset,
                                               model=model,
                                               metric=metric)
                    axes[j, 0].plot(k_values, metric_values.values, label=model)
                    max_k = np.nanargmax(metric_values)
                    axes[j, 0].plot(k_values[max_k], metric_values.values[max_k], "ok",
                                   fillstyle = "none", label=None)

                axes[j, 0].plot([], [], "ok", fillstyle="none", label="Max")
                axes[j, 0].set_ylabel(metric)
                axes[j, 0].yaxis.grid(True, zorder=0, ls="--")
            
            axes[j, 0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            axes[j, 0].set_xlabel("Number of trees")
            axes[j, 0].set_xlim(xmin=0)
            fig.suptitle(performance.name + " for " + dataset.name)

            fig_list.append(fig)

    elif compare == "datasets":
        for model in performance.coords['model'].values:
            fig, axes = plt.subplots(len(performance.coords['metric'].values),
                                     sharex=True, squeeze=False, figsize=(8, 8))
            for j, metric in enumerate(performance.coords['metric'].values):  
                for k, dataset in enumerate(performance.coords['dataset'].values):
                    k_values = performance.indexes['k'].values
                    metric_values = performance.sel(dataset=dataset,
                                               model=model,
                                               metric=metric)
                    axes[j, 0].plot(k_values, metric_values.values, label=dataset.name)
                    max_k = np.nanargmax(metric_values)
                    axes[j, 0].plot(k_values[max_k], metric_values.values[max_k], "ok",
                                   fillstyle = "none", label=None)
                axes[j, 0].plot([], [], "ok", fillstyle="none", label="Max")
                axes[j, 0].set_ylabel(metric)
                axes[j, 0].yaxis.grid(True, zorder=0, ls="--")
                    
            axes[j, 0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            axes[j, 0].set_xlabel("Number of trees")
            axes[j, 0].set_xlim(xmin=0)
            fig.suptitle(performance.name + " for " + model.name)

            fig_list.append(fig)
    
    return fig_list


def plot_tree_wise_average_contribution(performance):
    """
    This method plots the results obtained from the
    tree_wise_average_contribution analysis.

    Parameters
    ----------
    performance: xarray
        The xarray obtained after computing tree_wise_average_contribution.

    Returns
    -------
    fig_list : list
        The list of figures.
    """

    fig_list = []

    for dataset in performance.coords['dataset'].values:
        fig, axes = plt.subplots(len(performance.coords['model'].values),
                                 sharex=True, squeeze=False, figsize=(8, 8))
        fig.suptitle(performance.name + " for " + dataset.name)
        
        for i, model in enumerate(performance.coords['model'].values):
            metric_values = performance.sel(dataset=dataset, model=model)
            axes[i, 0].plot( metric_values.values)
            axes[i, 0].legend((model,), loc='upper center')
            axes[i, 0].yaxis.grid(True, zorder=0, ls="--")

        axes[i, 0].set_xlabel("Number of trees")
        axes[i, 0].set_xlim(xmin=0)
   
        fig_list.append(fig)
    
    return fig_list


def plot_query_wise_performance(performance, compare="models"):
    """
    This method plots the results obtained from the query_wise_performance
    analysis.

    Parameters
    ----------
    performance: xarray
        The xarray obtained after computing query_wise_performance.
    compare: string
        The compare parameter indicates what elements to compare between
        each other.
        Accepted values are 'models' or 'metrics'.

    Returns
    -------
    fig_list : list
        The list of figures.
    """

    if compare not in ["models", "metrics"]:
        raise RuntimeError("Please select compare method from "
                           "['models', 'metrics']")
    
    fig_list = []

    for dataset in performance.coords['dataset'].values:
        if compare == "metrics":
            fig, axes = plt.subplots(len(performance.coords['model'].values),
                                     sharex=True, squeeze=False, figsize=(8, 8))
            fig.suptitle(performance.name + " for " + dataset.name)
            for i, model in enumerate(performance.coords['model'].values):
                for j, metric in enumerate(performance.coords['metric'].values):
                    metric_values = performance.sel(dataset=dataset,
                                               model=model,
                                               metric=metric)
                    axes[i, 0].plot(performance.coords['bin'].values,
                                    metric_values.values*100, label=metric)

                axes[i, 0].set_ylabel("Query %")
                axes[i, 0].yaxis.set_ticks(np.arange(0, 101, 25))
                axes[i, 0].yaxis.grid(True, zorder=0, ls="--")
            axes[i, 0].set_xlabel("Metric Score")
            axes[i, 0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig_list.append(fig)

        elif compare == "models":
            fig, axes = plt.subplots(len(performance.coords['metric'].values),
                                     sharex=True, squeeze=False, figsize=(8, 8))
            fig.suptitle(performance.name + " for " + dataset.name)
            for j, metric in enumerate(performance.coords['metric'].values):
                for i, model in enumerate(performance.coords['model'].values):
                    metric_values = performance.sel(dataset=dataset,
                                               model=model,
                                               metric=metric)
                    axes[j, 0].plot(performance.coords['bin'].values,
                                    metric_values.values*100, label = model)

                axes[j, 0].set_ylabel("Query %")
                axes[j, 0].yaxis.set_ticks(np.arange(0, 101, 25))
                axes[j, 0].yaxis.grid(True, zorder=0, ls="--")
            axes[j, 0].set_xlabel(metric)
            axes[j, 0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig_list.append(fig)
    return fig_list


def plot_document_graded_relevance(performance):
    """
    This method plots the results obtained from the document_graded_relevance
    analysis.

    Parameters
    ----------
    performance: xarray
        The xarray obtained after computing document_graded_relevance.

    Returns
    -------
    fig_list : list
        The list of figures.
    """

    fig_list = []

    for dataset in performance.coords['dataset'].values:
        fig, axes = plt.subplots(len(performance.coords['model'].values),
                                 sharex=True, squeeze=False, figsize=(8, 8))
        for i, model in enumerate(performance.coords['model'].values):
            for label in performance.coords['label'].values:
                values = performance.sel(dataset=dataset,
                                         model=model,
                                         label=label).values
                a = axes[i, 0].plot(performance.coords['bin'].values, values)
            axes[i, 0].yaxis.grid(True, zorder=0, ls="--")
            axes[i, 0].set_title(performance.name + " for " + dataset.name +
                                 " and model " + model.name)
            axes[i, 0].set_ylabel("Relevance")
        axes[i, 0].set_xlabel("Predicted score")
        axes[i, 0].legend(["Label "+str(int(l)) for l in
                           performance.coords['label'].values])

        fig_list.append(fig)

    return fig_list


def is_log_scale_matrix(matrix):
    """
    This method receives in input a matrix created as
    performance.sel(dataset=X, model=Y) with li and lj as axes.

    In case the first values is at least 2 times bigger than the second values,
    we return True and the matrix will be rescaled in
    plot_rank_confusion_matrix by applying log2; otherwise we return False
    and nothing happens.

    Parameters
    ----------
    matrix : xarray
         created as performance.sel(dataset=X, model=Y) with li and lj as axes

    Returns
    -------
    : bool
        True or False

    """
    flat = matrix.values.flatten()
    flat.sort()
    if flat[-1]/flat[-2] > 2:
        return True
    else:
        return False


def plot_rank_confusion_matrix(performance):
    """
    This method plots the results obtained from the rank_confusion_matrix
    analysis.

    Parameters
    ----------
    performance: xarray
        The xarray obtained after computing rank_confusion_matrix.

    Returns
    -------
    fig_list : list
        The list of figures.
    """

    fig_list = []

    for dataset in performance.coords['dataset'].values:
        fig, axes = plt.subplots(len(performance.coords['model'].values),
                                 squeeze=False, figsize=(8, 8))
        for i, model in enumerate(performance.coords['model'].values):
            matrix = performance.sel(dataset=dataset, model=model)
            if is_log_scale_matrix(matrix):
                matrix = np.log(matrix)
            axes[i, 0].pcolormesh(matrix)
            axes[i, 0].set_title(performance.name + " for " + dataset.name +
                                " and model " + model.name)
            axes[i, 0].set_ylabel("Label j")
            axes[i, 0].set_xlabel("Label i")
            # axes[i,0].legend(["Label "+str(int(l)) for l in
            # performance.coords['label'].values])

        fig_list.append(fig)

    return fig_list


def plot_query_class_performance(performance, show_values=False, compare="models"):
    """
    This method plots the results obtained from the query_class_performance
    analysis.

    Parameters
    ----------
    performance: xarray
        The xarray obtained after computing query_class_performance.
    compare: string
        The compare parameter indicates what elements to compare between
        each other.
        Accepted values are 'models' or 'metrics'.
    show_values: bool
        If show values is True, we add numeric labels on each bar in the plot
        with the rounded value to which the bar corresponds. The default is
        False and shows no values on the bars.

    Returns
    -------
    fig_list : list
        The list of figures.

    """

    if compare not in ["models", "metrics"]:
        raise RuntimeError("Please select compare method from "
                           "['models', 'metrics']")

    fig_list = []

    for dataset in performance.coords['dataset'].values:
        num_metrics = len(performance.coords['metric'].values)
        num_models = len(performance.coords['model'].values)
        num_classes = len(performance.coords['classes'].values)
        max_width = .95

        if compare == "metrics":
            fig, axes = plt.subplots(num_models, squeeze=False, figsize=(8, 8))
            width = max_width / num_metrics
            ind = np.arange(num_classes)
            
            for i, model in enumerate(performance.coords['model'].values):
                for j, metric in enumerate(performance.coords['metric'].values):
                    classes = performance.sel(dataset=dataset,
                                              model=model,
                                              metric=metric)
                    a = axes[i, 0].bar(ind + (j * width), classes.values, width, 
                                      align="center", zorder=3)

                    # add column values on the bars
                    if show_values:
                        for k, bar in enumerate(a):
                            coords = [bar.get_height(), bar.get_width()]
                            axes[i, 0].text(k + (j * width), coords[0], 
                                            round(classes.values[k], 3),
                                            ha='center', va='bottom', zorder=3)

                # add some text for labels, title and axes ticks
                axes[i, 0].yaxis.grid(True, zorder=0, ls="--")
                axes[i, 0].set_ylabel("Metric Score")
                axes[i, 0].set_xticks(ind - width/2. + max_width / 2.)
                axes[i, 0].set_xticklabels(performance.coords['classes'].values)
                y_max = np.ceil(performance.values.max()*1.4*10.)/10.
                axes[i, 0].set_ylim([0, y_max])

            axes[i, 0].legend(performance.coords['metric'].values,
                              bbox_to_anchor=(1.05, 1),
                              loc=2, borderaxespad=0.)
            fig.suptitle(performance.name + " for " + dataset.name +
                         " and model " + model.name)

            fig_list.append(fig)

        elif compare == "models":
            fig, axes = plt.subplots(num_metrics, squeeze=False, figsize=(8, 8))
            width = max_width / num_models
            ind = np.arange(num_classes)
            
            for i, metric in enumerate(performance.coords['metric'].values):
                for j, model in enumerate(performance.coords['model'].values):
                    classes = performance.sel(dataset=dataset,
                                              model=model,
                                              metric=metric)
                    a = axes[i, 0].bar(ind + (j * width), classes.values, width,
                                       align="center", zorder=3)

                    # add column values on the bars
                    if show_values:
                        for k, bar in enumerate(a):
                            coords = [bar.get_height(), bar.get_width()]
                            axes[i, 0].text(k + (j * width), coords[0], 
                                            round(classes.values[k], 3),
                                            ha='center', va='bottom', zorder=3)

                axes[i, 0].yaxis.grid(True, zorder=0, ls="--")
                axes[i, 0].set_ylabel(metric)
                axes[i, 0].set_xticks(ind - width / 2 + max_width / 2.)
                axes[i, 0].set_xticklabels(performance.coords['classes'].values)
                y_max = np.ceil(performance.values.max()*1.4*10.)/10.
                axes[i, 0].set_ylim([0, y_max])

            axes[i, 0].legend(performance.coords['model'].values,
                              bbox_to_anchor=(1.05, 1),
                              loc=2, borderaxespad=0.)
            plt.suptitle(performance.name + " for " + dataset.name)

            fig_list.append(fig)

    return fig_list

