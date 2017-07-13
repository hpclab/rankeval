from __future__ import print_function

"""
This package provides visualizations for several effectiveness analysis focused on assessing
the performance of the models in terms of accuracy. 
"""

import numpy as np
import matplotlib.pyplot as plt


plt.style.use("seaborn-notebook")


def pretty_print_model_performance(performance):
    """
    Prints model performance for each dataset in performance in a tabular form.
    These additional information will be displayed only if working inside a ipython notebook. (?)

    Parameters
    ----------
    performance: xarray.DataArray

    Returns
    -------

    """
    try:
        from IPython.display import display, HTML
        for dataset in performance.coords['dataset'].values:
            display(HTML("<h3>Dateset: %s</h3>" % dataset))
            display(performance.sel(dataset=dataset).to_pandas())
    except ImportError:
        pass


def plot_model_performance(performance, compare="metrics", show_values=False):
    for dataset in performance.coords['dataset'].values:
        fig, axes = plt.subplots()
        num_metrics = len(performance.coords['metric'].values)
        num_models = len(performance.coords['model'].values)
        width = 1. / (num_metrics * num_models)

        if num_metrics == 1 or num_models == 1:
            shrinkage = 0.5
        else:
            shrinkage = 1

        if compare == "metrics":
            ind = np.arange(num_metrics)
            for i, model in enumerate(performance.coords['model'].values):
                metrics = performance.sel(dataset=dataset, model=model)
                a = axes.bar(ind + (i * width), metrics.values, width * shrinkage)

                # add column values on the bars
                if show_values:
                    for j, bar in enumerate(a):
                        coords = [bar.get_height(), bar.get_width()]
                        axes.text(j + (i * width), 0.9 * coords[0], metrics.values[j],
                                  ha='center', va='bottom', rotation=45)

            # add some text for labels, title and axes ticks
            axes.set_title(performance.name + " for " + dataset.name)
            axes.set_xticks(ind + width / 2)
            axes.set_xticklabels(performance.coords['metric'].values)

            axes.legend(performance.coords['model'].values)

            plt.tight_layout()

        # should i have multiple subplots here? one per metric?
        elif compare == "models":
            ind = np.arange(num_models)
            for i, metric in enumerate(performance.coords['metric'].values):
                models = performance.sel(dataset=dataset, metric=metric)
                a = axes.bar(ind + (i * width), models.values, width * shrinkage)

                # add column values on the bars
                if show_values:
                    for j, bar in enumerate(a):
                        coords = [bar.get_height(), bar.get_width()]
                        axes.text(j + (i * width), 0.9 * coords[0], models.values[j],
                                  ha='center', va='bottom', rotation=45)

            # add some text for labels, title and axes ticks
            axes.set_title(performance.name + " for " + dataset.name)
            axes.set_xticks(ind + width / 2)
            axes.set_xticklabels(performance.coords['model'].values)

            axes.legend(performance.coords['metric'].values)

            plt.tight_layout()

            "You need to choose the compare option: ['models','metrics']"


            # http://xarray.pydata.org/en/stable/examples/monthly-means.html


def plot_tree_wise_model_performance(performance, compare="model"):
    # test k reset xtick and xticks_labels for bigger k datatsets like istella

    for dataset in performance.coords['dataset'].values:
        if compare == "metric":
            fig, axes = plt.subplots(len(performance.coords['model'].values), sharex=True)
            for i, model in enumerate(performance.coords['model'].values):
                for j, metric in enumerate(performance.coords['metric'].values):
                    k_values = performance.sel(dataset=dataset, model=model, metric=metric)
                    a = axes[i].plot(k_values.values)

                axes[i].legend(performance.coords['metric'].values)
                axes[i].set_ylabel(model)
                axes[i].set_xticks(np.arange(len(performance.coords['k'].values)))
                axes[i].set_xticklabels(performance.coords['k'].values)

            axes[i].set_xlabel("Number of trees")
            fig.suptitle(performance.name + " for " + dataset.name, size=16)
            fig.subplots_adjust(top=0.88)
            plt.tight_layout()


        elif compare == "model":
            fig, axes = plt.subplots(len(performance.coords['metric'].values), sharey=True)
            for j, metric in enumerate(performance.coords['metric'].values):  # we need to change figure!!!!
                for i, model in enumerate(performance.coords['model'].values):
                    k_values = performance.sel(dataset=dataset, model=model, metric=metric)
                    a = axes[j].plot(k_values.values)

                axes[j].legend(performance.coords['model'].values)
                axes[j].set_ylabel(metric)
                axes[j].set_xticks(np.arange(len(performance.coords['k'].values)))
                axes[j].set_xticklabels(performance.coords['k'].values)

            axes[j].set_xlabel("Number of trees")
            fig.suptitle(performance.name + " for " + dataset.name, size=16)
            fig.subplots_adjust(top=0.88)
            plt.tight_layout()


def plot_tree_wise_average_contribution(performance):
    for dataset in performance.coords['dataset'].values:
        if len(performance.coords['model'].values) > 1:
            # we can either plot this together on in different subfigs
            fig, axes = plt.subplots(len(performance.coords['model'].values), sharex=True)
            for i, model in enumerate(performance.coords['model'].values):

                # check if more models. we need subplot
                k_values = performance.sel(dataset=dataset, model=model)
                a = axes[i].plot(k_values.values)

                axes[i].set_title(performance.name + " for " + dataset.name)
                axes[i].set_xlabel("Number of trees")
                axes[i].legend((model,), loc='upper center')

            plt.tight_layout()

        else:
            fig, axes = plt.subplots()
            model = performance.coords['model'].values[0]
            k_values = performance.sel(dataset=dataset, model=model)
            a = axes.plot(k_values.values)

            axes.set_title(performance.name + " for " + dataset.name)
            axes.set_xlabel("Number of trees")
            axes.legend(performance.coords['model'].values)


def plot_query_wise_performance(performance, compare_by="Model"):
    #     fig = plt.Figure(figsize=(20, 3))
    #     gs = gridspec.GridSpec(2, 1, width_ratios=[20,10], height_ratios=[10,5])
    for dataset in performance.coords['dataset'].values:
        if compare_by == "Model":
            fig, axes = plt.subplots(len(performance.coords['model'].values), sharex=True)
            for i, model in enumerate(performance.coords['model'].values):
                for j, metric in enumerate(performance.coords['metric'].values):
                    k_values = performance.sel(dataset=dataset, model=model, metric=metric)
                    a = axes[i].plot(k_values.values)

                axes[i].set_title(performance.name + " for " + dataset.name + " and model " + model.name)
                axes[i].set_ylabel("Number of queries")
                axes[i].set_xlabel("Bins")
                axes[i].legend(performance.coords['metric'].values)
                axes[i].yaxis.set_ticks(np.arange(0, 1, 0.1))
            plt.tight_layout()

        else:
            fig, axes = plt.subplots(len(performance.coords['metric'].values))
            for j, metric in enumerate(performance.coords['metric'].values):  # we need to change figure!!!!
                for i, model in enumerate(performance.coords['model'].values):
                    k_values = performance.sel(dataset=dataset, model=model, metric=metric)
                    a = axes[j].plot(k_values.values)

                axes[j].set_title(performance.name + " for " + dataset.name + "and metric " + str(metric))
                axes[j].set_ylabel("Number of queries")
                axes[j].set_xlabel("Bins")
                axes[j].legend(performance.coords['model'].values)
                axes[j].yaxis.set_ticks(np.arange(0, 1, 0.1))
            plt.tight_layout()


def plot_document_graded_relevance(performance):
    for dataset in performance.coords['dataset'].values:
        fig, axes = plt.subplots(len(performance.coords['model'].values), squeeze=False)
        for i, model in enumerate(performance.coords['model'].values):
            for label in performance.coords['label'].values:
                values = performance.sel(dataset=dataset, model=model, label=label).values
                a = axes[i,0].plot(values)
            axes[i,0].set_title(performance.name + " for " + dataset.name + " and model " + model.name)
            axes[i,0].set_ylabel("Relevance")
            axes[i,0].set_xlabel("Bins")
            axes[i,0].legend(["Label "+str(int(l)) for l in performance.coords['label'].values])
            plt.tight_layout()


def is_log_scale_matrix(matrix):
    flat = matrix.values.flatten()
    flat.sort()
    if flat[-1]/flat[-2] > 2:
        return True
    else:
        return False


def plot_rank_confusion_matrix(performance):
    for dataset in performance.coords['dataset'].values:
        fig, axes = plt.subplots(len(performance.coords['model'].values), squeeze=False)
        for i, model in enumerate(performance.coords['model'].values):
            matrix = performance.sel(dataset=dataset, model=model)
            #if scale of very off -> take log
            if is_log_scale_matrix(matrix):
                matrix = np.log(matrix)
#             axes[i,0]= matrix.plot()
            axes[i,0].pcolormesh(matrix)
#             axes[i,0].set_title(performance.name + " for " + dataset.name + " and model " + model.name)
            axes[i,0].set_ylabel("Label j")
            axes[i,0].set_xlabel("Label i")
            #axes[i,0].legend(["Label "+str(int(l)) for l in performance.coords['label'].values])
            plt.tight_layout()


def plot_query_class_performance(performance, show_values=False, compare="model"):
    for dataset in performance.coords['dataset'].values:
        num_metrics = len(performance.coords['metric'].values)
        num_models = len(performance.coords['model'].values)
        num_classes = len(performance.coords['classes'].values)

        if num_metrics == 1 or num_models == 1:
            shrinkage = 2

        if compare == "metric":

            fig, axes = plt.subplots(num_models, squeeze=False)
            width = 1. / (num_classes * num_models)

            ind = np.arange(num_classes)
            for i, model in enumerate(performance.coords['model'].values):
                for j, metric in enumerate(performance.coords['metric'].values):
                    classes = performance.sel(dataset=dataset, model=model, metric=metric)
                    a = axes[i, 0].bar(ind + (j * width), classes.values, width)

                    # add column values on the bars
                    if show_values:
                        for k, bar in enumerate(a):
                            coords = [bar.get_height(), bar.get_width()]
                            axes[i, 0].text(k + (j * width), 0.5 * coords[0], classes.values[k],
                                            ha='center', va='bottom', rotation=65)

                # add some text for labels, title and axes ticks
                axes[i, 0].set_title(performance.name + " for " + dataset.name + " and model " + model.name)
                axes[i, 0].set_xticks(ind + width / 2)
                axes[i, 0].set_xticklabels(performance.coords['classes'].values)

                axes[i, 0].legend(performance.coords['metric'].values)

                plt.tight_layout()

        elif compare == "model":
            fig, axes = plt.subplots(num_metrics, squeeze=False, figsize=(8, 8))
            width = 1. / (num_classes * num_metrics)

            ind = np.arange(num_classes)
            for i, metric in enumerate(performance.coords['metric'].values):
                for j, model in enumerate(performance.coords['model'].values):
                    classes = performance.sel(dataset=dataset, model=model, metric=metric)
                    a = axes[i, 0].bar(ind + (j * width), classes.values, width * shrinkage)

                    # add column values on the bars
                    if show_values:
                        for k, bar in enumerate(a):
                            coords = [bar.get_height(), bar.get_width()]
                            axes[i, 0].text(k + (j * width), 0.5 * coords[0], classes.values[k],
                                            ha='center', va='bottom', rotation=65)

                # add some text for labels, title and axes ticks
                axes[i, 0].set_title(performance.name + " for " + dataset.name + " and metric " + metric.name)
                axes[i, 0].set_xticks(ind + width / 2)
                axes[i, 0].set_xticklabels(performance.coords['classes'].values)

                axes[i, 0].legend(performance.coords['model'].values)

                plt.tight_layout()