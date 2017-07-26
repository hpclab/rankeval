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
    '''
    Initilize plot style for RankEval visualization utilities.
    '''
    plt.style.use("seaborn-notebook")
    sns.set_palette("deep")

def plot_model_performance(performance, compare="models", show_values=False):
    for dataset in performance.coords['dataset'].values:
        fig, axes = plt.subplots()
        num_metrics = len(performance.coords['metric'].values)
        num_models = len(performance.coords['model'].values)
        width = 1. / (num_metrics * num_models)

        if num_metrics == 1 or num_models == 1:
            shrinkage = 0.5
        else:
            shrinkage = 1

        if compare == "models":
            ind = np.arange(num_metrics)
            for i, model in enumerate(performance.coords['model'].values):
                metrics = performance.sel(dataset=dataset, model=model)
                a = axes.bar(ind + (i * width), metrics.values, width * shrinkage, align="center")

                # add column values on the bars
                if show_values:
                    for j, bar in enumerate(a):
                        coords = [bar.get_height(), bar.get_width()]
                        axes.text(j + (i * width), 0.9 * coords[0], round(metrics.values[j],3),
                                  ha='center', va='bottom', rotation=45)
                        
            axes.set_title(performance.name + " for " + dataset.name)
            axes.set_xticks(ind + width*shrinkage / 2.)
            axes.set_xticklabels(performance.coords['metric'].values)
            axes.set_ylim([0, 1])

            axes.legend(performance.coords['model'].values)

            plt.tight_layout()
            
        elif compare == "metrics":
            ind = np.arange(num_models)
            for i, metric in enumerate(performance.coords['metric'].values):
                models = performance.sel(dataset=dataset, metric=metric)
                a = axes.bar(ind + (i * width), models.values, width * shrinkage, align="center")

                # add column values on the bars
                if show_values:
                    for j, bar in enumerate(a):
                        coords = [bar.get_height(), bar.get_width()]
                        axes.text(j + (i * width), 0.9 * coords[0], round(models.values[j],3),
                                  ha='center', va='bottom', rotation=45)
                        
            axes.set_title(performance.name + " for " + dataset.name)
            if num_models > 1:
                axes.set_xticks(ind + width*shrinkage / 2)
                axes.set_xticklabels(performance.coords['model'].values)
            else:
                axes.set_xlabel(performance.coords['model'].values[0].name)
                axes.get_xaxis().set_ticks([])
            axes.set_ylim([0, 1])

            axes.legend(performance.coords['metric'].values)

            plt.tight_layout()
    return fig

def resolvexticks(performance):
    sampling_factor = len(performance.coords['k'].values) / 10.
    new_xtick = islice(np.arange(len(performance.coords['k'].values)), 0, None, sampling_factor)
    new_xticklabel = islice(performance.coords['k'].values, 0, None, sampling_factor)
    xticks = list(new_xtick)
    xticks.append(np.arange(len(performance.coords['k'].values))[-1])
    xticks_labels = list(new_xticklabel)
    xticks_labels.append(performance.coords['k'].values[-1])
    return xticks, xticks_labels

def plot_tree_wise_model_performance(performance, compare="models"):
    if compare == "metrics":
        for dataset in performance.coords['dataset'].values:
            fig, axes = plt.subplots(len(performance.coords['model'].values), sharex=True, squeeze=False)
            for i, model in enumerate(performance.coords['model'].values):
                for j, metric in enumerate(performance.coords['metric'].values):
                    k_values = performance.sel(dataset=dataset, model=model, metric=metric)
                    a = axes[i,0].plot(k_values.values)

                axes[i,0].legend(performance.coords['metric'].values)
                axes[i,0].set_ylabel(model)

                if len(performance.coords['k'].values) > 10:
                    xticks, xticks_labels = resolvexticks(performance)
                    axes[i,0].set_xticks(xticks)
                    axes[i,0].set_xticklabels(xticks_labels)

            axes[i,0].set_xlabel("Number of trees")
            fig.suptitle(performance.name + " for " + dataset.name)
            #plt.tight_layout()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    elif compare == "models":
        for dataset in performance.coords['dataset'].values:
            fig, axes = plt.subplots(len(performance.coords['metric'].values), sharex=True, squeeze=False)
            for j, metric in enumerate(performance.coords['metric'].values):  
                for i, model in enumerate(performance.coords['model'].values):
                    k_values = performance.sel(dataset=dataset, model=model, metric=metric)
                    a = axes[j,0].plot(k_values.values)

                axes[j,0].legend(performance.coords['model'].values)
                axes[j,0].set_ylabel(metric)

                if len(performance.coords['k'].values) > 10:
                    xticks, xticks_labels = resolvexticks(performance)
                    axes[j,0].set_xticks(xticks)
                    axes[j,0].set_xticklabels(xticks_labels)
                #axes[j,0].set_xlim([0, len(k_values)])
                #axes[j,0].set_ylim([0, 1])

            axes[j,0].set_xlabel("Number of trees")
            fig.suptitle(performance.name + " for " + dataset.name)
            fig.subplots_adjust(top=0.88)
            plt.tight_layout()

    elif compare == "datasets":
        for model in performance.coords['model'].values:
            fig, axes = plt.subplots(len(performance.coords['metric'].values), sharex=True, squeeze=False)
            for j, metric in enumerate(performance.coords['metric'].values):  
                for k, dataset in enumerate(performance.coords['dataset'].values):
                    k_values = performance.sel(dataset=dataset, model=model, metric=metric)
                    a = axes[j,0].plot(k_values.values)

                axes[j,0].legend(performance.coords['dataset'].values)
                axes[j,0].set_ylabel(metric)

                if len(performance.coords['k'].values) > 10:
                    xticks, xticks_labels = resolvexticks(performance)
                    axes[j,0].set_xticks(xticks)
                    axes[j,0].set_xticklabels(xticks_labels)

            axes[j,0].set_xlabel("Number of trees")
            fig.suptitle(performance.name + " for " + model.name) 
            fig.subplots_adjust(top=0.88)
            plt.tight_layout()
    
    return fig

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

            plt.tight_layout()


def plot_query_wise_performance(performance, compare="model"):
    for dataset in performance.coords['dataset'].values:
        if compare == "metric":
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

        elif compare == "model":
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


def logMatrix(matrix):
    flat=matrix.values.flatten()
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
            if logMatrix(matrix):
                matrix = np.log(matrix)
#             axes[i,0]= matrix.plot()
            axes[i,0].pcolormesh(matrix)
#             axes[i,0].set_title(performance.name + " for " + dataset.name + " and model " + model.name)
            axes[i,0].set_ylabel("Label j")
            axes[i,0].set_xlabel("Label i")
            #axes[i,0].legend(["Label "+str(int(l)) for l in performance.coords['label'].values])
            plt.tight_layout()


def plot_query_class_performance(performance, show_values=False, compare="model"):
    # we assume it's only 1 dataset
    for dataset in performance.coords['dataset'].values:
        num_metrics = len(performance.coords['metric'].values)
        num_models = len(performance.coords['model'].values)
        num_classes = len(performance.coords['classes'].values)

        if num_metrics == 1 or num_models == 1:
            shrinkage = 2

        if compare == "metrics":

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

        elif compare == "models":
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
