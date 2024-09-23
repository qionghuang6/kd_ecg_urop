'''
Calculate patient level ROC
'''

# %%
import sklearn.metrics
import numpy as np
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

from src.detection_analysis_3d.calc_froc import plot_roc_with_ci

from typing import List, Tuple, Union, Any


# %%
def to_patient_level_results(
    preds: np.array,
    labels: np.array,
    mrns: np.array,
    mrns_empty: np.array,
) -> pd.DataFrame:
    '''
    Convert the detection results to patient level ROC.
    Take the maximum of labels and predictions for each patient
    '''

    df = []
    unique_mrns = list(np.unique(mrns)) + list(mrns_empty)
    for mrn in unique_mrns:
        inds = np.where(mrns == mrn)[0]
        if len(inds) > 0:
            pred = preds[inds].max()
            label = labels[inds].max()
        else:
            pred = 0
            label = 0
        df.append({'MRN': mrn, 'pred': pred, 'label': label})
    return pd.DataFrame(df)


def roc_with_ci(
    preds: np.array,
    labels: np.array,
    ci: float = 95,
    nbst: int = 1000,
    seed: Union[int, np.random.Generator] = None,
) -> dict:
    fprs, tprs, ths = sklearn.metrics.roc_curve(labels, preds)
    auc = sklearn.metrics.auc(fprs, tprs)

    # bootstrap to get confidence interval
    tpr_bst_list = []
    auc_bst_list = []
    rng = np.random.default_rng(seed)
    for i in range(nbst):
        # sampling with replacement
        ind = rng.integers(0, len(labels), len(labels))
        label_bst = labels[ind]
        pred_bst = preds[ind]

        # calculate ROC for each bootstrap
        fpr_bst, tpr_bst, _ = sklearn.metrics.roc_curve(label_bst, pred_bst)

        # resample the roc for each fprs
        tpr_sample = np.interp(fprs, fpr_bst, tpr_bst)
        tpr_bst_list.append(tpr_sample)
        auc_bst_list.append(sklearn.metrics.auc(fpr_bst, tpr_bst))

    tpr_bst_list = np.array(tpr_bst_list)
    tprs_ci = np.percentile(tpr_bst_list, [(100 - ci) / 2, 100 - (100 - ci) / 2], axis=0)
    auc_ci = np.percentile(auc_bst_list, [(100 - ci) / 2, 100 - (100 - ci) / 2])

    return {
        'fprs': fprs,
        'tprs': tprs,
        'ths': ths,
        'tprs_ci': tprs_ci,
        'auc': auc,
        'auc_ci': auc_ci
    }


# %%
def calc_patient_rocs(
    froc_manifest_list: List[pd.DataFrame],
    mrn_list: List[str],
    legends: List[str] = None,
    seed: int = 0,
    plot_ci: bool = True,
    verbose: int = 1
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, List[dict]]:
    '''
    @froc_manifest_list: each manifest must have three columns: "prob", "label"
    @mrn_list: the full list of mrns, used to identify which patients have no label or predictions
    @seed: the random seed for ci calculation
    @plot_ci: if plot confidence interval or not. It will always be calculated.

    @return
    @ax: the axes object that contains the plot
    @roc_curves: a list of all the roc curve information (dictionary).
    '''

    rng = np.random.default_rng(seed)
    rocs = []
    if verbose > 0:
        print('Number of FROCS = {0}'.format(len(froc_manifest_list)))
    for i, manifest in enumerate(froc_manifest_list):
        mrn_empty = np.array([m for m in mrn_list if m not in manifest['MRN'].values])

        # fpr
        if verbose > 0:
            print('Plotting fpr ROC {0}'.format(i), flush=True)
        roc = to_patient_level_results(
            manifest['prob'].values,
            manifest['label'].values,
            manifest['MRN'].values,
            mrn_empty
        )
        roc = roc_with_ci(
            roc['pred'].values,
            roc['label'].values,
            seed=rng
        )

        rocs.append(roc)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyle_cycle = ['-', '--']
    fig, ax = plt.subplots(figsize=[6, 4], dpi=100)
    for i in range(len(rocs)):
        color = colors[i // 2]
        linestyle = linestyle_cycle[i % 2]
        if plot_ci:
            ci = rocs[i]['tprs_ci']
        else:
            ci = None
        ax = plot_roc_with_ci(
            ax, rocs[i]['fprs'], rocs[i]['tprs'], ci, color=color, linestyle=linestyle
        )

    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if legends is not None:
        ax.legend([legends[i] for i in range(len(legends))])

    return fig, ax, rocs
