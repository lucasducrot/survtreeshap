import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import survshap
import seaborn as sns 
import survinsights
import time
import numba
import sklearn.ensemble
import itertools
import random
from sksurv.util import Surv
from sksurv.metrics import brier_score
#from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sksurv.datasets import load_whas500
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import concordance_index_censored

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

from survinsights import local_explaination
from survinsights.explainer import explainer

from tqdm import tqdm
from scipy.integrate import simpson





def Shapvalues(model= None, data=None, times=None):
    if times is None:
        times=model.estimators_[0].unique_times_

    x = np.array(times)
    y = np.array([
        TreeExplainer(model,t=i).shap_values(data)
        for i in tqdm(range(len(times)))
        ])
    
    return(x,y)

def STP_graph(x=None, y=None, columns=None, selection=None, min=None, max=None):

    if selection=="minmax":
        top_values= [y[:,i].max() for i in range(len(y[1])-1)]
        top_indices = np.argsort(top_values)[-max:]
        bot_values= [y[:,i].min() for i in range(len(y[1])-1)]
        bot_indices = np.argsort(bot_values)[:min]

        indices=np.concatenate((top_indices, bot_indices))

        y_plot=np.array([y[:,i] for i in indices]).T
        legend=[columns[i] for i in indices]

        plt.plot(x,y_plot,label=legend)

        # Ajouter des labels et une légende
        plt.xlabel("Temps (mois)")
        plt.ylabel("Shap_values")
        plt.title("Shap_values les plus contributives pour le patient méthode minmax")
        plt.legend(loc="upper left",fontsize=8)

        # Afficher le graphique
        plt.show()
    
    elif selection=="area":
        area = [np.trapz(y[:,i], x) for i in range(len(y[1])-1)]
        top_indices = np.argsort(area)[-max:]
        bot_indices = np.argsort(area)[:min]

        indices=np.concatenate((top_indices, bot_indices))

        y_plot=np.array([y[:,i] for i in indices]).T
        legend=[columns[i] for i in indices]

        plt.plot(x,y_plot,label=legend)

        # Ajouter des labels et une légende
        plt.xlabel("Temps (mois)")
        plt.ylabel("Shap_values")
        plt.title("Shap_values les plus contributives pour le patient méthode aire")
        plt.legend(loc="upper left",fontsize=8)

        # Afficher le graphique
        plt.show()
    
    else:
        y_plot=np.array([y[:,i] for i in range(len(y[1])-1)]).T

        plt.plot(x,y_plot,label=columns)

        # Ajouter des labels et une légende
        plt.xlabel("Temps (mois)")
        plt.ylabel("Shap_values")
        plt.title("Toutes les Shap_values du patient")
        plt.legend(loc="upper left",fontsize=8)

        # Afficher le graphique
        plt.show() 

def Shapvaluesrank(model= None, data=None, times=None, sample_size=None, number_of_values=None ):
    if sample_size==None:
        sample_size=len(data)
    sample_size=min(sample_size, len(data))
    if number_of_values==None:
        number_of_values = len(data.columns)
    aires=np.zeros(len([data.columns]))
    for i in range(sample_size):
        x,y=Shapvalues(model= model, data=data.iloc[i, :], times=times)
        aires = aires+ np.array([np.trapz(np.abs(y[:,i]), x) for i in range(len(y[1])-1)])

    top_indices = np.argsort(aires)[-number_of_values:]
    return([( number_of_values-n, data.columns[i], aires[i]/sample_size) for n, i in enumerate(top_indices)][::-1])

def VariableRank(model= None, data=None, times=None, sample_size=None, plot=True ):
    if sample_size==None:
        sample_size=len(data)
    sample_size=min(sample_size, len(data))
    rank=[]
    for i in range(sample_size):
        x,y=Shapvalues(model= model, data=data.iloc[i, :], times=times)
        aires = np.array([np.trapz(np.abs(y[:,i]), x) for i in range(len(y[1])-1)])
        top_indices = np.argsort(aires)[-len(data.columns):]
        rank.append(top_indices)

    # Matrice de comptage : lignes = rangs, colonnes = variables
    counts = np.zeros((len(data.columns), len(data.columns)), dtype=int)

    # Comptage des apparitions
    for arr in rank:
        for pos, val in enumerate(arr):
            counts[pos, val] += 1

    # DataFrame pour tracer facilement
    df = pd.DataFrame(counts, columns=[i for i in range(len(data.columns))])
    df.index = [f"{len(data.columns)-i}" for i in range(len(data.columns))]

    if plot: 
        fig, ax = plt.subplots(figsize=(10, 6))

        bottom = np.zeros(len(df))
        for i, col in enumerate(df.columns):
            ax.barh(df.index, df[col], left=bottom, label=data.columns[col])
            bottom += df[col]

        # Mise en forme
        ax.set_xlabel("Nombre d'apparitions à cette position")
        ax.set_ylabel("Rang d'importance")
        ax.set_title("Distribution des variables par rang d'importance")
        ax.legend(title="Variables", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()
    df.columns = list(data.columns)
    return(df)

class TreeExplainer:
    def __init__(self, model, t=0, **kwargs):
        if str(type(model)).endswith("sklearn.ensemble._forest.RandomForestRegressor'>") or 1==1: #attention
            # self.trees = [Tree(e.tree_) for e in model.estimators_]
            self.trees = [
                Tree(
                    children_left=e.tree_.children_left,
                    children_right=e.tree_.children_right,
                    children_default=e.tree_.children_right,
                    feature=e.tree_.feature,
                    threshold=e.tree_.threshold,
                    value=e.tree_.value[:, t, 1],
                    node_sample_weight=e.tree_.weighted_n_node_samples,
                )
                for e in model.estimators_
            ]

        # Preallocate space for the unique path data
        maxd = np.max([t.max_depth for t in self.trees]) + 2
        s = (maxd * (maxd + 1)) // 2
        self.feature_indexes = np.zeros(s, dtype=np.int32)
        self.zero_fractions = np.zeros(s, dtype=np.float64)
        self.one_fractions = np.zeros(s, dtype=np.float64)
        self.pweights = np.zeros(s, dtype=np.float64)

    def shap_values(self, X, **kwargs):
        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
            X = X.values

        assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

        # single instance
        if len(X.shape) == 1:
            phi = np.zeros(X.shape[0] + 1)
            x_missing = np.zeros(X.shape[0], dtype=bool)
            for t in self.trees:
                self.tree_shap(t, X, x_missing, phi)
            phi /= len(self.trees)
        elif len(X.shape) == 2:
            phi = np.zeros((X.shape[0], X.shape[1] + 1))
            x_missing = np.zeros(X.shape[1], dtype=bool)
            for i in range(X.shape[0]):
                for t in self.trees:
                    self.tree_shap(t, X[i, :], x_missing, phi[i, :])
            phi /= len(self.trees)
        return phi

    def tree_shap(self, tree, x, x_missing, phi, condition=0, condition_feature=0):
        # update the bias term, which is the last index in phi
        # (note the paper has this as phi_0 instead of phi_M)
        if condition == 0:
            phi[-1] += tree.values[0]

        # start the recursive algorithm
        tree_shap_recursive(
            tree.children_left,
            tree.children_right,
            tree.children_default,
            tree.features,
            tree.thresholds,
            tree.values,
            tree.node_sample_weight,
            x,
            x_missing,
            phi,
            0,
            0,
            self.feature_indexes,
            self.zero_fractions,
            self.one_fractions,
            self.pweights,
            1,
            1,
            -1,
            condition,
            condition_feature,
            1,
        )

# extend our decision path with a fraction of one and zero extensions
@numba.jit(
    numba.types.void(
        numba.types.int32[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.int32,
        numba.types.float64,
        numba.types.float64,
        numba.types.int32,
    ),
    nopython=True,
    nogil=True,
)
def extend_path(
    feature_indexes,
    zero_fractions,
    one_fractions,
    pweights,
    unique_depth,
    zero_fraction,
    one_fraction,
    feature_index,
):
    feature_indexes[unique_depth] = feature_index
    zero_fractions[unique_depth] = zero_fraction
    one_fractions[unique_depth] = one_fraction
    if unique_depth == 0:
        pweights[unique_depth] = 1
    else:
        pweights[unique_depth] = 0

    for i in range(unique_depth - 1, -1, -1):
        pweights[i + 1] += one_fraction * pweights[i] * (i + 1) / (unique_depth + 1)
        pweights[i] = zero_fraction * pweights[i] * (unique_depth - i) / (unique_depth + 1)


# undo a previous extension of the decision path
@numba.jit(
    numba.types.void(
        numba.types.int32[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.int32,
        numba.types.int32,
    ),
    nopython=True,
    nogil=True,
)
def unwind_path(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0:
            tmp = pweights[i]
            pweights[i] = next_one_portion * (unique_depth + 1) / ((i + 1) * one_fraction)
            next_one_portion = tmp - pweights[i] * zero_fraction * (unique_depth - i) / (unique_depth + 1)
        else:
            pweights[i] = (pweights[i] * (unique_depth + 1)) / (zero_fraction * (unique_depth - i))

    for i in range(path_index, unique_depth):
        feature_indexes[i] = feature_indexes[i + 1]
        zero_fractions[i] = zero_fractions[i + 1]
        one_fractions[i] = one_fractions[i + 1]


# determine what the total permuation weight would be if
# we unwound a previous extension in the decision path
@numba.jit(
    numba.types.float64(
        numba.types.int32[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.int32,
        numba.types.int32,
    ),
    nopython=True,
    nogil=True,
)
def unwound_path_sum(feature_indexes, zero_fractions, one_fractions, pweights, unique_depth, path_index):
    one_fraction = one_fractions[path_index]
    zero_fraction = zero_fractions[path_index]
    next_one_portion = pweights[unique_depth]
    total = 0

    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0:
            tmp = next_one_portion * (unique_depth + 1) / ((i + 1) * one_fraction)
            total += tmp
            next_one_portion = pweights[i] - tmp * zero_fraction * ((unique_depth - i) / (unique_depth + 1))
        else:
            total += (pweights[i] / zero_fraction) / ((unique_depth - i) / (unique_depth + 1))

    return total


class Tree:
    def __init__(
        self,
        children_left,
        children_right,
        children_default,
        feature,
        threshold,
        value,
        node_sample_weight,
    ):
        self.children_left = children_left.astype(np.int32)
        self.children_right = children_right.astype(np.int32)
        self.children_default = children_default.astype(np.int32)
        self.features = feature.astype(np.int32)
        self.thresholds = threshold
        self.values = value
        self.node_sample_weight = node_sample_weight

        self.max_depth = compute_expectations(
            self.children_left,
            self.children_right,
            self.node_sample_weight,
            self.values,
            0,
        )


@numba.jit(nopython=True)
def compute_expectations(children_left, children_right, node_sample_weight, values, i, depth=0):
    if children_right[i] == -1:
        values[i] = values[i]
        return 0
    else:
        li = children_left[i]
        ri = children_right[i]
        depth_left = compute_expectations(children_left, children_right, node_sample_weight, values, li, depth + 1)
        depth_right = compute_expectations(children_left, children_right, node_sample_weight, values, ri, depth + 1)
        left_weight = node_sample_weight[li]
        right_weight = node_sample_weight[ri]
        v = (left_weight * values[li] + right_weight * values[ri]) / (left_weight + right_weight)
        values[i] = v
        return max(depth_left, depth_right) + 1


# recursive computation of SHAP values for a decision tree
@numba.jit(
    numba.types.void(
        numba.types.int32[:],
        numba.types.int32[:],
        numba.types.int32[:],
        numba.types.int32[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.boolean[:],
        numba.types.float64[:],
        numba.types.int64,
        numba.types.int64,
        numba.types.int32[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64[:],
        numba.types.float64,
        numba.types.float64,
        numba.types.int64,
        numba.types.int64,
        numba.types.int64,
        numba.types.float64,
    ),
    nopython=True,
    nogil=True,
)
def tree_shap_recursive(
    children_left,
    children_right,
    children_default,
    features,
    thresholds,
    values,
    node_sample_weight,
    x,
    x_missing,
    phi,
    node_index,
    unique_depth,
    parent_feature_indexes,
    parent_zero_fractions,
    parent_one_fractions,
    parent_pweights,
    parent_zero_fraction,
    parent_one_fraction,
    parent_feature_index,
    condition,
    condition_feature,
    condition_fraction,
):
    # stop if we have no weight coming down to us
    if condition_fraction == 0:
        return

    # extend the unique path
    feature_indexes = parent_feature_indexes[unique_depth + 1 :]
    feature_indexes[: unique_depth + 1] = parent_feature_indexes[: unique_depth + 1]
    zero_fractions = parent_zero_fractions[unique_depth + 1 :]
    zero_fractions[: unique_depth + 1] = parent_zero_fractions[: unique_depth + 1]
    one_fractions = parent_one_fractions[unique_depth + 1 :]
    one_fractions[: unique_depth + 1] = parent_one_fractions[: unique_depth + 1]
    pweights = parent_pweights[unique_depth + 1 :]
    pweights[: unique_depth + 1] = parent_pweights[: unique_depth + 1]

    if condition == 0 or condition_feature != parent_feature_index:
        extend_path(
            feature_indexes,
            zero_fractions,
            one_fractions,
            pweights,
            unique_depth,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        )

    split_index = features[node_index]

    # leaf node
    if children_right[node_index] == -1:
        for i in range(1, unique_depth + 1):
            w = unwound_path_sum(
                feature_indexes,
                zero_fractions,
                one_fractions,
                pweights,
                unique_depth,
                i,
            )
            phi[feature_indexes[i]] += (
                w * (one_fractions[i] - zero_fractions[i]) * values[node_index] * condition_fraction
            )

    # internal node
    else:
        # find which branch is "hot" (meaning x would follow it)
        hot_index = 0
        cleft = children_left[node_index]
        cright = children_right[node_index]
        if x_missing[split_index] == 1:
            hot_index = children_default[node_index]
        elif x[split_index] < thresholds[node_index]:
            hot_index = cleft
        else:
            hot_index = cright
        cold_index = cright if hot_index == cleft else cleft
        w = node_sample_weight[node_index]
        hot_zero_fraction = node_sample_weight[hot_index] / w
        cold_zero_fraction = node_sample_weight[cold_index] / w
        incoming_zero_fraction = 1
        incoming_one_fraction = 1

        # see if we have already split on this feature,
        # if so we undo that split so we can redo it for this node
        path_index = 0
        while path_index <= unique_depth:
            if feature_indexes[path_index] == split_index:
                break
            path_index += 1

        if path_index != unique_depth + 1:
            incoming_zero_fraction = zero_fractions[path_index]
            incoming_one_fraction = one_fractions[path_index]
            unwind_path(
                feature_indexes,
                zero_fractions,
                one_fractions,
                pweights,
                unique_depth,
                path_index,
            )
            unique_depth -= 1

        # divide up the condition_fraction among the recursive calls
        hot_condition_fraction = condition_fraction
        cold_condition_fraction = condition_fraction
        if condition > 0 and split_index == condition_feature:
            cold_condition_fraction = 0
            unique_depth -= 1
        elif condition < 0 and split_index == condition_feature:
            hot_condition_fraction *= hot_zero_fraction
            cold_condition_fraction *= cold_zero_fraction
            unique_depth -= 1

        tree_shap_recursive(
            children_left,
            children_right,
            children_default,
            features,
            thresholds,
            values,
            node_sample_weight,
            x,
            x_missing,
            phi,
            hot_index,
            unique_depth + 1,
            feature_indexes,
            zero_fractions,
            one_fractions,
            pweights,
            hot_zero_fraction * incoming_zero_fraction,
            incoming_one_fraction,
            split_index,
            condition,
            condition_feature,
            hot_condition_fraction,
        )

        tree_shap_recursive(
            children_left,
            children_right,
            children_default,
            features,
            thresholds,
            values,
            node_sample_weight,
            x,
            x_missing,
            phi,
            cold_index,
            unique_depth + 1,
            feature_indexes,
            zero_fractions,
            one_fractions,
            pweights,
            cold_zero_fraction * incoming_zero_fraction,
            0,
            split_index,
            condition,
            condition_feature,
            cold_condition_fraction,
        )
