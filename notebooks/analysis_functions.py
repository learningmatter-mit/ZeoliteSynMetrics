"""This document contains functions that I use frequently
to analyze and manipulate data for OSDA-zeolite affinities.
I've put all of these functions into a central location so
that I don't repeatedly define them in my notebooks.
List of functions:
- my_mpl_settings: Set some of my preferred matplotlib settings (Arial font,
    200 dpi)
- remove_outliers: Remove outliers from a dataframe based on the interquartile
- get_num_atoms: Get the number of atoms in a chemical formula
- add_competition_or_directivity_softmax: Compute the competitions for all
    OSDAs between different frameworks
- templating_energy: Get the templating energy for a dataframe with a set of
    column names
- make_df_col_string: Takes the name of a variable that you've used for a column
    name in a dataframe and returns a string version of it for evaluation in a
    formula
- parse_sisso_eqn: Parses the equations that SISSO outputs into a format that can
    be evaluated by python for a given dataframe with a set of column names
- get_recall: Get the literature recall for a given metric from a dataframe
- get_norm_auc: Get the normalized AUC for a given metric
- compare_aucs: Collect the AUCs or normalized AUCs for all metrics and substrates
- plot_recall_curves: Plot recall curves for all metrics in the dataframe
- get_rot_entropy: Get the rotational entropy of a molecule's conformers
- get_one_dim_trans_entropy: Get the one-dimensional translational entropy for a
    molecule from its ASE atoms object using ideal gas statistical mechanics
- get_fractional_zeo_rot_entropy_exp: Get the fractional rotational entropy of a
    molecule in a zeolite
- compute_zeo_ads_entropy: Compute the zeolite adsorption entropy, as estimated by
    Dauenhauer and Abdelrahman, ACS Cent. Sci. (2018)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import units
from constants import EV_TO_KJ_MOL, K_BOLTZMANN
from file_manipulation_functions import csv_to_dict
from matplotlib import cm
from scipy.special import softmax
from tqdm import tqdm

if TYPE_CHECKING:
    from ase.atoms import Atoms


SYNTH_TEMP = 400  # K
F_ROTSLAB = 0.03  # unitless
ZEO_ROT_COEFF = 1 / 7  # unitless
V_CRITICAL = 127.3  # Angstrom^3
V_CRIT_ALT = 100.0  # Angstrom^3
S_ARGON = 0.1548  # kJ/mol/K
M_ARGON = 39.948  # amu
SACKUR_TETRODE_TEMP = 298  # K
ALPHA = 6.253e-3  # unitless
BETA = 0.5  # unitless
V_CRIT_ALT = 50.0  # Angstrom^3

REFERENCEPRESSURE = 1.0e5  # Pa (equivalent to 1 bar)

file_path = "../data/occupiable_volumes.csv"
V_OCC = csv_to_dict(file_path, "Code", "V_occ")

for k, v in V_OCC.items():
    V_OCC[k] = float(v)


def my_mpl_settings() -> None:
    """Set some of my preferred matplotlib settings (Arial font,
    200 dpi)
    """
    mpl.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    mpl.font_manager.findfont("Arial")
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.family"] = "Arial"


def remove_outliers(
    df_: pd.DataFrame, cols: list[int] = (), n_std: int = 3, method: str = "IQR"
) -> pd.DataFrame:
    """A function to remove outliers based on the interquartile range of the data
    from a dataframe. This is a more robust method than using the standard deviation.
    Because we have very large tails and some extreme outliers that will affect the
    standard deviation, the IQR should provide a more reliable way to exclude outliers.

    Args:
        df_ (pd.DataFrame): dataframe to remove outliers from
        cols (List[int], optional): columns to remove outliers from. If None,
            all numeric columns are used. Defaults to None.
        n_std (int, optional): number of standard deviations to use for the IQR.
            Defaults to 3. Previously used 2.22 (which would equate to 3 std devs
            in a normal distribution for the IQR)
        method (str): method to use for removing outliers, either "IQR" or
            "STDDEV". Defaults to "IQR".

    Returns:
        pd.DataFrame: dataframe with outliers removed
    """
    method_ = method.upper()
    if not cols:
        cols = df_.select_dtypes("number").columns
    dfi_cp = df_.copy()
    df_sub = df_.loc[:, cols]

    if method_ == "STDDEV":
        lim = np.abs((df_sub - df_sub.mean()) / df_sub.std(ddof=0)) < n_std
    elif method_ == "IQR":
        iqr = df_sub.quantile(0.75) - df_sub.quantile(0.25)
        lim = np.abs((df_sub - df_sub.median()) / iqr) < n_std

    dfi_cp.loc[:, cols] = df_sub.where(lim, np.nan)
    return dfi_cp


def get_num_atoms(formula: str) -> int:
    """Get the total number of atoms in a formula

    Args:
        formula (str): chemical formula

    Returns:
        int: total atoms
    """
    numbers = re.findall("[A-Z][a-z]?([0-9]*)", formula)
    return sum([int(n) if n != "" else 1 for n in numbers])


def add_competition_or_directivity_softmax(
    df: pd.DataFrame,
    metric: str,
    metric_col: str,
    binding_col: str,
    osda_col: str = "SMILES",
    framework_col: str = "Zeolite",
    temperature: float = SYNTH_TEMP,
) -> pd.DataFrame:
    """Compute the competitions for all OSDAs between different frameworks.

    Args:
        df (pd.Dataframe): a dataframe with binding data
        metric (str): the name of the metric to compute, either "competition" or "directivity"
        metric_col (str): name of the column to store the metric data
        binding_col (str): name of the column in the dataframe with the binding energies
        osda_col (str): name of the column in the dataframe with OSDA SMILES
        framework_col (str): name of the column in the dataframe with the framework names
        temperature (float): temperature in K. Defaults to SYNTH_TEMP.

    Returns:
        pd.DataFrame: dataframe with the competition data
    """
    mdf = df.copy()
    if metric == "competition":
        # compute softmax with sum of weights for each framework
        zeolist = mdf[framework_col].unique()
        for z in tqdm(zeolist):
            mdf.loc[mdf[framework_col] == z, metric_col] = softmax(
                -mdf.loc[mdf[framework_col] == z, binding_col] / (K_BOLTZMANN * temperature),
                axis=0,
            )
    elif metric == "directivity":
        mollist = mdf[osda_col].unique()
        for m in tqdm(mollist):
            mdf.loc[mdf[osda_col] == m, metric_col] = softmax(
                -mdf.loc[mdf[osda_col] == m, binding_col] / (K_BOLTZMANN * temperature),
                axis=0,
            )
    return mdf


def templating_energy(
    df: pd.DataFrame,
    column_names: list[str],
    temperature: float = SYNTH_TEMP,
) -> pd.DataFrame:
    """Get the templating energy for a dataframe with a set of column names.

    Args:
        df (pd.DataFrame): dataframe with the data
        column_names (list[str]): list of column names to use in the templating energy, the
            competition and directivity columns
        temperature (float, optional): temperature in K. Defaults to SYNTH_TEMP.

    Returns:
        pd.Series: series with the templating energy
    """
    return -0.25 * K_BOLTZMANN * temperature * np.log(df[column_names].product(axis=1))


def make_df_col_string(col_name: str, df_name: str = "small_df") -> str:
    """Takes the name of a variable that you've used for a column name in a dataframe
       and returns a string version of it for evaluation in a formula.

    Args:
        col_name (str): the name of the column/variable
        df_name (str, optional): the dataframe name. Defaults to 'small_df'.

    Returns:
        str: the string version of the column name with the dataframe name
    """
    return df_name + "['" + col_name + "']"


def parse_sisso_eqn(eqn: str, col_names: list | tuple, df_name: str = "small_df") -> str:
    """Parses the equations that SISSO outputs into a format that can be evaluated
    by python for a given dataframe with a set of column names that the user specifies.

    Args:
        eqn (str): equation output from SISSO
        col_names (iterable): list or tuple of column names that you used in the dataframe
        df_name (str): name of the dataframe as a string (e.g. 'small_df')

    Returns:
        str: parsed equation for your dataframe that python can evaluate
    """
    equation = eqn[:]

    # replace the variable names with the dataframe column names
    for name in col_names:
        raw_string = r"\b(" + name + r")\b"
        df_string = make_df_col_string(name, df_name=df_name)
        equation = re.sub(raw_string, df_string, equation)

    # some functions need to be replace with the python version
    equation = re.sub(r"\^", "**", equation)
    equation = re.sub(r"exp", "e**", equation)
    equation = re.sub(r"log", "np.log", equation)

    return equation  # noqa: RET504


def get_recall(
    df_: pd.DataFrame, metric: str, syn_col: str = "syn", ascending: bool = True
) -> pd.Series:
    """Get the literature recall for a given metric from a dataframe.

    Args:
        df_ (pd.DataFrame): dataframe with the data
        metric (str): metric (sisso equation) to sort by
        syn_col (str, optional): column name for the synthesis data. Defaults to "syn",
            but in Daniel's data it's "In literature?".
        ascending (bool, optional): whether or not data should be sorted in ascending
            order. Defaults to True.

    Returns:
        pd.Series: series with the recall at each point.
    """
    sdf = df_.sort_values(metric, ascending=ascending)
    return sdf[syn_col].cumsum() / sdf[syn_col].sum()


def get_norm_auc(df_: pd.DataFrame, metric: str, syn_col: str = "syn") -> float:
    """Get the normalized AUC for a given metric.

    Args:
        df_ (pd.DataFrame): dataframe with the data
        metric (str): metric (sisso equation) to sort by
        syn_col (str, optional): column name for the synthesis data. Defaults to "syn",
            but in Daniel's data it's "In literature?".

    Returns:
        float: normalized AUC for the recall plot (0-1)
    """
    s = get_recall(df_, metric, syn_col=syn_col)
    best_case = get_recall(df_, metric=syn_col, syn_col=syn_col, ascending=False)
    worst_case = get_recall(df_, metric=syn_col, syn_col=syn_col, ascending=True)

    return (s.mean() - worst_case.mean()) / (best_case.mean() - worst_case.mean())


def compare_aucs(
    df: pd.DataFrame,
    idxmin: pd.DataFrame,
    substrates: list,
    syn_col: str = "syn",
    sort_values: str = "norm_auc",
    metrics: list | tuple = ("Templating", "A_T"),
) -> pd.DataFrame:
    """Collect the AUCs or normalized AUCs for all metrics and substrates in a dataframe.

    Args:
        df (pd.DataFrame): dataframe with comparison metrics
        idxmin (pd.DataFrame): dataframe with the index of the minimum value for each substrate
        substrates (list): list of substrates to include
        syn_col (str, optional): column name for the synthesis data. Defaults to "syn",
            but in Daniel's data it's "In literature?".
        sort_values (str, optional): column to sort the results by. Defaults to "norm_auc", but
            can also be "auc".
        metrics (list | tuple, optional): list of metrics to include.
            Defaults to ("Templating", "A_T").

    Returns:
        pd.DataFrame: a dataframe with the AUCs for each metric (columns) and substrate (rows).
            You can take the average to compare the performance of different metrics.
    """
    all_subs = []
    all_metrics = []
    all_aucs = []
    all_norm_aucs = []

    for metric in tqdm(metrics):
        for subst in substrates:
            try:
                sdf = df.loc[idxmin.loc[subst, metric].tolist()]
            except KeyError:
                continue
            recall = get_recall(sdf, metric, syn_col=syn_col)

            auc = recall.mean()
            norm_auc = get_norm_auc(sdf, metric, syn_col=syn_col)

            all_subs.append(subst)
            all_metrics.append(metric)
            all_aucs.append(auc)
            all_norm_aucs.append(norm_auc)

    auc_df = pd.DataFrame(
        {
            "zeolite": all_subs,
            "metric": all_metrics,
            "auc": all_aucs,
            "norm_auc": all_norm_aucs,
        }
    )

    return pd.pivot_table(auc_df, values=sort_values, index=["zeolite"], columns=["metric"])


def plot_recall_curves(
    lit_df: pd.DataFrame,
    idxmin: pd.DataFrame,
    metrics: list[str],
    plot_items: list[str],
    syn_col: str = "syn",
    substance: str = "zeolite",
    cmap=cm.coolwarm,
    size: float = 2.5,
):
    """Plot recall curves for all metrics in the dataframe.

    Args:
        lit_df (pd.DataFrame): dataframe with the literature data
        idxmin (pd.DataFrame): dataframe with the index of the minimum value for each substrate
        metrics (list[str]): list of metrics to include (e.g., "Templating"). Must match
            column names in the literature dataframe.
        plot_items (list[str]): list of substances to include. Can be either zeolite codes
            or OSDA SMILES.
        syn_col (str, optional): column name for the synthesis data. Defaults to "syn",
            but in Daniel's data it's "In literature?".
        substance (str, optional): the substance for which you are plotting recall, either
            "zeolite" or "osda". If "zeolite", each curve in the plot will represent the
            recall of molecules for a given framework (and so on for molecules). Defaults
            to "zeolite".
        cmap (cm, optional): colormap to use for the plot. Defaults to cm.coolwarm.
        size (float, optional): size of each plot in inches. Defaults to 2.5.
    """
    norm = mpl.colors.Normalize(vmin=0.3, vmax=0.85)

    num_plots = len(metrics)
    # print(num_plots)
    # subplots should be a tuple with up to three plots per row
    subplot_dims = (1, num_plots)
    fig_size = (subplot_dims[1] * size, subplot_dims[0] * size)
    fig, ax = plt.subplots(*subplot_dims, figsize=fig_size, sharey=True)

    for i, metric in tqdm(enumerate(metrics)):
        for subst in plot_items:
            try:
                sdf = lit_df.loc[idxmin.loc[subst, metric].tolist()]
            except KeyError as err:
                print(err)
                continue
            recall = get_recall(sdf, metric)
            x = np.linspace(0, 1, len(recall))

            norm_auc = get_norm_auc(sdf, metric, syn_col=syn_col)
            # print(i, subst, norm_auc, len(recall))
            ax[i].plot(x, recall.to_numpy(), label=subst, color=cmap(norm(norm_auc)))

    xticks = np.linspace(0, 1, 6)

    for i, a in enumerate(ax.flatten()):
        x = np.linspace(0, 1, 10)
        a.plot(x, x, color="black", linestyle="--", linewidth=0.8)
        a.legend(
            ncol=1,
            frameon=False,
            fontsize="medium",
            loc="lower right",
            bbox_to_anchor=(1.03, 0.0),
        )
        a.set_xlim(0, 1.05)
        a.set_ylim(0, 1.05)
        a.set_aspect("equal")
        a.set_xticks(xticks)
        a.set_xticklabels(["%d" % t for t in reversed(xticks * 100)])
        a.set_yticks(xticks)
        a.set_yticklabels(["%d" % t for t in (xticks * 100)])
        if substance == "osda":
            a.set_xlabel("Zeolite Percentile", fontweight="bold")
        else:
            a.set_xlabel("Molecule Percentile", fontweight="bold")
        if i == 0:
            a.set_ylabel("True Positives Recalled (%)", fontweight="bold")

    return fig, ax


def get_rot_entropy(
    atoms: Atoms,
    sigma: int = 1,
    temperature: float = SYNTH_TEMP,
    geometry: str = "nonlinear",
) -> float:
    """Get the rotational entropy of a molecule's conformers

    Args:
        atoms (Atoms): the molecule for which you want to compute S_r
        sigma (int): the symmetry number. Defaults to 1.
        temperature (float): the temperature (in K)
        geometry (str): the type of geometry for symmetry purposes, either "monatomic",
            "linear", or "nonlinear"

    Returns:
        float: the rotational entropy of the molecule based on ideal
            gas-phase statistical mechanics treatment in kJ/mol/K
    """
    # Rotational entropy (term inside the log is in SI units).
    if geometry == "monatomic":
        S_r = 0.0
    elif geometry == "nonlinear":
        inertias = atoms.get_moments_of_inertia() * units._amu / (10.0**10) ** 2  # kg m^2
        S_r = np.sqrt(np.pi * np.prod(inertias)) / sigma
        S_r *= (8.0 * np.pi**2 * units._k * temperature / units._hplanck**2) ** (3.0 / 2.0)
        S_r = units.kB * (np.log(S_r) + 3.0 / 2.0)
    elif geometry == "linear":
        inertias = atoms.get_moments_of_inertia() * units._amu / (10.0**10) ** 2  # kg m^2
        inertia = max(inertias)  # should be two identical and one zero
        S_r = 8 * np.pi**2 * inertia * units._k * temperature / sigma / units._hplanck**2
        S_r = units.kB * (np.log(S_r) + 1.0)
    return S_r * EV_TO_KJ_MOL  # rotational entropy in kJ/mol/K


def get_one_dim_trans_entropy(
    atoms: Atoms,
    temperature: float = SYNTH_TEMP,
    method: str = "sackur-tetrode",
    debug: bool = False,
) -> float:
    """Get the one-dimensional translational entropy for a molecule from its
    ASE atoms object using ideal gas statistical mechanics

    Args:
        atoms (Atoms): the molecule of interest
        temperature (float): the temperature in K
        method (str): the way to compute entropy, either "ideal" or "sackur-tetrode".
            The former uses original statistical mechanics equations, while the latter
            computes it relative to the entropy of Ar at 298 K.
        debug (bool): whether to print debug information

    Returns:
        float: one-dim translational entropy
    """
    mass = sum(atoms.get_masses())  # amu
    if method == "ideal":
        mass *= units._amu  # kg/molecule
        # Translational entropy (term inside the log is in SI units).
        S_t = (2 * np.pi * mass * units._k * temperature / units._hplanck**2) ** (3.0 / 2)
        S_t *= units._k * temperature / REFERENCEPRESSURE
        S_t = units.kB * (np.log(S_t) + 5.0 / 2.0)
        S_t *= EV_TO_KJ_MOL

    elif method == "sackur-tetrode":
        if debug:
            print(f"mass ratio: {mass/M_ARGON}")
            print(f"temperature ratio: {temperature / SACKUR_TETRODE_TEMP}")
            print(
                f"product: {(mass / M_ARGON) ** 1.5 * (temperature / SACKUR_TETRODE_TEMP) ** 2.5}"
            )
        S_t = np.log(((mass / M_ARGON) ** 1.5) * ((temperature / SACKUR_TETRODE_TEMP) ** 2.5))
        if debug:
            print(f"S_t from ratios: {S_t}")
        S_t *= K_BOLTZMANN
        S_t += S_ARGON

    else:
        raise NotImplementedError

    # divide by three to account for loss of two dimensions of motion
    return S_t / 3  # 1-D translational entropy in kJ/mol/K


def get_fractional_zeo_rot_entropy_exp(
    v_occ: float,
    v_crit: float = V_CRIT_ALT,
) -> float:
    """Get the fractional rotational entropy of a molecule in a zeolite

    Args:
        v_occ (float): the occupiable volume of the zeolite in Angstrom^3
        v_crit (float): the critical volume of the molecule in Angstrom^3. Defaults to
            50.0, the value in my own fit.

    Returns:
        float: the fractional rotational entropy of the molecule in the zeolite
    """
    return (
        F_ROTSLAB
        + np.exp(-ALPHA * (v_occ - v_crit))
        + (
            (1 - F_ROTSLAB - np.exp(-ALPHA * (v_occ - v_crit)))
            / (1 + np.exp(BETA * (v_occ - v_crit)))
        )
    )


def compute_zeo_ads_entropy(
    framework: str,
    entropy_trans: float,
    entropy_rot: float,
    v_crit: float = V_CRITICAL,
    zeo_coeff: float = ZEO_ROT_COEFF,
    func_form: str = "DA",
) -> float:
    """Compute the zeolite adsorption entropy, as estimated by Omar and Paul
    in their entropy paper, Dauenhauer and Abdelrahman, ACS Cent. Sci. (2018).

    Args:
        framework (str): the framework in which the molecule is adsorbed
        entropy_trans (float): the one-dimensional translational entropy of the
            molecule in kJ/mol/K
        entropy_rot (float): the rotational entropy of the molecule in kJ/mol/K
        v_crit (float): the critical volume of the molecule in Angstrom^3. Defaults to
            127.3, the value in Dauenahuer and Abdelrahman's paper.
        zeo_coeff (float): the coefficient for the zeolite rotational entropy term.
            Defaults to 1/7, the value in Dauenahuer and Abdelrahman's paper.
        func_form (str): the functional form of the entropy equation, either "DA" for the form
            in Dauenahuer and Abdelrahman's paper, or "exp" for my own fit. Defaults to "DA".
    """
    v_occ = V_OCC.get(framework, None)
    if func_form == "DA":
        try:
            f_rot_zeo = zeo_coeff * (((1 - (v_crit / v_occ)) ** (-3)) - 1)
        except TypeError:
            return None
    elif func_form == "exp":
        f_rot_zeo = get_fractional_zeo_rot_entropy_exp(v_occ, v_crit)

    return -(entropy_trans + (F_ROTSLAB + f_rot_zeo) * entropy_rot)
