import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import linregress
from pandas import read_csv
from typing import Union

"""
To generate all plots, in the terminal run:
    for i in {1..17}; do echo $i | python main.py -v; done
"""


"""
Set data path. Make sure a directory called "pdfs" exists in working
directory, as plots will be generated here.
"""

path_to_data = "/Users/alistairsterling/ajsterling@lbl.gov - Google Drive/My Drive/Postdoc/Legacy/Onedrive_backup/" \
               "Onedrive-backup/Propellanes/Data/To_upload/"

"""
Parameters for plotting. Changed default font to Helvetica.
"""

fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(8.0, 7.0)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "pdf.fonttype": 3})


def calc_marcus_error(ts_barrier: np.ndarray,
                      reaction_energy: np.ndarray,
                      marcus_intrinsic_barrier: np.ndarray) -> np.ndarray:
    """
    Marcus TS barrier prediction: ∆H^‡_pred = ∆H‡_int + 0.5*∆H_r + ( (∆H_r)**2 / 16*∆H‡_int )
    """

    reaction_energy_sq = reaction_energy * reaction_energy
    marcus_prediction = (marcus_intrinsic_barrier + 0.5 * reaction_energy +
                         (reaction_energy_sq / (16 * marcus_intrinsic_barrier)))
    marcus_error = marcus_prediction - ts_barrier

    return marcus_error


def linear_model(xs: np.ndarray,
                 y: np.ndarray) -> LinearRegression:
    """
    Linear regression model that takes y and stacked multiple xs and returns linear best fit using sklearn
    'LinearRegression' method
    """

    lin_model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
    lin_model.fit(xs, y)

    return lin_model


def predict_xs(lin_model: LinearRegression,
               xs: np.ndarray) -> np.ndarray:
    """
    Uses linear model defined in 'linear_model()' function to predict y values from array xs
    """

    pred = lin_model.predict(xs)

    return pred


def calc_rsq2(lin_model: LinearRegression,
              xs: np.ndarray,
              y: np.ndarray) -> float:
    """
    Returns coefficient of determination (R^2) using the linear model defined in 'linear_model()' function
    """

    model_r_sq2 = lin_model.score(xs, y)

    return model_r_sq2


def calc_rmse(y: np.ndarray,
              pred: np.ndarray) -> float:
    """
    Returns root mean squared error (RMSE) of linear model defined in 'linear_model()' function
    """

    model_rmse = np.sqrt(metrics.mean_squared_error(y, pred))

    return model_rmse


def model_intercept(lin_model: LinearRegression) -> float:
    """
    Returns y intercept of linear model defined in 'linear_model()' function
    """

    model_delh0 = float(lin_model.intercept_)

    return model_delh0


def x_coefs(lin_model: LinearRegression) -> list:
    """
    Returns first x coefficient ( for the linear model defined in linear_model()
    """

    x_coef = lin_model.coef_.reshape((-1, 1))
    model_x_coefs = [float(x_coef[i]) for i in range(len(x_coef))]

    return model_x_coefs


def print_model_coefficients(model_rmse: float,
                             model_r_sq2: float,
                             model_delh0: float,
                             model_x1_coef: float,
                             model_x2_coef: Union[float, None],
                             model_x3_coef: Union[float, None]) -> None:
    """
    Prints values for model RMSE, R^2, ∆H0, and variable coefficients based on linear model defined in linear_model()
    """

    print(f'Root Mean Squared Error: {model_rmse:.2f}')
    print(f'Coefficient of determination: {model_r_sq2:.2f}')
    print(f'Intercept: {model_delh0:.2f}')

    if model_x1_coef is not None:
        print(f'Model coefficient 0: {model_x1_coef:.2f}')
    if model_x2_coef is not None:
        print(f'Model coefficient 1: {model_x2_coef:.2f}')
    if model_x3_coef is not None:
        print(f'Model coefficient 2: {model_x3_coef:.2g}')

    return None


def calc_delta_r_sq2(sm_r: np.ndarray,
                     ts_r: np.ndarray) -> np.ndarray:
    """
    Calculate (∆r‡)^2
    """

    delta_r_sq2 = (ts_r - sm_r) ** 2

    return delta_r_sq2


if __name__ == "__main__":

    """
    Import hydrocarbon + methyl radical data from CSV file (DLPNO-CCSD(T)/def2-QZVPP//B2PLYP-D3BJ/def2-TZVP)
    """

    df_hydrocarbons = np.genfromtxt(path_to_data + r'hydrocarbons_data.csv', delimiter=',', skip_header=1)

    ts_barrier_hydrocarbons = df_hydrocarbons[:, [1]]  # Column 1: TS enthalpy, ∆H‡, in kcal/mol
    reaction_energy_hydrocarbons = df_hydrocarbons[:, [2]]  # Column 2: Reaction enthalpy, ∆H0, in kcal/mol
    elf_hydrocarbons = df_hydrocarbons[:, [3]]  # Column 3: 1-ELF at bond critical point
    two_minus_nocc_hydrocarbons = df_hydrocarbons[:, [4]]  # Column 4: Delocalisation from NBO parameter, 2-Nocc
    n3_hydrocarbons = df_hydrocarbons[:, [5]]  # Column 5: Number of 3-membered rings (n3) fused to breaking bond
    reactant_cc_distance = df_hydrocarbons[:, [6]]  # Column 7: Length of breaking C–C bond in reactant
    ts_cc_distance = df_hydrocarbons[:, [7]]  # Column 8: Length of breaking C–C bond in TS
    homo_energy = df_hydrocarbons[:, [8]]  # Column 9: HOMO energy in Ha
    lumo_energy = df_hydrocarbons[:, [9]]  # Column 10: LUMO energy in Ha
    x_hydrocarbons = []

    """
    Import hydrocarbon + amide anion addition data from CSV file
    """

    df_amide = np.genfromtxt(path_to_data + r'/amide_data.csv', delimiter=',', skip_header=1)

    ts_barrier_amide = df_amide[:, [1]]  # Column 1: TS enthalpy, ∆H‡, in kcal/mol
    reaction_energy_amide = df_amide[:, [2]]  # Column 2: Reaction enthalpy, ∆H0, in kcal/mol
    n3_amide = df_amide[:, [3]]  # Column 3: Number of 3-membered rings (n3) fused to breaking bond
    two_minus_nocc_amide = df_amide[:, [4]]  # Column 4: Delocalisation from NBO parameter, 2-Nocc
    x_amide = []

    """
    Import heterosubstitution data from CSV file (data from 10.1021/jo001412t and 10.1039/b314869f)
    """

    df_hoz_pandas = read_csv(path_to_data + r'/Hoz_data.csv', delimiter=',').to_numpy()
    df_hoz_numpy = np.genfromtxt(path_to_data + r'/Hoz_data.csv', delimiter=',', skip_header=1)

    reaction_type_hoz = df_hoz_pandas[:, [0]]  # Column 1: Reaction type (anionic or radical)
    reaction_energy_hoz = df_hoz_numpy[:, [3]]  # Column 2: Reaction energy, ∆E0, in kcal/mol
    ts_barrier_hoz = df_hoz_numpy[:, [4]]  # Column 3: TS energy, Ea, in kcal/mol
    marcus_intrinsic_barrier_hoz = df_hoz_numpy[:, [5]]  # Column 4: Intrinsic Marcus TS barrier in kcal/mol
    two_minus_nocc_hoz = df_hoz_numpy[:, [8]]  # Column 7: Delocalisation from NBO parameter, 2-Nocc
    n3_hoz = df_hoz_numpy[:, [9]]  # Column 8: Number of 3-membered rings (n3) fused to breaking bond
    heteroatom_type_hoz = df_hoz_pandas[:, [10]]  # Column 9: Type of heteroatom in molecule (C, N or O)
    x_hoz = []

    """
    Import cycloaddition data from CSV file
    """

    df_cycloaddition = np.genfromtxt(path_to_data + r'/cycloaddition_data.csv', delimiter=',', skip_header=1)

    ts_barrier_cycloaddition = df_cycloaddition[:, [1]]  # Column 1: TS enthalpy, ∆H‡, in kcal/mol
    reaction_energy_cycloaddition = df_cycloaddition[:, [2]]  # Column 2: Reaction enthalpy, ∆H0, in kcal/mol
    two_minus_nocc_cycloaddition = df_cycloaddition[:, [3]]  # Column 3: Delocalisation from NBO parameter, mean 2-Nocc
    x_cycloaddition = []

    """
    Choose which plot to generate
    """

    plot_number = input("Plot 1: y = ∆H‡, x0 = ∆H0, x1 = 2-Nocc (CH3• addition)\n"
                        "Plot 2: y = ∆H‡, x0 = ∆H0, x1 = 1-ELF (CH3• addition)\n"
                        "Plot 3: y = ∆H‡, x0 = ∆H0, x1 = n3 (CH3• addition)\n"
                        "Plot 4: y = ∆H‡, x0 = ∆H0 (CH3• addition); ∆Hr vs ∆H‡\n"
                        "Plot 5: y = ∆H‡, x0 = ∆H0 (CH3• addition); ∆H‡ calc vs pred\n"
                        "Plot 6: y = ∆H‡, x0 = ∆H0, x1 = 2-Nocc, x2 = (∆H0)^2 (CH3• addition)\n"
                        "Plot 7: y = ∆H‡, x0 = ∆H0, x1 = 1-ELF, x2 = (∆H0)^2 (CH3• addition)\n"
                        "Plot 8: y = ∆H‡, x0 = (∆r‡)^2 (CH3• addition)\n"
                        "Plot 9: y = ∆H‡, x0 = ∆H0, x1 = n3 (NH2- addition)\n"
                        "Plot 10: y = Marcus Ea error, x0 = n3 (heterosubstitution)\n"
                        "Plot 11: y = ∆H‡, x0 = ∆H0, x1 = (∆H0)^2 (CH3• addition)\n"
                        "Plot 12: y = ∆H‡, x0 = ∆H0, x1 = 2-Nocc (NH2- addition)\n"
                        "Plot 13: y = ∆H‡, x0 = ∆H0 (cycloaddition); ∆H‡ calc vs pred\n"
                        "Plot 14: y = ∆H‡, x0 = ∆H0, x1 = mean 2-Nocc (cycloaddition)\n"
                        "Plot 15: y = ∆H‡, x0 = ∆H0, x1 = E_HOMO (CH3• addition)\n"
                        "Plot 16: y = ∆H‡, x0 = ∆H0, x1 = E_LUMO (CH3• addition)\n"
                        "Plot 17: y = ∆H‡, x0 = ∆H0, x1 = ∆E_HOMO-LUMO (CH3• addition)\n"
                        "Enter plot number to be generated: ")

    if plot_number == "1":

        """
        Generate linear model from TS enthalpy, reaction enthalpy and 2-Nocc
        """

        print(f"Generating plot {plot_number}...")

        x_hydrocarbons = np.column_stack((reaction_energy_hydrocarbons, two_minus_nocc_hydrocarbons))
        model_hydrocarbons = linear_model(x_hydrocarbons, ts_barrier_hydrocarbons)
        predicted_hydrocarbons = predict_xs(model_hydrocarbons, x_hydrocarbons)
        rmse_hydrocarbons = calc_rmse(ts_barrier_hydrocarbons, predicted_hydrocarbons)
        r_sq2_hydrocarbons = calc_rsq2(model_hydrocarbons, x_hydrocarbons, ts_barrier_hydrocarbons)
        delH0_hydrocarbons = model_intercept(model_hydrocarbons)
        x_coefs_hydrocarbons = x_coefs(model_hydrocarbons)
        print_model_coefficients(rmse_hydrocarbons, r_sq2_hydrocarbons, delH0_hydrocarbons, x_coefs_hydrocarbons[0],
                                 x_coefs_hydrocarbons[1], None)

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_hydrocarbons, predicted_hydrocarbons, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_hydrocarbons[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(ts_barrier_hydrocarbons, intercept + slope * ts_barrier_hydrocarbons, color='black')
        ax.plot(ts_barrier_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-2, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(0, 51)
        ax.set_xticks(range(0, 60, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 47, 'R$^2$ = ' f"{r_sq2_hydrocarbons:.2f}", fontsize=20)
        ax.text(1, 44, 'RMSE = ' f"{rmse_hydrocarbons:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 15, '∆H$^{\u2021}_{int}$ = ' f"{delH0_hydrocarbons:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 12, '\u03B1 = ' f"{x_coefs_hydrocarbons[0]:.2}", fontsize=20)
        ax.text(20, 9, '\u03B2 = ' f"{x_coefs_hydrocarbons[1]:.1f}" ' kcal mol$^{-1}$ e$^{-1}$', fontsize=20)

        # plt.savefig(path_to_data + r'/Plot1.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot1.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "2":

        """
        Generate linear model from TS enthalpy, reaction enthalpy and 1-ELF
        """

        print(f"Generating plot {plot_number}...")

        x_hydrocarbons = np.column_stack((ts_barrier_hydrocarbons, reaction_energy_hydrocarbons, elf_hydrocarbons))
        x_hydrocarbons = x_hydrocarbons[~np.isnan(x_hydrocarbons).any(axis=1)]  # remove nan rows in data
        ts_barrier_hydrocarbons, reaction_energy_hydrocarbons, elf_hydrocarbons = (x_hydrocarbons[:, [0]],
                                                                                   x_hydrocarbons[:, [1]],
                                                                                   x_hydrocarbons[:, [2]])
        x_hydrocarbons = np.column_stack((reaction_energy_hydrocarbons, elf_hydrocarbons))

        model_hydrocarbons = linear_model(x_hydrocarbons, ts_barrier_hydrocarbons)
        predicted_hydrocarbons = predict_xs(model_hydrocarbons, x_hydrocarbons)
        rmse_hydrocarbons = calc_rmse(ts_barrier_hydrocarbons, predicted_hydrocarbons)
        r_sq2_hydrocarbons = calc_rsq2(model_hydrocarbons, x_hydrocarbons, ts_barrier_hydrocarbons)
        delH0_hydrocarbons = model_intercept(model_hydrocarbons)
        x_coefs_hydrocarbons = x_coefs(model_hydrocarbons)
        print_model_coefficients(rmse_hydrocarbons, r_sq2_hydrocarbons, delH0_hydrocarbons, x_coefs_hydrocarbons[0],
                                 x_coefs_hydrocarbons[1], None)

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_hydrocarbons, predicted_hydrocarbons, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_hydrocarbons[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(ts_barrier_hydrocarbons, intercept + slope * ts_barrier_hydrocarbons, color='black')
        ax.plot(ts_barrier_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-2, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(0, 51)
        ax.set_xticks(range(0, 60, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 47, 'R$^2$ = ' f"{r_sq2_hydrocarbons:.2f}", fontsize=20)
        ax.text(1, 44, 'RMSE = ' f"{rmse_hydrocarbons:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 15, '∆H$^{\u2021}_{int}$ = ' f"{delH0_hydrocarbons:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 12, '\u03B1 = ' f"{x_coefs_hydrocarbons[0]:.2}", fontsize=20)
        ax.text(20, 9, '\u03B2 = ' f"{x_coefs_hydrocarbons[1]:.1f}" ' kcal mol$^{-1}$', fontsize=20)

        # plt.savefig(path_to_data + r'/Plot2.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot2.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "3":

        """
        Generate linear model from TS enthalpy, reaction enthalpy and n3
        """

        print(f"Generating plot {plot_number}...")

        x_hydrocarbons = np.column_stack((reaction_energy_hydrocarbons, n3_hydrocarbons))
        model_hydrocarbons = linear_model(x_hydrocarbons, ts_barrier_hydrocarbons)
        predicted_hydrocarbons = predict_xs(model_hydrocarbons, x_hydrocarbons)
        rmse_hydrocarbons = calc_rmse(ts_barrier_hydrocarbons, predicted_hydrocarbons)
        r_sq2_hydrocarbons = calc_rsq2(model_hydrocarbons, x_hydrocarbons, ts_barrier_hydrocarbons)
        delH0_hydrocarbons = model_intercept(model_hydrocarbons)
        x_coefs_hydrocarbons = x_coefs(model_hydrocarbons)
        print_model_coefficients(rmse_hydrocarbons, r_sq2_hydrocarbons, delH0_hydrocarbons, x_coefs_hydrocarbons[0],
                                 x_coefs_hydrocarbons[1], None)

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_hydrocarbons, predicted_hydrocarbons, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_hydrocarbons[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(ts_barrier_hydrocarbons, intercept + slope * ts_barrier_hydrocarbons, color='black')
        ax.plot(ts_barrier_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-2, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(0, 51)
        ax.set_xticks(range(0, 60, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 47, 'R$^2$ = ' f"{r_sq2_hydrocarbons:.2f}", fontsize=20)
        ax.text(1, 44, 'RMSE = ' f"{rmse_hydrocarbons:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 15, '∆H$^{\u2021}_{int}$ = ' f"{delH0_hydrocarbons:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 12, '\u03B1 = ' f"{x_coefs_hydrocarbons[0]:.2}", fontsize=20)
        ax.text(20, 9, '\u03B2 = ' f"{x_coefs_hydrocarbons[1]:.1f}" ' kcal mol$^{-1}$', fontsize=20)

        # plt.savefig(path_to_data + r'/Plot3.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot3.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "4":

        """
        Generate linear model from TS enthalpy and reaction enthalpy (∆Hr vs ∆H‡)
        """

        print(f"Generating plot {plot_number}...")

        x = reaction_energy_hydrocarbons

        model_hydrocarbons_linear = linear_model(x, ts_barrier_hydrocarbons)
        predicted_hydrocarbons_linear = predict_xs(model_hydrocarbons_linear, x)
        rmse_hydrocarbons_linear = calc_rmse(ts_barrier_hydrocarbons, predicted_hydrocarbons_linear)
        r_sq2_hydrocarbons_linear = calc_rsq2(model_hydrocarbons_linear, x, ts_barrier_hydrocarbons)
        delH0_hydrocarbons_linear = model_intercept(model_hydrocarbons_linear)
        x_coefs_hydrocarbons_linear = x_coefs(model_hydrocarbons_linear)
        print_model_coefficients(rmse_hydrocarbons_linear, r_sq2_hydrocarbons_linear, delH0_hydrocarbons_linear,
                                 x_coefs_hydrocarbons_linear[0], None, None)

        """
        Plot ∆H‡ vs ∆Hr for hydrocarbon data
        """

        ax.scatter(x, ts_barrier_hydrocarbons, color='#00899d', s=40)

        ax.set_ylabel('∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('∆H$_r$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_hydrocarbons_linear[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(x, delH0_hydrocarbons_linear + x_coefs_hydrocarbons_linear[0] * x, color='grey')
        # ax.plot(ts_barrier_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-3, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(-91, 1)
        ax.set_xticks(range(-90, 10, 20))
        ax.tick_params(labelsize=20)

        ax.text(-89, 47, 'R$^2$ = ' f"{r_sq2_hydrocarbons_linear:.2f}", fontsize=20)
        ax.text(-89, 44, 'RMSE = ' f"{rmse_hydrocarbons_linear:.1f}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(-89, 41, '∆H$^{\u2021}_{int}$ = ' f"{delH0_hydrocarbons_linear:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(-89, 38, '\u03B1 = ' f"{x_coefs_hydrocarbons_linear[0]:.2}", fontsize=20)

        # plt.savefig(path_to_data + r'/Plot4.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot4.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "5":

        """
        Generate linear model from TS enthalpy and reaction enthalpy (∆H‡ predicted vs calculated)
        """

        print(f"Generating plot {plot_number}...")

        x = reaction_energy_hydrocarbons

        model_hydrocarbons_linear = linear_model(x, ts_barrier_hydrocarbons)
        predicted_hydrocarbons_linear = predict_xs(model_hydrocarbons_linear, x)
        rmse_hydrocarbons_linear = calc_rmse(ts_barrier_hydrocarbons, predicted_hydrocarbons_linear)
        r_sq2_hydrocarbons_linear = calc_rsq2(model_hydrocarbons_linear, x, ts_barrier_hydrocarbons)
        delH0_hydrocarbons_linear = model_intercept(model_hydrocarbons_linear)
        x_coefs_hydrocarbons_linear = x_coefs(model_hydrocarbons_linear)
        print_model_coefficients(rmse_hydrocarbons_linear, r_sq2_hydrocarbons_linear, delH0_hydrocarbons_linear,
                                 x_coefs_hydrocarbons_linear[0], None, None)

        """
        Plot predicted vs calculated ∆H‡ for hydrocarbon data
        """

        ax.scatter(ts_barrier_hydrocarbons, predicted_hydrocarbons_linear, color='grey', s=40)

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_hydrocarbons_linear[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(ts_barrier_hydrocarbons, intercept + slope * ts_barrier_hydrocarbons, color='black')
        ax.plot(ts_barrier_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-2, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(0, 51)
        ax.set_xticks(range(0, 60, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 47, 'R$^2$ = ' f"{r_sq2_hydrocarbons_linear:.2f}", fontsize=20)
        ax.text(1, 44, 'RMSE = ' f"{rmse_hydrocarbons_linear:.1f}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(1, 41, '∆H$^{\u2021}_{int}$ = ' f"{delH0_hydrocarbons_linear:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(1, 38, '\u03B1 = ' f"{x_coefs_hydrocarbons_linear[0]:.2}", fontsize=20)

        # plt.savefig(path_to_data + r'/Plot5.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot5.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "6":

        """
        Generate linear model from TS enthalpy, reaction enthalpy, 2-Nocc and squared reaction enthalpy
        """

        print(f"Generating plot {plot_number}...")

        reaction_energy_hydrocarbons_sq2 = reaction_energy_hydrocarbons * reaction_energy_hydrocarbons
        x = np.column_stack((reaction_energy_hydrocarbons,
                             two_minus_nocc_hydrocarbons,
                             reaction_energy_hydrocarbons_sq2))

        model_hydrocarbons_marcus = linear_model(x,
                                                 ts_barrier_hydrocarbons)
        predicted_hydrocarbons_marcus = predict_xs(model_hydrocarbons_marcus,
                                                   x)
        rmse_hydrocarbons_marcus = calc_rmse(ts_barrier_hydrocarbons,
                                             predicted_hydrocarbons_marcus)
        r_sq2_hydrocarbons_marcus = calc_rsq2(model_hydrocarbons_marcus,
                                              x,
                                              ts_barrier_hydrocarbons)
        delH0_hydrocarbons_marcus = model_intercept(model_hydrocarbons_marcus)
        x_coefs_hydrocarbons_marcus = x_coefs(model_hydrocarbons_marcus)
        print_model_coefficients(rmse_hydrocarbons_marcus,
                                 r_sq2_hydrocarbons_marcus,
                                 delH0_hydrocarbons_marcus,
                                 x_coefs_hydrocarbons_marcus[0],
                                 x_coefs_hydrocarbons_marcus[1],
                                 x_coefs_hydrocarbons_marcus[2])

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_hydrocarbons, predicted_hydrocarbons_marcus, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_hydrocarbons_marcus[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(ts_barrier_hydrocarbons, intercept + slope * ts_barrier_hydrocarbons, color='black')
        ax.plot(ts_barrier_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-2, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(0, 51)
        ax.set_xticks(range(0, 60, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 47, 'R$^2$ = ' f"{r_sq2_hydrocarbons_marcus:.2f}", fontsize=20)
        ax.text(1, 44, 'RMSE = ' f"{rmse_hydrocarbons_marcus:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 15, '∆H$^{\u2021}_{int}$ = ' f"{delH0_hydrocarbons_marcus:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 12, '\u03B1 = ' f"{x_coefs_hydrocarbons_marcus[0]:.2f}", fontsize=20)
        ax.text(20, 9, '\u03B2 = ' f"{x_coefs_hydrocarbons_marcus[1]:.1f}" ' kcal mol$^{-1}$ e$^{-1}$', fontsize=20)
        ax.text(20, 6, '\u03B3 = ' f"{x_coefs_hydrocarbons_marcus[2]:.3f}" ' mol kcal$^{-1}$', fontsize=20)

        # plt.savefig(path_to_data + r'/Plot6.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot6.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "7":

        """
        Generate linear model from TS enthalpy, reaction enthalpy, 1-ELF and squared reaction enthalpy
        """

        print(f"Generating plot {plot_number}...")

        reaction_energy_hydrocarbons_sq2 = reaction_energy_hydrocarbons * reaction_energy_hydrocarbons
        x_hydrocarbons = np.column_stack((ts_barrier_hydrocarbons,
                                          reaction_energy_hydrocarbons,
                                          elf_hydrocarbons,
                                          reaction_energy_hydrocarbons_sq2))
        x_hydrocarbons = x_hydrocarbons[~np.isnan(x_hydrocarbons).any(axis=1)]  # remove nan rows in data
        ts_barrier_hydrocarbons, reaction_energy_hydrocarbons, elf_hydrocarbons, reaction_energy_hydrocarbons_sq2 \
            = x_hydrocarbons[:, [0]], x_hydrocarbons[:, [1]], x_hydrocarbons[:, [2]], x_hydrocarbons[:, [3]]
        x_hydrocarbons = np.column_stack((reaction_energy_hydrocarbons,
                                          elf_hydrocarbons,
                                          reaction_energy_hydrocarbons_sq2))

        model_hydrocarbons_marcus = linear_model(x_hydrocarbons,
                                                 ts_barrier_hydrocarbons)
        predicted_hydrocarbons_marcus = predict_xs(model_hydrocarbons_marcus,
                                                   x_hydrocarbons)
        rmse_hydrocarbons_marcus = calc_rmse(ts_barrier_hydrocarbons,
                                             predicted_hydrocarbons_marcus)
        r_sq2_hydrocarbons_marcus = calc_rsq2(model_hydrocarbons_marcus,
                                              x_hydrocarbons,
                                              ts_barrier_hydrocarbons)
        delH0_hydrocarbons_marcus = model_intercept(model_hydrocarbons_marcus)
        x_coefs_hydrocarbons_marcus = x_coefs(model_hydrocarbons_marcus)
        print_model_coefficients(rmse_hydrocarbons_marcus,
                                 r_sq2_hydrocarbons_marcus,
                                 delH0_hydrocarbons_marcus,
                                 x_coefs_hydrocarbons_marcus[0],
                                 x_coefs_hydrocarbons_marcus[1],
                                 x_coefs_hydrocarbons_marcus[2])

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_hydrocarbons, predicted_hydrocarbons_marcus, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_hydrocarbons_marcus[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(ts_barrier_hydrocarbons, intercept + slope * ts_barrier_hydrocarbons, color='black')
        ax.plot(ts_barrier_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-2, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(0, 51)
        ax.set_xticks(range(0, 60, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 47, 'R$^2$ = ' f"{r_sq2_hydrocarbons_marcus:.2f}", fontsize=20)
        ax.text(1, 44, 'RMSE = ' f"{rmse_hydrocarbons_marcus:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 15, '∆H$^{\u2021}_{int}$ = ' f"{delH0_hydrocarbons_marcus:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 12, '\u03B1 = ' f"{x_coefs_hydrocarbons_marcus[0]:.2f}", fontsize=20)
        ax.text(20, 9, '\u03B2 = ' f"{x_coefs_hydrocarbons_marcus[1]:.1f}" ' kcal mol$^{-1}$ e$^{-1}$', fontsize=20)
        ax.text(20, 6, '\u03B3 = ' f"{x_coefs_hydrocarbons_marcus[2]:.3f}" ' mol kcal$^{-1}$', fontsize=20)

        # plt.savefig(path_to_data + r'/Plot7.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot7.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "8":

        """
        Generate linear model from TS barrier and (∆r‡)^2
        """

        print(f"Generating plot {plot_number}...")

        delta_r_sq2_hydrocarbons = calc_delta_r_sq2(reactant_cc_distance,
                                                    ts_cc_distance)

        model_delta_r_sq2 = linear_model(delta_r_sq2_hydrocarbons,
                                         ts_barrier_hydrocarbons)
        predicted_delta_r_sq2 = predict_xs(model_delta_r_sq2,
                                           delta_r_sq2_hydrocarbons)
        rmse_delta_r_sq2 = calc_rmse(ts_barrier_hydrocarbons,
                                     predicted_delta_r_sq2)
        r_sq2_delta_r_sq2 = calc_rsq2(model_delta_r_sq2,
                                      delta_r_sq2_hydrocarbons,
                                      ts_barrier_hydrocarbons)
        delH0_delta_r_sq2 = model_intercept(model_delta_r_sq2)
        x_coefs_delta_r_sq2 = x_coefs(model_delta_r_sq2)
        print_model_coefficients(rmse_delta_r_sq2,
                                 r_sq2_delta_r_sq2,
                                 delH0_delta_r_sq2,
                                 x_coefs_delta_r_sq2[0],
                                 None,
                                 None)

        """
        Plot (∆r‡)^2 vs ∆H‡
        """

        ax.scatter(delta_r_sq2_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d')

        ax.set_ylabel("∆H$^‡$ / kcal mol$^{-1}$", fontsize=20)
        ax.set_xlabel(r"$(∆r^‡)^2$ / $\AA^2$", fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_delta_r_sq2[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(delta_r_sq2_hydrocarbons, delH0_delta_r_sq2 + x_coefs_delta_r_sq2[0] * delta_r_sq2_hydrocarbons,
                color='grey')

        ax.set_ylim(-3, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(-0.01, 0.21)
        ax.set_xticks(np.arange(0, 0.25, 0.05))
        ax.tick_params(labelsize=20)

        ax.text(0.00, 46, 'R$^2$ = ' f"{r_sq2_delta_r_sq2:.2f}", fontsize=20)
        ax.text(0.06, 9, '∆H$^{\u2021}$ = ' f"{x_coefs_delta_r_sq2[0]:.1f}" '$(∆r^‡)^2 -$ '
                         f"{abs(delH0_delta_r_sq2):.1f}", fontsize=20)

        # plt.savefig(path_to_data + r'/Plot8.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot8.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "9":

        """
        Generate linear model from TS enthalpy, reaction enthalpy and n3 for NH2- addition
        """

        print(f"Generating plot {plot_number}...")

        x = np.column_stack((reaction_energy_amide,
                             n3_amide))

        model_amide = linear_model(x,
                                   ts_barrier_amide)
        predicted_amide = predict_xs(model_amide,
                                     x)
        rmse_amide = calc_rmse(ts_barrier_amide,
                               predicted_amide)
        r_sq2_amide = calc_rsq2(model_amide,
                                x,
                                ts_barrier_amide)
        delH0_amide = model_intercept(model_amide)
        x_coefs_amide = x_coefs(model_amide)
        print_model_coefficients(rmse_amide,
                                 r_sq2_amide,
                                 delH0_amide,
                                 x_coefs_amide[0],
                                 x_coefs_amide[1],
                                 None)

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_amide, predicted_amide, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_amide[:, 0], predicted_amide[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(ts_barrier_amide, intercept + slope * ts_barrier_amide, color='black')
        ax.plot(ts_barrier_amide, ts_barrier_amide, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(0, 71)
        ax.set_yticks(range(0, 80, 10))
        ax.set_xlim(0, 71)
        ax.set_xticks(range(0, 80, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 66, 'R$^2$ = ' f"{r_sq2_amide:.2f}", fontsize=20)
        ax.text(1, 62, 'RMSE = ' f"{rmse_amide:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(30, 15, '∆H$^{\u2021}_{int}$ = ' f"{delH0_amide:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(30, 11, '\u03B1 = ' f"{x_coefs_amide[0]:.2f}", fontsize=20)
        ax.text(30, 7, '\u03B2 = ' f"{x_coefs_amide[1]:.1f}" ' kcal mol$^{-1}$ e$^{-1}$', fontsize=20)

        # plt.savefig(path_to_data + r'/Plot9.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot9.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "10":

        """
        Generate linear model from TS enthalpy error (Marcus) and n3 for heterosubstituted systems
        Data from 10.1021/jo001412t and 10.1039/b314869f
        """

        print(f"Generating plot {plot_number}...")

        marcus_error_hoz = calc_marcus_error(ts_barrier_hoz,
                                             reaction_energy_hoz,
                                             marcus_intrinsic_barrier_hoz)
        marcus_model_hoz = linear_model(n3_hoz,
                                        marcus_error_hoz)
        marcus_predicted_hoz = predict_xs(marcus_model_hoz,
                                          n3_hoz)
        marcus_rmse_hoz = calc_rmse(marcus_error_hoz,
                                    marcus_predicted_hoz)
        marcus_r_sq2_hoz = calc_rsq2(marcus_model_hoz,
                                     n3_hoz,
                                     marcus_error_hoz)
        marcus_intercept_hoz = model_intercept(marcus_model_hoz)
        marcus_x_coefs_hoz = x_coefs(marcus_model_hoz)
        print_model_coefficients(marcus_rmse_hoz,
                                 marcus_r_sq2_hoz,
                                 marcus_intercept_hoz,
                                 marcus_x_coefs_hoz[0],
                                 None,
                                 None)

        """
        Plot Marcus Ea error vs 2-Nocc and add least squares regression line
        """

        heteroatoms = heteroatom_type_hoz

        colour_list = ['darkgrey', 'red', 'deepskyblue', 'mediumorchid', 'orange']

        marker_list = ['o', 'x']

        reaction_type = reaction_type_hoz


        def x_value_for_reaction_type(arr, reaction_types):
            return [x_value for index, x_value in enumerate(arr) if reaction_type[index] == reaction_types]


        def x_value_for_rxn_heteroatom(arr, reaction_types, heteroatom):
            return [x_value for index, x_value in enumerate(arr) if reaction_type[index] == reaction_types and heteroatoms[index] == heteroatom]


        for idx, rxn_type in enumerate(('anionic', 'radical')):
            for jdx, symbol in enumerate(('C', 'O', 'N', 'P', 'S')):
                ax.scatter(x_value_for_rxn_heteroatom(marcus_predicted_hoz, reaction_types=rxn_type, heteroatom=symbol),
                           x_value_for_rxn_heteroatom(marcus_error_hoz, reaction_types=rxn_type, heteroatom=symbol),
                           color=colour_list[jdx],
                           marker=marker_list[idx])

        ax.set_ylabel('Marcus $E_a$ error / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Predicted $E_a$ error / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(marcus_predicted_hoz[:, 0], marcus_error_hoz[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(marcus_predicted_hoz, intercept + slope * marcus_predicted_hoz, color='black')
        # ax.plot(marcus_predicted_hoz, marcus_predicted_hoz, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-3, 22)
        ax.set_yticks(range(0, 25, 5))
        ax.set_xlim(-3, 22)
        ax.set_xticks(range(0, 25, 5))
        ax.tick_params(labelsize=20)

        ax.text(-2, 20, 'R$^2$ = ' f"{marcus_r_sq2_hoz:.2f}", fontsize=20)
        ax.text(-2, 18, 'RMSE = ' f"{marcus_rmse_hoz:.2}" ' kcal mol$^{-1}$', fontsize=20)
        plt.text(-2, 15.65, '•', fontsize=30, color='black')
        plt.text(-1, 16, 'anionic', fontsize=20, color='black')
        plt.text(-2, 14, r'$\times$', fontsize=15, color='black')
        plt.text(-1, 14, 'radical', fontsize=20, color='black')
        ax.text(4, 1,
                'Predicted $E_a$ error = ' f"{marcus_x_coefs_hoz[0]:.2}" '$n_3 +$ ' f"{abs(marcus_intercept_hoz):.1f}",
                fontsize=20)

        plt.figtext(0.225, 0.90, '•', fontsize=30, color='darkgrey')
        plt.figtext(0.250, 0.91, 'C–C', fontsize=20)
        plt.figtext(0.350, 0.90, '•', fontsize=30, color='deepskyblue')
        plt.figtext(0.375, 0.91, 'C–N', fontsize=20)
        plt.figtext(0.475, 0.90, '•', fontsize=30, color='red')
        plt.figtext(0.500, 0.91, 'C–O', fontsize=20)
        plt.figtext(0.600, 0.90, '•', fontsize=30, color='orange')
        plt.figtext(0.625, 0.91, 'C–S', fontsize=20)
        plt.figtext(0.725, 0.90, '•', fontsize=30, color='mediumorchid')
        plt.figtext(0.750, 0.91, 'C–P', fontsize=20)

        # plt.savefig(path_to_data + r'/Plot10.png', dpi=300)
        # plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot10.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "11":

        """
        Generate linear model from TS enthalpy, reaction enthalpy, and squared reaction enthalpy
        (i.e. Marcus equation)
        """

        print(f"Generating plot {plot_number}...")

        reaction_energy_hydrocarbons_sq2 = reaction_energy_hydrocarbons * reaction_energy_hydrocarbons
        x = np.column_stack((reaction_energy_hydrocarbons,
                             reaction_energy_hydrocarbons_sq2))

        model_hydrocarbons_marcus = linear_model(x,
                                                 ts_barrier_hydrocarbons)
        predicted_hydrocarbons_marcus = predict_xs(model_hydrocarbons_marcus,
                                                   x)
        rmse_hydrocarbons_marcus = calc_rmse(ts_barrier_hydrocarbons,
                                             predicted_hydrocarbons_marcus)
        r_sq2_hydrocarbons_marcus = calc_rsq2(model_hydrocarbons_marcus,
                                              x,
                                              ts_barrier_hydrocarbons)
        delH0_hydrocarbons_marcus = model_intercept(model_hydrocarbons_marcus)
        x_coefs_hydrocarbons_marcus = x_coefs(model_hydrocarbons_marcus)
        print_model_coefficients(rmse_hydrocarbons_marcus,
                                 r_sq2_hydrocarbons_marcus,
                                 delH0_hydrocarbons_marcus,
                                 x_coefs_hydrocarbons_marcus[0],
                                 x_coefs_hydrocarbons_marcus[1],
                                 None)

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_hydrocarbons, predicted_hydrocarbons_marcus, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_hydrocarbons_marcus[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(ts_barrier_hydrocarbons, intercept + slope * ts_barrier_hydrocarbons, color='black')
        ax.plot(ts_barrier_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-2, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(0, 51)
        ax.set_xticks(range(0, 60, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 47, 'R$^2$ = ' f"{r_sq2_hydrocarbons_marcus:.2f}", fontsize=20)
        ax.text(1, 44, 'RMSE = ' f"{rmse_hydrocarbons_marcus:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(1, 41, '∆H$^{\u2021}_{int}$ = ' f"{delH0_hydrocarbons_marcus:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(1, 38, '\u03B1 = ' f"{x_coefs_hydrocarbons_marcus[0]:.2f}", fontsize=20)
        ax.text(1, 35, '\u03B3 = ' f"{x_coefs_hydrocarbons_marcus[1]:.3f}" ' mol kcal$^{-1}$ e$^{-1}$', fontsize=20)

        # plt.savefig(path_to_data + r'/Plot11.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot11.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "12":

        """
        Generate linear model from TS enthalpy, reaction enthalpy and 2-Nocc for NH2- addition
        """

        print(f"Generating plot {plot_number}...")

        x = np.column_stack((reaction_energy_amide,
                             two_minus_nocc_amide))

        model_amide = linear_model(x,
                                   ts_barrier_amide)
        predicted_amide = predict_xs(model_amide,
                                     x)
        rmse_amide = calc_rmse(ts_barrier_amide,
                               predicted_amide)
        r_sq2_amide = calc_rsq2(model_amide,
                                x,
                                ts_barrier_amide)
        delH0_amide = model_intercept(model_amide)
        x_coefs_amide = x_coefs(model_amide)
        print_model_coefficients(rmse_amide,
                                 r_sq2_amide,
                                 delH0_amide,
                                 x_coefs_amide[0],
                                 x_coefs_amide[1],
                                 None)

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_amide, predicted_amide, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_amide[:, 0], predicted_amide[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(ts_barrier_amide, intercept + slope * ts_barrier_amide, color='black')
        ax.plot(ts_barrier_amide, ts_barrier_amide, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(0, 71)
        ax.set_yticks(range(0, 80, 10))
        ax.set_xlim(0, 71)
        ax.set_xticks(range(0, 80, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 66, 'R$^2$ = ' f"{r_sq2_amide:.2f}", fontsize=20)
        ax.text(1, 62, 'RMSE = ' f"{rmse_amide:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(30, 15, '∆H$^{\u2021}_{int}$ = ' f"{delH0_amide:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(30, 11, '\u03B1 = ' f"{x_coefs_amide[0]:.2}", fontsize=20)
        ax.text(30, 7, '\u03B2 = ' f"{x_coefs_amide[1]:.1f}" ' kcal mol$^{-1}$ e$^{-1}$', fontsize=20)

        # plt.savefig(path_to_data + r'/Plot12.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot12.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "13":

        """
        Generate linear model from TS enthalpy and reaction enthalpy (∆H‡ predicted vs calculated)
        """

        print(f"Generating plot {plot_number}...")

        x = reaction_energy_cycloaddition

        model_cycloaddition_linear = linear_model(x,
                                                  ts_barrier_cycloaddition)
        predicted_cycloaddition_linear = predict_xs(model_cycloaddition_linear,
                                                    x)
        rmse_cycloaddition_linear = calc_rmse(ts_barrier_cycloaddition,
                                              predicted_cycloaddition_linear)
        r_sq2_cycloaddition_linear = calc_rsq2(model_cycloaddition_linear,
                                               x,
                                               ts_barrier_cycloaddition)
        delH0_cycloaddition_linear = model_intercept(model_cycloaddition_linear)
        x_coefs_cycloaddition_linear = x_coefs(model_cycloaddition_linear)
        print_model_coefficients(rmse_cycloaddition_linear,
                                 r_sq2_cycloaddition_linear,
                                 delH0_cycloaddition_linear,
                                 x_coefs_cycloaddition_linear[0],
                                 None,
                                 None)

        """
        Plot predicted vs calculated ∆H‡ for hydrocarbon data
        """

        ax.scatter(ts_barrier_cycloaddition, predicted_cycloaddition_linear, color='grey', s=40)

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_cycloaddition[:, 0],
                                                                 predicted_cycloaddition_linear[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(ts_barrier_cycloaddition, intercept + slope * ts_barrier_cycloaddition, color='black')
        ax.plot(ts_barrier_cycloaddition, ts_barrier_cycloaddition, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(0, 20)
        ax.set_yticks(range(0, 25, 5))
        ax.set_xlim(0, 20)
        ax.set_xticks(range(0, 25, 5))
        ax.tick_params(labelsize=20)

        ax.text(0.5, 18.5, 'R$^2$ = ' f"{r_sq2_cycloaddition_linear:.2f}", fontsize=20)
        ax.text(0.5, 17.3, 'RMSE = ' f"{rmse_cycloaddition_linear:.1f}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(0.5, 16.1, '∆H$^{\u2021}_{int}$ = ' f"{delH0_cycloaddition_linear:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(0.5, 14.9, '\u03B1 = ' f"{x_coefs_cycloaddition_linear[0]:.2}", fontsize=20)

        # plt.savefig(path_to_data + r'/Plot13.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot13.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "14":

        """
        Generate linear model from TS enthalpy, reaction enthalpy and 2-Nocc
        """

        print(f"Generating plot {plot_number}...")

        x_cycloaddition = np.column_stack((reaction_energy_cycloaddition,
                                           two_minus_nocc_cycloaddition))
        model_cycloaddition = linear_model(x_cycloaddition,
                                           ts_barrier_cycloaddition)
        predicted_cycloaddition = predict_xs(model_cycloaddition,
                                             x_cycloaddition)
        rmse_cycloaddition = calc_rmse(ts_barrier_cycloaddition,
                                       predicted_cycloaddition)
        r_sq2_cycloaddition = calc_rsq2(model_cycloaddition,
                                        x_cycloaddition,
                                        ts_barrier_cycloaddition)
        delH0_cycloaddition = model_intercept(model_cycloaddition)
        x_coefs_cycloaddition = x_coefs(model_cycloaddition)
        print_model_coefficients(rmse_cycloaddition,
                                 r_sq2_cycloaddition,
                                 delH0_cycloaddition,
                                 x_coefs_cycloaddition[0],
                                 x_coefs_cycloaddition[1],
                                 None)

        print(ts_barrier_cycloaddition, predicted_cycloaddition)

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_cycloaddition, predicted_cycloaddition, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_cycloaddition[:, 0],
                                                                 predicted_cycloaddition[:, 0])
        print(f'P-value: {p_value:.2g}')
        ax.plot(ts_barrier_cycloaddition, intercept + slope * ts_barrier_cycloaddition, color='black')
        ax.plot(ts_barrier_cycloaddition, ts_barrier_cycloaddition, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(0, 20)
        ax.set_yticks(range(0, 25, 5))
        ax.set_xlim(0, 20)
        ax.set_xticks(range(0, 25, 5))
        ax.tick_params(labelsize=20)

        ax.text(0.5, 18.5, 'R$^2$ = ' f"{r_sq2_cycloaddition:.2f}", fontsize=20)
        ax.text(0.5, 17.3, 'RMSE = ' f"{rmse_cycloaddition:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(0.5, 16.1, '∆H$^{\u2021}_{int}$ = ' f"{delH0_cycloaddition:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(0.5, 14.9, '\u03B1 = ' f"{x_coefs_cycloaddition[0]:.2}", fontsize=20)
        ax.text(0.5, 13.7, '\u03B2 = ' f"{x_coefs_cycloaddition[1]:.1f}" ' kcal mol$^{-1}$ e$^{-1}$', fontsize=20)

        # plt.savefig(path_to_data + r'/Plot14.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot14.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "15":

        """
        Generate linear model from TS enthalpy, reaction enthalpy and HOMO energy
        """

        print(f"Generating plot {plot_number}...")

        homo_energy_kcal = homo_energy * 627.509

        x_hydrocarbons = np.column_stack((reaction_energy_hydrocarbons,
                                          homo_energy_kcal))
        model_hydrocarbons = linear_model(x_hydrocarbons,
                                          ts_barrier_hydrocarbons)
        predicted_hydrocarbons = predict_xs(model_hydrocarbons,
                                            x_hydrocarbons)
        rmse_hydrocarbons = calc_rmse(ts_barrier_hydrocarbons,
                                      predicted_hydrocarbons)
        r_sq2_hydrocarbons = calc_rsq2(model_hydrocarbons,
                                       x_hydrocarbons,
                                       ts_barrier_hydrocarbons)
        delH0_hydrocarbons = model_intercept(model_hydrocarbons)
        x_coefs_hydrocarbons = x_coefs(model_hydrocarbons)
        print_model_coefficients(rmse_hydrocarbons,
                                 r_sq2_hydrocarbons,
                                 delH0_hydrocarbons,
                                 x_coefs_hydrocarbons[0],
                                 x_coefs_hydrocarbons[1],
                                 None)

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_hydrocarbons, predicted_hydrocarbons, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_hydrocarbons[:, 0])
        print('P-value:', p_value)
        ax.plot(ts_barrier_hydrocarbons, intercept + slope * ts_barrier_hydrocarbons, color='black')
        ax.plot(ts_barrier_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-2, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(0, 51)
        ax.set_xticks(range(0, 60, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 47, 'R$^2$ = ' f"{r_sq2_hydrocarbons:.2f}", fontsize=20)
        ax.text(1, 44, 'RMSE = ' f"{rmse_hydrocarbons:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 10, '∆H$^{\u2021}_{int}$ = ' f"{delH0_hydrocarbons:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 7, '\u03B1 = ' f"{x_coefs_hydrocarbons[0]:.2}", fontsize=20)
        ax.text(20, 4, '\u03B2 = ' f"{x_coefs_hydrocarbons[1]:.1f}", fontsize=20)

        # plt.savefig(path_to_data + r'/Plot15.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot15.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "16":

        """
        Generate linear model from TS enthalpy, reaction enthalpy and LUMO energy
        """

        print(f"Generating plot {plot_number}...")

        lumo_energy_kcal = lumo_energy * 627.509

        x_hydrocarbons = np.column_stack((reaction_energy_hydrocarbons,
                                          lumo_energy_kcal))
        model_hydrocarbons = linear_model(x_hydrocarbons,
                                          ts_barrier_hydrocarbons)
        predicted_hydrocarbons = predict_xs(model_hydrocarbons,
                                            x_hydrocarbons)
        rmse_hydrocarbons = calc_rmse(ts_barrier_hydrocarbons,
                                      predicted_hydrocarbons)
        r_sq2_hydrocarbons = calc_rsq2(model_hydrocarbons,
                                       x_hydrocarbons,
                                       ts_barrier_hydrocarbons)
        delH0_hydrocarbons = model_intercept(model_hydrocarbons)
        x_coefs_hydrocarbons = x_coefs(model_hydrocarbons)
        print_model_coefficients(rmse_hydrocarbons,
                                 r_sq2_hydrocarbons,
                                 delH0_hydrocarbons,
                                 x_coefs_hydrocarbons[0],
                                 x_coefs_hydrocarbons[1],
                                 None)

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_hydrocarbons, predicted_hydrocarbons, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_hydrocarbons[:, 0])
        print('P-value:', p_value)
        ax.plot(ts_barrier_hydrocarbons, intercept + slope * ts_barrier_hydrocarbons, color='black')
        ax.plot(ts_barrier_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-2, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(0, 51)
        ax.set_xticks(range(0, 60, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 47, 'R$^2$ = ' f"{r_sq2_hydrocarbons:.2f}", fontsize=20)
        ax.text(1, 44, 'RMSE = ' f"{rmse_hydrocarbons:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 10, '∆H$^{\u2021}_{int}$ = ' f"{delH0_hydrocarbons:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 7, '\u03B1 = ' f"{x_coefs_hydrocarbons[0]:.2}", fontsize=20)
        ax.text(20, 4, '\u03B2 = ' f"{x_coefs_hydrocarbons[1]:.1f}", fontsize=20)

        # plt.savefig(path_to_data + r'/Plot16.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot16.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")

    if plot_number == "17":

        """
        Generate linear model from TS enthalpy, reaction enthalpy and HOMO-LUMO gap
        """

        print(f"Generating plot {plot_number}...")

        homo_lumo_gap = lumo_energy - homo_energy
        homo_lumo_gap_kcal = homo_lumo_gap * 627.509

        x_hydrocarbons = np.column_stack((reaction_energy_hydrocarbons,
                                          homo_lumo_gap_kcal))
        model_hydrocarbons = linear_model(x_hydrocarbons,
                                          ts_barrier_hydrocarbons)
        predicted_hydrocarbons = predict_xs(model_hydrocarbons,
                                            x_hydrocarbons)
        rmse_hydrocarbons = calc_rmse(ts_barrier_hydrocarbons,
                                      predicted_hydrocarbons)
        r_sq2_hydrocarbons = calc_rsq2(model_hydrocarbons,
                                       x_hydrocarbons,
                                       ts_barrier_hydrocarbons)
        delH0_hydrocarbons = model_intercept(model_hydrocarbons)
        x_coefs_hydrocarbons = x_coefs(model_hydrocarbons)
        print_model_coefficients(rmse_hydrocarbons,
                                 r_sq2_hydrocarbons,
                                 delH0_hydrocarbons,
                                 x_coefs_hydrocarbons[0],
                                 x_coefs_hydrocarbons[1],
                                 None)

        """
        Plot predicted vs calculated ∆H‡
        """

        ax.scatter(ts_barrier_hydrocarbons, predicted_hydrocarbons, color='grey')

        ax.set_ylabel('Predicted ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)
        ax.set_xlabel('Calculated ∆H$^‡$ / kcal mol$^{-1}$', fontsize=20)

        slope, intercept, r_value, p_value, std_err = linregress(ts_barrier_hydrocarbons[:, 0],
                                                                 predicted_hydrocarbons[:, 0])
        print('P-value:', p_value)
        ax.plot(ts_barrier_hydrocarbons, intercept + slope * ts_barrier_hydrocarbons, color='black')
        ax.plot(ts_barrier_hydrocarbons, ts_barrier_hydrocarbons, color='#00899d', ls='--', dashes=(5, 5))

        ax.set_ylim(-2, 51)
        ax.set_yticks(range(0, 60, 10))
        ax.set_xlim(0, 51)
        ax.set_xticks(range(0, 60, 10))
        ax.tick_params(labelsize=20)

        ax.text(1, 47, 'R$^2$ = ' f"{r_sq2_hydrocarbons:.2f}", fontsize=20)
        ax.text(1, 44, 'RMSE = ' f"{rmse_hydrocarbons:.2}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 10, '∆H$^{\u2021}_{int}$ = ' f"{delH0_hydrocarbons:.3}" ' kcal mol$^{-1}$', fontsize=20)
        ax.text(20, 7, '\u03B1 = ' f"{x_coefs_hydrocarbons[0]:.2}", fontsize=20)
        ax.text(20, 4, '\u03B2 = ' f"{x_coefs_hydrocarbons[1]:.1f}", fontsize=20)

        # plt.savefig(path_to_data + r'/Plot17.png', dpi=300)
        plt.tight_layout()
        plt.savefig(path_to_data + r'pdfs/Plot17.pdf')
        # plt.show()
        print(f"Plot number {plot_number} has been generated!")
