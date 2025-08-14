import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter


def simulate_cox_data_with_custom_betas(n=500, betas=None, lambda_0 = None, censoring_rate=0.3, seed=None):
    """
    Generate survival dataset from Cox model with custom coefficients.

    - n : number of individuals
    - betas : list or array of coefficients (of size p)
    - censoring_rate : proportion of censoring
    - seed : for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    if betas is None:
        raise ValueError("Provide coefficients 'betas'.")

    betas = np.array(betas)
    p = len(betas)

    # Covariates X ~ N(0,1)
    X = np.random.normal(size=(n, p))
    col_names = [f"X{i+1}" for i in range(p)]
    df = pd.DataFrame(X, columns=col_names)

    lin_pred = np.dot(X, betas)

    if lambda_0 is None:
        lambda_0 = 0.001

    # survival time generation
    U = np.random.uniform(size=n)
    survival_times = -np.log(U) / (lambda_0 * np.exp(lin_pred))

    # Censoring
    censoring_times = np.random.exponential(scale=survival_times.mean() / (1 - censoring_rate), size=n)
    observed_times = np.minimum(survival_times, censoring_times)
    event_observed = survival_times <= censoring_times

    df["time"] = observed_times
    df["event"] = event_observed

    return df


def simulate_cox_data(n_samples=1000, n_features=5, beta_scale=0.5, 
                      censoring_rate=0.5, baseline_hazard=0.01, random_state=None):
    rng = np.random.default_rng(random_state)

    # 1. Covariables
    X = rng.normal(size=(n_samples, n_features))

    # 2. Coefficients de Cox
    beta = rng.normal(loc=0, scale=beta_scale, size=n_features)

    # 3. Risque linéaire
    lin_pred = X @ beta

    # 4. Génération des temps de survie par inversion
    U = rng.uniform(size=n_samples)
    T = -np.log(U) / (baseline_hazard * np.exp(lin_pred))  # Formule basée sur S(t) = exp(-H0(t) * exp(Xβ))

    # 5. Génération des temps de censure
    C = rng.exponential(scale=T.mean() / censoring_rate, size=n_samples)

    # 6. Temps observé et indicateur de censure
    time = np.minimum(T, C)
    event = T <= C

    # Résultat en DataFrame
    df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(n_features)])
    df['time'] = time
    df['event'] = event.astype(int)

    return df, beta

def simulate_cox_data_with_fixed_betas(n=500, p=10, censoring_rate=0.3, seed=None):
    """
    Génère un dataset de survie avec p covariables et des coefficients fixés décroissants.
    - n : nombre d'individus
    - p : nombre de variables
    - censoring_rate : proportion censurée souhaitée
    - seed : pour la reproductibilité
    """

    if seed is not None:
        np.random.seed(seed)

    # Covariables X ~ N(0,1)
    X = np.random.normal(size=(n, p))
    col_names = [f"X{i+1}" for i in range(p)]
    df = pd.DataFrame(X, columns=col_names)

    # Coefficients décroissants (e.g., de 2 à 0.1)
    betas = np.linspace(2.0, 0.1, p)

    # Calcul du linéaire prédictif
    lin_pred = np.dot(X, betas)

    # Taux de base lambda
    lambda_0 = 0.01

    # Génération des temps de survie selon une loi exponentielle modifiée
    U = np.random.uniform(size=n)
    survival_times = -np.log(U) / (lambda_0 * np.exp(lin_pred))

    # Censure administrative aléatoire
    censoring_times = np.random.exponential(scale=survival_times.mean() / (1 - censoring_rate), size=n)
    observed_times = np.minimum(survival_times, censoring_times)
    event_observed = survival_times <= censoring_times

    df["time"] = observed_times
    df["event"] = event_observed

    return df, betas



def plot_individual_survival(df, n_individuals=5, time_horizon=None, random_state=None):

    # Préparation des données
    df_input = df.copy()
    duration_col = 'time'
    event_col = 'event'

    # Ajustement du modèle de Cox
    cph = CoxPHFitter()
    cph.fit(df_input, duration_col=duration_col, event_col=event_col)

    # Sélection d'individus à tracer
    rng = np.random.default_rng(random_state)
    indices = rng.choice(df_input.index, size=n_individuals, replace=False)
    individuals = df_input.loc[indices]

    # Courbes de survie individuelles
    surv_funcs = cph.predict_survival_function(individuals)

    # Tracé
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(surv_funcs.columns):
        plt.step(surv_funcs.index, surv_funcs[idx], where="post", label=f'Individu {indices[i]}')

    if time_horizon:
        plt.xlim(0, time_horizon)
    plt.ylim(0, 1)
    plt.xlabel("Temps")
    plt.ylabel("Probabilité de survie")
    plt.title("Fonctions de survie individuelles estimées")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
