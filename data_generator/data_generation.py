from time import time
import jax
import numpy as np
import pandas as pd
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from datetime import datetime, timedelta
import ensemble_kalman_filter as enkf
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from rebayes_mini.methods import robust_filter as rkf
from rebayes_mini import callbacks
import os
sin = jnp.sin
cos = jnp.cos

def rk4_step(y, i, dt, f):
    h = dt
    t = dt * i
    k1 = h * f(y, t)
    k2 = h * f(y + k1 / 2, dt * i + h / 2)
    k3 = h * f(y + k2 / 2, t + h / 2)
    k4 = h * f(y + k3, t + h)

    y_next = y + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next

    

@partial(jax.jit, static_argnames=("f",))
def rk4(ys, dt, N, f):
    """
    Based on
    https://colab.research.google.com
    github/google/jax/blob/master/cloud_tpu_colabs/Lorentz_ODE_Solver
    """
    @jax.jit
    def step(i, ys):
        ysi = rk4_step(ys[i - 1], i, dt, f)
        return ys.at[i].set(ysi)
    return jax.lax.fori_loop(1, N, step, ys)

def generate_and_save_data(M, N, dt, dim_state, dim_obs, mu0, cov0, key_init, key_obs, f, fC, h, h_step, save_dir, model_name, save_flag, corrupted=False, p_err=0.01, f_inacc=False):
    """
    Generate Monte Carlo simulation data and save to CSV files.

    Parameters:
    - M: Number of Monte Carlo samples
    - N: Number of time steps
    - dt: Time step size
    - dim_state: Dimension of the state variables
    - dim_obs: Dimension of the observation variables
    - mu0: Mean of the initial state distribution
    - cov0: Covariance of the initial state distribution
    - key_init: Initial random key for JAX
    - key_obs: Observation random key for JAX
    - f: Accurate state transition function
    - fC: Inaccurate state transition function (used if f_inacc is True)
    - h: Observation function
    - h_step: Function to compute observations
    - save_dir: Directory to save the CSV files
    - model_name: Base name for the CSV files
    - save_flag: Boolean flag to enable saving
    - corrupted: Boolean flag to generate corrupted observations
    - p_err: Probability of corruption in observations
    - f_inacc: Boolean flag to use inaccurate state transition

    Returns:
    - statev: Generated state data with shape (M, N, dim_state)
    - yv: Generated observation data with shape (M, N, dim_obs)
    - yv_corrupted: Corrupted observation data with shape (M, N, dim_obs) (if corrupted is True)
    """
    xs_mc = []
    ys_mc = []
    ys_corrupted_mc = []
    key_obs, key_mc = jax.random.split(key_obs)

    # Generate data for each Monte Carlo sample
    for i in range(M):
        key_mc, subkey = jax.random.split(key_mc)
        x0 = jax.random.multivariate_normal(key_init, mean=mu0, cov=cov0)
        xs1 = jnp.zeros((N,) + x0.shape)
        xs1 = xs1.at[0].set(x0)

        # Select the state transition function
        fpart = partial(fC, D=dim_state) if f_inacc else partial(f, D=dim_state)
        xs1 = rk4(xs1, dt, N, fpart)

        # Generate observations
        hpart = partial(h, D=dim_obs)
        ys1 = h_step(xs1, dt, N, hpart)

        # Optionally corrupt observations
        ys1_corrupted = ys1.copy()
        if corrupted:
            errs_map = jax.random.bernoulli(key_init, p=p_err, shape=ys1_corrupted.shape)
            ys1_corrupted = ys1_corrupted * (~errs_map) + 100.0 * errs_map

        # Append results
        xs_mc.append(xs1)
        ys_mc.append(ys1)
        ys_corrupted_mc.append(ys1_corrupted)

    # Convert lists to arrays
    statev = jnp.array(xs_mc)
    yv = jnp.array(ys_mc)
    yv_corrupted = jnp.array(ys_corrupted_mc) if corrupted else None

    # Flatten arrays for saving
    xs = statev.reshape(-1, dim_state)
    ys = yv.reshape(-1, dim_obs)
    ys_corrupted = yv_corrupted.reshape(-1, dim_obs) if corrupted else None

    # Generate a time series for the data
    start_time = datetime(2024, 7, 2, 13, 0, 0)
    time_interval = timedelta(seconds=1)
    time_series = [start_time + i * time_interval for i in range(xs.shape[0])]

    # Save data if required
    if save_flag:
        # Save state data
        column_names = [f'x_{i}' for i in range(1, xs.shape[1] + 1)]
        df_states = pd.DataFrame(xs, columns=column_names)
        df_states.insert(0, 'date', time_series)
        csv_path_states = save_dir + model_name + '_states.csv'
        df_states.to_csv(csv_path_states, index=False)

        # Save observation data
        column_names = [f'y_{i}' for i in range(1, ys.shape[1] + 1)]
        df_obs = pd.DataFrame(ys, columns=column_names)
        df_obs.insert(0, 'date', time_series)
        csv_path_obs = save_dir + model_name + '_obs.csv'
        df_obs.to_csv(csv_path_obs, index=False)

        # Save corrupted observation data (if applicable)
        if corrupted:
            column_names = [f'y_{i}' for i in range(1, ys_corrupted.shape[1] + 1)]
            df_obs_corrupted = pd.DataFrame(ys_corrupted, columns=column_names)
            df_obs_corrupted.insert(0, 'date', time_series)
            csv_path_obs_corrupted = save_dir + model_name + '_obs_corrupted.csv'
            df_obs_corrupted.to_csv(csv_path_obs_corrupted, index=False)
            
    return statev, yv, yv_corrupted



def load_data(save_dir, model_name, M, N, dim_state, dim_obs, corrupted):
    """
    Load saved state, observation, and corrupted observation data from CSV files.

    Parameters:
    - save_dir: Directory where the CSV files are saved
    - model_name: Model name used in the CSV filenames
    - M: Number of Monte Carlo samples
    - N: Number of time steps
    - dim_state: Dimension of the state variables
    - dim_obs: Dimension of the observation variables
    - corrupted: Whether corrupted observations were saved

    Returns:
    - statev: State variables with shape (M, N, dim_state)
    - yv: Observations with shape (M, N, dim_obs)
    - yv_corrupted: Corrupted observations with shape (M, N, dim_obs) (if `corrupted` is True)
    """
    # Load states (statev)
    csv_path_states = save_dir + model_name + '.csv'
    df_states = pd.read_csv(csv_path_states)
    xs = df_states.filter(like='x_').values  # Extract only columns starting with 'x_'
    statev = xs.reshape(M, N, dim_state)

    # Load observations (yv)
    csv_path_obs = save_dir + model_name + '_obs.csv'
    df_obs = pd.read_csv(csv_path_obs)
    ys = df_obs.filter(like='y_').values  # Extract only columns starting with 'y_'
    yv = ys.reshape(M, N, dim_obs)

    # Load corrupted observations (yv_corrupted) if applicable
    yv_corrupted = None
    if corrupted:
        csv_path_obs_corrupted = save_dir + model_name + '_obs_corrupted.csv'
        df_obs_corrupted = pd.read_csv(csv_path_obs_corrupted)
        ys_corrupted = df_obs_corrupted.filter(like='y_').values  # Extract only columns starting with 'y_'
        yv_corrupted = ys_corrupted.reshape(M, N, dim_obs)

    return statev, yv, yv_corrupted
