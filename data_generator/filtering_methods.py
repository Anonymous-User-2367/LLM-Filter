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
from rebayes_mini.methods.particle_filter import ParticleFilter
from rebayes_mini import callbacks
from pprint import pprint
np.set_printoptions(precision=4)

def rk4_step(y, i, dt, f):
    h = dt
    t = dt * i
    k1 = h * f(y, t)
    k2 = h * f(y + k1 / 2, dt * i + h / 2)
    k3 = h * f(y + k2 / 2, t + h / 2)
    k4 = h * f(y + k3, t + h)

    y_next = y + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next

def callback_fn(particles, particles_pred, y, i):
    return (jnp.power(particles.mean(axis=0) - xs[i], 2).mean()), particles.mean(axis=0)

def calculate_mse(errs):
    return jnp.mean(jnp.square(errs))
def calculate_mae(errs):
    return jnp.mean(jnp.abs(errs))

def calculate_rmse(errs):
    return jnp.sqrt(jnp.mean(jnp.square(errs)))

def calculate_test_error(errs):
    mse = calculate_mse(errs)
    mae = calculate_mae(errs)
    rmse = calculate_rmse(errs) 
    # print(f'mse:{mse}, mae:{mae}, rmse:{rmse}')
    return mse, mae, rmse

# @jax.jit(static_argnames=('ff', 'hh'))
def filter_ekf(ff, hh, x0, measurements, state, key_eval):
    agent_imq = rkf.ExtendedKalmanFilterIMQ(
        ff, hh,
        dynamics_covariance=jnp.eye(state.shape[-1]),
        observation_covariance=jnp.eye(measurements.shape[-1]),
        soft_threshold=1e8,
    )
    init_bel = agent_imq.init_bel(x0, cov=1.0)
    filterfn = partial(agent_imq.scan, callback_fn=callbacks.get_updated_mean)
    
    _, hist = filterfn(init_bel, measurements, jnp.ones(state.shape[0]))
    err = hist - state
    return err, hist


def filter_wlfimq(ff, hh, x0, threshold, measurements, state, key_eval):
    agent_imq = rkf.ExtendedKalmanFilterIMQ(
        ff, hh,
        dynamics_covariance=jnp.eye(state.shape[-1]),
        observation_covariance=jnp.eye(measurements.shape[-1]),
        soft_threshold=threshold
    )
    
    init_bel = agent_imq.init_bel(x0, cov=1e-8)
    
    _, hist = agent_imq.scan(
        init_bel, measurements, jnp.ones(state.shape[0]), callback_fn=callbacks.get_updated_mean
    )

    # err = jnp.sqrt(jnp.power(hist - state, 2).sum(axis=0))
    err = hist - state
    return err, hist

def filter_wlfsq(ff, hh, x0, threshold, measurements, state, key_eval):
    agent_sq = rkf.ExtendedKalmanFilterSQ(
        ff, hh,
        dynamics_covariance=jnp.eye(state.shape[-1]),
        observation_covariance=jnp.eye(measurements.shape[-1]),
        soft_threshold=threshold
    )
    
    init_bel = agent_sq.init_bel(x0, cov=1e-8)
    
    _, hist = agent_sq.scan(
        init_bel, measurements, jnp.ones(state.shape[0]), callback_fn=callbacks.get_updated_mean
    )

    # err = jnp.sqrt(jnp.power(hist - state, 2).sum(axis=0))
    err = hist - state
    return err, hist

def filter_wlfmd(ff, hh, x0, threshold, measurements, state, key_eval):
    agent= rkf.ExtendedKalmanFilterMD(
        ff, hh,
        dynamics_covariance=jnp.eye(state.shape[-1]),
        observation_covariance=jnp.eye(measurements.shape[-1]),
        threshold=threshold
    )
    
    init_bel = agent.init_bel(x0, cov=1e-8)
    
    _, hist = agent.scan(
        init_bel, measurements, jnp.ones(state.shape[0]), callback_fn=callbacks.get_updated_mean
    )

    # err = jnp.sqrt(jnp.power(hist - state, 2).sum(axis=0))
    err = hist - state
    return err, hist


def filter_enkf(ff, hh, x0, n_particles, measurements, state, key_eval, 
                dt = 0.01):
    def fn(x, key, i):
        """
        State function
        """
        err = jax.random.normal(key, (state.shape[-1],))
        @jax.jit
        def f(x, t):
            return ff(x)  + err
        return rk4_step(x, i, dt, f)
    def hn(x, key, i):
        """
        Measurement function
        """
        err = jax.random.normal(key, (measurements.shape[-1],))
        return hh(x, dt) + err
    agent = enkf.EnsembleKalmanFilter(fn, hn, n_particles)
    key_init_particles, key_scan = jax.random.split(key_eval, 2)
    X0 = agent.init_bel(key_init_particles, state.shape[-1])
    particles_end, (particles_hist) = agent.scan(X0, key_scan, measurements, callback_fn=callbacks.get_updated_bel)
    hist = particles_hist.mean(axis=1)
    errs = hist - state
    return errs, hist

def filter_enkfi(ff, hh, x0, inflation_factor, n_particles, 
                 measurements, state, key_eval, 
                 dt = 0.01):
    def fn(x, key, i):
        """
        State function
        """
        err = jax.random.normal(key, (state.shape[-1],))
        @jax.jit
        def f(x, t):
            return ff(x)  + err
        return rk4_step(x, i, dt, f)
    def hn(x, key, i):
        """
        Measurement function
        """
        err = jax.random.normal(key, (measurements.shape[-1],))
        return hh(x, dt) + err
    agent = enkf.EnsembleKalmanFilterInflation(fn, hn, n_particles, inflation_factor=inflation_factor)
    key_init_particles, key_scan = jax.random.split(key_eval, 2)
    X0 = agent.init_bel(key_init_particles, state.shape[-1])
    particles_end, (particles_hist) = agent.scan(X0, key_scan, measurements, callback_fn=callbacks.get_updated_bel)
    hist = particles_hist.mean(axis=1)
    errs = hist - state
    return errs, hist


def filter_hubenkf(ff, hh, x0, inflation_factor, n_particles, 
                 measurements, state, key_eval, 
                 dt = 0.01):
    def fn(x, key, i):
        """
        State function
        """
        err = jax.random.normal(key, (state.shape[-1],))
        @jax.jit
        def f(x, t):
            return ff(x)  + err
        return rk4_step(x, i, dt, f)
    def hn(x, key, i):
        """
        Measurement function
        """
        err = jax.random.normal(key, (measurements.shape[-1],))
        return hh(x, dt) + err
    agent = enkf.HubEnsembleKalmanFilter(fn, hn, n_particles, inflation_factor=inflation_factor)
    key_init_particles, key_scan = jax.random.split(key_eval, 2)
    X0 = agent.init_bel(key_init_particles, state.shape[-1])
    particles_end, (particles_hist) = agent.scan(X0, key_scan, measurements, callback_fn=callbacks.get_updated_bel)
    hist = particles_hist.mean(axis=1)
    errs = hist - state
    return errs, hist


def filter_enkfs(ff, hh, x0, parameter, n_particles, 
                 measurements, state, key_eval, 
                 dt = 0.01):
    def fn(x, key, i):
        """
        State function
        """
        err = jax.random.normal(key, (state.shape[-1],))
        @jax.jit
        def f(x, t):
            return ff(x)  + err
        return rk4_step(x, i, dt, f)
    def hn(x, key, i):
        """
        Measurement function
        """
        err = jax.random.normal(key, (measurements.shape[-1],))
        return hh(x, dt) + err
    agent = enkf.WLEnsembleKalmanFilterSoft(fn, hn, n_particles, c=parameter)
    key_init_particles, key_scan = jax.random.split(key_eval, 2)
    X0 = agent.init_bel(key_init_particles, state.shape[-1])
    particles_end, (particles_hist) = agent.scan(X0, key_scan, measurements, callback_fn=callbacks.get_updated_bel)
    hist = particles_hist.mean(axis=1)
    errs = hist - state
    return errs, hist

def filter_pf(ff, hh, x0, num_particle,measurements, state, key_eval):
    agent_particle=ParticleFilter(ff, hh,
        dynamics_covariance=jnp.eye(state.shape[-1]),
        observation_covariance=jnp.eye(measurements.shape[-1]),
        n_particles=num_particle
        )
    key_init_particles, key_scan = jax.random.split(key_eval, 2)
    init_particles=agent_particle._initialize_particles(key_init_particles, x0, cov=1.0)
    filterfn = partial(agent_particle.scan, callback_fn=callbacks.get_updated_mean)
    _, hist = filterfn(init_particles, key_scan, measurements, jnp.ones(state.shape[0]))
    err = hist - state
    return err, hist

def run_filtering(methods=['EKF'], yv=None, statev=None, key_eval=None, ff=None, hh=None, parameter_range=(0.2, 10),num_particle=1000):
    time_methods = {}
    errs_methods = {}
    configs_methods = {}
    for method in methods:
        print(method)
        if method not in ('EKF', 'EnKF', 'PF'):
            def bo_filter(parameter):
                if method == 'WLFIMQ':
                    err, _ = filter_wlfimq(ff, hh, statev[0][0], parameter,  yv[0], statev[0], key_eval)
                elif method == 'WLFSQ':
                    err, _ = filter_wlfsq(ff, hh, statev[0][0], parameter,  yv[0], statev[0], key_eval)
                elif method == 'WLFMD':
                    err, _ = filter_wlfmd(ff, hh, statev[0][0], parameter,  yv[0], statev[0], key_eval)
                elif method == 'EnKFI':
                    err, _ = filter_enkfi(ff, hh, statev[0][0], parameter, num_particle, yv[0], statev[0], key_eval)
                elif method == 'HubEnKF':
                    err, _ = filter_hubenkf(ff, hh, statev[0][0], parameter, num_particle, yv[0], statev[0], key_eval)
                elif method == 'EnKFS':
                    err, _ = filter_enkfs(ff, hh, statev[0][0], parameter, num_particle, yv[0], statev[0], key_eval)
                else:
                    raise ValueError('Method not found')
                err = err.max()
                err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
                return -calculate_rmse(err)
            
            bo = BayesianOptimization(
            bo_filter,
            pbounds={
                "parameter": parameter_range
            },
            random_state=314,
            verbose=0)
            bo.maximize(init_points=20, n_iter=20)
            parameter = bo.max["params"]["parameter"]
            # try:
            #     # 尝试运行贝叶斯优化
            #     bo.maximize(init_points=10, n_iter=10)

            #     # 检查优化结果中是否有 NaN
            #     if "params" in bo.max and not np.isnan(bo.max["params"]["parameter"]):
            #         parameter = bo.max["params"]["parameter"]
            #     else:
            #         raise ValueError("Optimization result contains NaN")  # 手动触发异常
            # except (KeyError, ValueError, TypeError):
            #     # 如果有 NaN 或获取失败，随机选择一个参数
            #     print("NaN or exception encountered in Bayesian optimization, selecting a random parameter.")
            #     parameter = np.random.choice(parameter_range)

            # 保存到 configs_methods
            configs_methods[method] = parameter
            
        hist_bel = []
        times = []
        errs = 0
        tinit = time()
        for y, state in tqdm(zip(yv, statev), total=statev.shape[0]): 
            key_eval, key_scan = jax.random.split(key_eval)
            if method == 'EKF':
                errs, _ = filter_ekf(ff, hh, state[0], y, state, key_eval)
            elif method == 'EnKF':
                errs, _ = filter_enkf(ff, hh, state[0], num_particle, y, state, key_eval)
            elif method == 'WLFIMQ':
                errs, _ = filter_wlfimq(ff, hh, state[0], parameter,  y, state, key_eval)
            elif method == 'WLFSQ':
                errs, _ = filter_wlfsq(ff, hh, state[0], parameter,  y, state, key_eval)
            elif method == 'WLFMD':
                errs, _ = filter_wlfmd(ff, hh, state[0], parameter,  y, state, key_eval)
            elif method == 'EnKFI':
                errs, _ = filter_enkfi(ff, hh, state[0], parameter, num_particle, y, state, key_eval)
            elif method == 'HubEnKF':
                errs, _ = filter_hubenkf(ff, hh, state[0], parameter, num_particle, y, state, key_eval)
            elif method == 'EnKFS':
                errs, _ = filter_enkfs(ff, hh, state[0], parameter, num_particle, y, state, key_eval)
            elif method=='PF':
                errs, _ = filter_pf(ff, hh, state[0], num_particle, y, state, key_eval)
            else:
                raise ValueError('Method not found')
            
            
            hist_bel.append(errs)
            
        tend = time()
        times = tend - tinit
        tim = times / statev.shape[0] / statev.shape[1]
        errs = np.stack(hist_bel)
        time_methods[method] = np.sum(tim) * 1000

        _, _, errs_methods[method] = calculate_test_error(errs)
    print('Done')
    print('RMSE')
    pprint(errs_methods)
    print('Time')
    pprint(time_methods)
    return time_methods, errs_methods, configs_methods
