import jax
import distrax
import jax.numpy as jnp
from functools import partial
from rebayes_mini import callbacks
from rebayes_mini.states import GaussState
from jax.flatten_util import ravel_pytree
from jax.scipy.stats import multivariate_normal
import jax
import jax.numpy as jnp
from jax import lax




class ParticleFilter:
    def __init__(
        self, fn_latent, fn_obs, dynamics_covariance, observation_covariance,n_particles=1000
    ):
        self.fn_latent = fn_latent
        self.fn_obs = fn_obs
        self.dynamics_covariance = dynamics_covariance
        self.observation_covariance = observation_covariance
        self.n_particles=1000
    def _initalise_vector_fns(self, latent):
        vlatent, rfn = ravel_pytree(latent)
        @jax.jit # ht(z)
        def vobs_fn(latent, x):
            latent = rfn(latent)
            return self.fn_obs(latent, x)

        @jax.jit # ft(z, u)
        def vlatent_fn(latent):
            return self.fn_latent(latent)

        return rfn, vlatent_fn, vobs_fn, vlatent
    def _initialize_particles(self, key, mean, cov):
        self.rfn, self.vlatent_fn, self.vobs_fn, vlatent = self._initalise_vector_fns(mean)
        self.dim_latent=len(vlatent)
        self.weights = jnp.ones(self.n_particles) / self.n_particles
        particles = jax.random.multivariate_normal(key, mean=mean, cov=cov*jnp.eye(self.dim_latent), shape=(self.n_particles,))
        return particles
    
    def _predict(self, particles, key):
        latent_pred = jax.vmap(self.vlatent_fn, in_axes=(0))(particles)
        return latent_pred+jax.random.multivariate_normal(key, jnp.zeros(self.dim_latent), self.dynamics_covariance, shape=(self.n_particles,))
    
    def _update(self, particles, y, x, key):
        def update_single(particle, x, current_weight):
            y_pred = self.vobs_fn(particle, x)
            likelihood = multivariate_normal.pdf(y, y_pred, self.observation_covariance)
            updated_weight = current_weight * likelihood
            return updated_weight
        self.weights = jax.vmap(update_single, in_axes=(0, None, 0))(particles, x, self.weights)
        self.weights += 1.e-300
        self.weights /= jnp.sum(self.weights) 
        effective_particles = self.neff(self.weights)
        particles = jax.lax.cond(
            effective_particles < self.n_particles ,
            lambda _: self.resample(particles, self.weights, key),
            lambda _: particles,
            operand=None
        )
        self.weights = jnp.ones(self.n_particles) / self.n_particles
        return particles

    def neff(self, weights):
        return 1. / jnp.sum(jnp.square(weights))

    def resample(self, particles, weights, key):
        cumulative_weights = jnp.cumsum(weights)
        random_values = jax.random.uniform(key, shape=(self.n_particles,))
        indices = jnp.searchsorted(cumulative_weights, random_values)
        resampled_particles = particles[indices]
        return resampled_particles

    def step(self, particles, xs, callback_fn, key):
        yt, xt, t = xs
        key = jax.random.fold_in(key, t)
        key_pred, key_update = jax.random.split(key)
        particles_pred = self._predict(particles, key_pred)
        particles_update = self._update(particles_pred, yt, xt, key_update)
        output = jnp.sum(particles_update * self.weights[:, None], axis=0) 
        return particles_update, output 

    def scan(self, particles, key_scan, y, X, callback_fn=None):
        n_steps = len(y)
        timesteps = jnp.arange(n_steps)
        xs = (y, X, timesteps)
        callback_fn = callbacks.get_null if callback_fn is None else callback_fn
        _step = partial(self.step, callback_fn=callback_fn, key=key_scan)
        particles, hist = jax.lax.scan(_step, particles, xs)
        return particles, hist
