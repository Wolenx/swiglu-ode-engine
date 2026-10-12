import jax.numpy as jnp

def RMSNorme(x, epsilon = 1e-7):
    n = x.shape[-1]
    return x/(jnp.sqrt(jnp.sum(jnp.square(x))/n) + epsilon)

def dRMSNorme(h, x, epsilon = 1e-7):
    n=x.shape[-1]
    r = (jnp.sqrt(jnp.sum(jnp.square(x))/n))
    denominator = n * r * (r + epsilon)**2
    return h/(r+epsilon) - x*(jnp.dot(x,h)/denominator)