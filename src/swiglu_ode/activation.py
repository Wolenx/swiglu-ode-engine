import jax.numpy as jnp
import jax

def swiGLU(x, Wgate, Wup, Wdown):

    #for my implementation we take Wdown as its transposed version

    def SiLuscal(a):
        return a / (1 + jnp.exp(-a))

    Silu = jax.vmap(SiLuscal)
    L = jnp.dot(Wdown, Silu(jnp.dot(Wgate, x)) * jnp.dot(Wup, x)) 

    return L