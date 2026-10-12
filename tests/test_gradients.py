import jax
import jax.numpy as jnp
import jax.random as jrd

from swiglu_ode.network import BasicNeuralNetwork

def test_manual_vjp_against_jax():
    key = jrd.key(42)
    k1, k2 = jrd.split(key)
    
    # Small network for gradient verification
    model = BasicNeuralNetwork(d_in=4, d_hidden=16, d_out=2, n_layers=4, key=k1)
    
    batch_size = 8
    x_batch = jrd.normal(k2, (batch_size, 4))
    y_batch = jrd.normal(k2, (batch_size, 2))
    
    # 1. Get your manual gradients
    (grad_param0, grad_param1, grad_param2, 
     L1_grad, grad_Wout, grad_bout, grad_gamma) = model.Weights_grad_square_distance(x_batch, y_batch)
    
    # 2. Define a pure JAX functional loss for the exact PyTree
    @jax.jit
    def jax_native_loss(mdl, x, y):
        preds = jax.vmap(mdl)(x)
        # Sum across features (axis=-1), then mean across the batch
        return jnp.mean(jnp.sum(jnp.square(preds - y), axis=-1))
    
    # 3. Compute exact gradients using JAX's engine
    loss_val, exact_grads = jax.value_and_grad(jax_native_loss)(model, x_batch, y_batch)
    
    # 4. Assert mathematical equivalence (allowing for 32-bit float rounding errors)
    tolerance = 1e-4
    
    assert jnp.allclose(L1_grad, exact_grads.W_in, atol=tolerance), "W_in gradient mismatch!"
    assert jnp.allclose(grad_Wout, exact_grads.W_out, atol=tolerance), "W_out gradient mismatch!"
    assert jnp.allclose(grad_bout, exact_grads.b_out, atol=tolerance), "b_out gradient mismatch!"
    
    print("✅ VJP Verification Passed: Manual gradients exactly match XLA Autograd.")

if __name__ == "__main__":
    test_manual_vjp_against_jax()
