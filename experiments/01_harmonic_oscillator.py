import jax
import jax.numpy as jnp
import jax.random as jrd

from swiglu_ode.network import BasicNeuralNetwork

A=BasicNeuralNetwork(5,6,2,4)
print(A(jnp.array([1,2,3,4,5])))

def true_vector_field(state):
    """
    2D Damped Harmonic Oscillator
    dx/dt = y
    dy/dt = -x - 0.5y
    """
    x, y = state[..., 0], state[..., 1]
    dx = y
    dy = -x - 0.5 * y
    return jnp.stack([dx, dy], axis=-1)

def generate_ode_dataset(num_samples, key):
    # Sample random points in the phase space [-2, 2] x [-2, 2]
    X_raw = jrd.uniform(key, shape=(num_samples, 2), minval=-2.0, maxval=2.0)
    # The targets are the true derivatives at those points
    Y_raw = true_vector_field(X_raw)
    return X_raw, Y_raw

def train_and_integrate():
    key = jrd.key(2027)
    k1, k2, k3 = jrd.split(key, 3)
    
    # 1. Dataset Generation & Standardization
    N_train = 2048
    X_raw, Y_raw = generate_ode_dataset(N_train, k1)
    
    X_mean, X_std = jnp.mean(X_raw, axis=0), jnp.std(X_raw, axis=0)
    Y_mean, Y_std = jnp.mean(Y_raw, axis=0), jnp.std(Y_raw, axis=0)
    
    X_norm = (X_raw - X_mean) / X_std
    Y_norm = (Y_raw - Y_mean) / Y_std
    
    # 2. Initialize your custom architecture
    model = BasicNeuralNetwork(d_in=2, d_hidden=64, d_out=2, n_layers=4, key=k2)
    
    epochs = 1500
    learning_rate = 0.005
    
    print("🚀 Compiling XLA graph and starting training...")
    
    @jax.jit
    def compile_training_run(initial_model, X, Y, lr):
        def epoch_step(current_mdl, _):
            # 1. Compute gradients (reads state without mutating)
            grads = current_mdl.Weights_grad_square_distance(X, Y)
            grad_param0, grad_param1, grad_param2, L1_grad, grad_Wout, grad_bout, grad_gamma = grads
            Lact_grad = jnp.stack([grad_param1, grad_param2, grad_param0], axis=1)

            # 2. Extract PyTree leaves
            leaves, aux_data = current_mdl.tree_flatten()
            W_in, W_out, b_out, L_gamma, L_W3 = leaves

            # 3. Functional state update
            updated_leaves = (
                W_in - lr * L1_grad,
                W_out - lr * grad_Wout,
                b_out - lr * grad_bout,
                L_gamma - lr * grad_gamma,
                L_W3 - lr * Lact_grad
            )

            # 4. Repack the PyTree 
            new_mdl = type(current_mdl).tree_unflatten(aux_data, updated_leaves)
            
            # Calculate loss to track history
            preds = jax.vmap(new_mdl)(X)
            loss = jnp.mean(jnp.sum(jnp.square(preds - Y), axis=-1))
            
            return new_mdl, loss

        # Run all epochs in a single XLA call
        final_model, loss_history = jax.lax.scan(epoch_step, initial_model, None, length=epochs)
        return final_model, loss_history

    # Execute the single dispatch
    model, loss_history = compile_training_run(model, X_norm, Y_norm, learning_rate)

    print(f"✅ Training Complete. Final MSE: {loss_history[-1]:.6f}")
    # 3. Continuous-Time Rollout (Euler Integration)
    print("\n⏱️ Integrating learned vector field over time...")
    
    def learned_vector_field(state):
        # Normalize, forward pass, un-normalize
        state_norm = (state - X_mean) / X_std
        derivative_norm = model(state_norm)
        return (derivative_norm * Y_std) + Y_mean

    # Initial condition for the simulation
    initial_state = jnp.array([1.5, 0.0])
    dt = 0.05
    steps = 40
    
    # JAX native scan for fast rollout
    def euler_step(state, _):
        true_dx = true_vector_field(state)
        pred_dx = learned_vector_field(state)
        
        next_state_true = state + true_dx * dt
        next_state_pred = state + pred_dx * dt
        
        error = jnp.linalg.norm(next_state_pred - next_state_true)
        return next_state_pred, (state, pred_dx, error)
    
    _, (trajectory, _, errors) = jax.lax.scan(euler_step, init=initial_state, xs=jnp.arange(steps))
    
    # 4. Terminal-friendly output
    print(f"{'Step':<6} | {'X_pred':<8} | {'Y_pred':<8} | {'Drift Error':<10}")
    print("-" * 40)
    for i in range(0, steps, 5):
        print(f"{i:<6} | {trajectory[i, 0]:>6.3f}   | {trajectory[i, 1]:>6.3f}   | {errors[i]:.6f}")

if __name__ == "__main__":
    train_and_integrate()