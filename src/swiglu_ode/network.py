import jax
import jax.random as jrd
import jax.numpy as jnp

from .activation import swiGLU
from .normalization import RMSNorme, dRMSNorme


@jax.tree_util.register_pytree_node_class
class BasicNeuralNetwork:
    def __init__(self, d_in, d_hidden, d_out, n_layers, key=jrd.key(2026)):
        kin, klayers, kwout ,kbout= jrd.split(key, 4)
        self.d_out = d_out
        self.n_layers = n_layers
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.W_in = jrd.normal(kin, shape=(d_hidden, d_in)) # last activation matrice of the first layer
        self.W_in = self.W_in / jnp.sqrt(d_in)
        self.L_W3 = jrd.normal(klayers, shape=(n_layers-2, 3, int(jnp.floor(8/3*d_hidden)), d_hidden)) # for each layer we need 3 matrices for the activation but not of same shapes we transpose the last one
        L=jnp.reshape(jnp.array([jnp.sqrt(d_hidden), jnp.sqrt(d_hidden), jnp.sqrt(int(jnp.floor(8/3*d_hidden)))]), shape=(3,1,1))
        self.L_W3 = self.L_W3 / L
        self.L_gamma = jnp.ones(shape=(n_layers-2, d_hidden))
        self.W_out = jrd.normal(kwout, shape=(d_out, d_hidden))
        self.W_out = self.W_out / jnp.sqrt(d_hidden)
        self.b_out = jrd.normal(kbout, shape=(d_out, ))

        # initial scaling will be implemented later 

    def tree_flatten(self):
        children = (self.W_in, self.W_out, self.b_out, self.L_gamma, self.L_W3)
        aux_data = (self.d_in, self.d_out, self.n_layers, self.d_hidden)  # static
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):

        d_in, d_out, n_layers, d_hidden = aux_data
        
        #Allocate an empty instance without calling __init__ to avoid giving junk weights
        obj = cls.__new__(cls)
        obj.d_in = d_in
        obj.d_out = d_out
        obj.d_hidden = d_hidden
        obj.n_layers = n_layers
        obj.W_in, obj.W_out, obj.b_out, obj.L_gamma, obj.L_W3 = children
            
        return obj

    def size(self):
        return ( self.d_out*self.d_hidden + self.d_hidden*self.d_in + self.d_out + (self.n_layers-2)*self.d_hidden*int(jnp.floor(8/3*self.d_hidden))*3 + (self.n_layers-2)*self.d_hidden ) * 4 / (1024)**2

    
    @jax.jit
    def __call__(self, x):
        """x need to be a jax array"""

        X_arr = jnp.dot(self.W_in, x)

        def forwardlin(X, Wgate, Wup, Wdown, Gamma):
            return X + swiGLU(RMSNorme(X)*Gamma, Wgate, Wup, Wdown)

        def st_func(X, param):

            W3, Gamma = param
            Wgate , Wup, TRSWdown = W3
            
            Xnew=forwardlin(X, Wgate, Wup, jnp.transpose(TRSWdown), Gamma)
            return Xnew, None
        
        X_final, _ = jax.lax.scan(st_func, init=X_arr, xs=(self.L_W3, self.L_gamma))

        X_final = jnp.dot(self.W_out, X_final) + self.b_out
        
        return X_final

    @jax.jit
    def forward_training(self, x):
        """make sur to send x as a jnp.array"""

        X_arr = jnp.dot(self.W_in, x)

        def forwardlin(X, Wgate, Wup, Wdown, Gamma):
            return X + swiGLU(RMSNorme(X)*Gamma, Wgate, Wup, Wdown)

        def st_func(X, param):

            W3, Gamma = param
            Wgate , Wup, TRSWdown = W3
            
            Xnew=forwardlin(X, Wgate, Wup, jnp.transpose(TRSWdown), Gamma)
            return Xnew, Xnew
        
        X_final, X_history = jax.lax.scan(st_func, init=X_arr, xs=(self.L_W3, self.L_gamma))
        X_history = jnp.concat([jnp.expand_dims(X_arr, axis=0),X_history])

        X_final = jnp.dot(self.W_out, X_final) + self.b_out

        return X_history, X_final

    @jax.jit
    def VJPlayer(self, v, x, param):
        """return the VJP (Backward Mode) associated with v for one layer"""
        #W3 = Wdown, W2 = Wgate, W1 = Wup

        def sigmoidscal(a):
            return 1/(1+jnp.exp(-a))
        
        def SiLuscal(a):
            return a/(1 + jnp.exp(-a))
        
        Silu = jax.vmap(SiLuscal)
        Sigmoid = jax.vmap(sigmoidscal)

        unormedx = x
        dRMSNormex=lambda h : dRMSNorme(h, unormedx)

        W3, Gamma = param
        Wgate, Wup, trsWdown = W3
        
        rx=RMSNorme(x)
        x=rx*Gamma

        Wgatex=jnp.dot(Wgate,x)
        Wupx=jnp.dot(Wup,x)
        WdownTv=jnp.dot(trsWdown,v)

        L=Sigmoid(Wgatex)
        temp2=(L+Silu(Wgatex)*(1-L))*WdownTv
        temp=Wupx*temp2

        y=v+dRMSNormex(jnp.dot(jnp.transpose(Wgate), temp)*Gamma)+dRMSNormex(jnp.dot(jnp.transpose(Wup),Silu(Wgatex)*WdownTv)*Gamma)
        dWdown=v[:, None] @ (Silu(Wgatex)*Wupx)[None, :]
        dWup=(Silu(Wgatex)*WdownTv)[:, None] @ x[None, :]
        dWgate=temp[:, None] @ x[None, :]
        dGamma=rx*jnp.dot(jnp.transpose(Wup), Silu(Wgatex)*WdownTv)+rx*jnp.dot(jnp.transpose(Wgate), temp)

        return (jnp.transpose(dWdown),dWgate,dWup,y,dGamma)

    @jax.jit
    def VJPInput(self, v, x, param):
        return v[: , None] @ x[None , :]

    def VJPOut(self, v, x, param):
        return (jnp.dot(jnp.transpose(param),v), v[: , None] @ x[None , :], v)
    
    @jax.jit
    def fullforwardbackwardpass(self, x, X_history, err_grad):
        """x needs to be a jnp.array """
        
        vjpout = self.VJPOut(err_grad, X_history[-1], self.W_out)

        grad_y = vjpout[0]

        def step_vjp(y, paramsinput):
            X_his, W3, Gamma = paramsinput
            dWdown,dWgate,dWup,y,dGamma = self.VJPlayer(y, X_his, (W3, Gamma))

            return y, (dWdown,dWgate,dWup,dGamma)

        Final_y, grad_history = jax.lax.scan(step_vjp, init=grad_y, xs=(X_history[:-1], self.L_W3, self.L_gamma), reverse=True)

        grad_In = self.VJPInput(Final_y, x, self.W_in)

        grad_Out = vjpout[1:]

        return grad_history, grad_In, grad_Out

    @jax.jit
    def Weights_grad_square_distance(self, x_batch, label_batch):

        """This function will calculate the weights gradient. Avoid mixing batch sizes to avoid XLA compilation bloating"""

        gradfunc=jax.vmap(self.fullforwardbackwardpass, in_axes=(0,0,0))
        
        batchforwardtrain=jax.vmap(self.forward_training)

        X_history_batch, X_pred_batch=batchforwardtrain(x_batch)

        batch_prediction=X_pred_batch
        batch_err_grad=2*(batch_prediction-label_batch)

        batch_param_grad, batch_L1_grad, batch_Lout_grad = gradfunc(x_batch, X_history_batch, batch_err_grad)

        batch_param_grad0, batch_param_grad1, batch_param_grad2, batch_gamma_grad = batch_param_grad
        batch_Lout_grad_Wout, batch_Lout_grad_bout = batch_Lout_grad

        grad_param0, grad_param1, grad_param2, grad_gamma = jnp.mean(batch_param_grad0, axis=0), jnp.mean(batch_param_grad1, axis=0), jnp.mean(batch_param_grad2, axis=0), jnp.mean(batch_gamma_grad, axis=0)
        L1_grad = jnp.mean(batch_L1_grad, axis=0)
        grad_Wout = jnp.mean(batch_Lout_grad_Wout, axis=0)
        grad_bout = jnp.mean(batch_Lout_grad_bout, axis=0)


        return grad_param0, grad_param1, grad_param2, L1_grad, grad_Wout, grad_bout, grad_gamma

    def train_square_distance(self, x_batch, label_batch, learning_rate=0.1):

        grad_param0, grad_param1, grad_param2, L1_grad, grad_Wout, grad_bout, grad_gamma = self.Weights_grad_square_distance(x_batch, label_batch)

        Lact_grad = jnp.stack([grad_param1, grad_param2, grad_param0], axis = 1) # axis 1 since 0 corresponds to the layer number

        #print(Lact_grad, L1_grad)
        #print(self.Lact_param_in)

        self.L_W3 = self.L_W3 - learning_rate*Lact_grad
        self.W_in = self.W_in - learning_rate*L1_grad
        self.W_out = self.W_out - learning_rate*grad_Wout
        self.b_out = self.b_out - learning_rate*grad_bout
        self.L_gamma = self.L_gamma - learning_rate*grad_gamma
