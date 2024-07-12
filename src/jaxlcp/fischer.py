import jax.numpy as jnp
from jax import Array
from jax import jit, vmap
from jax.experimental.sparse import BCOO, BCSR, bcoo_concatenate, eye
from jax.experimental.sparse.linalg import spsolve
import jax


@jit
def T(M,q,w):
    return jnp.concatenate([M @ w[:,0]+ q-w[:,1], jnp.sqrt(w[:,0]**2+w[:,1]**2)-w[:,0]-w[:,1]])


def build_G(M, mu, nu):
    n = M.shape[0]
    G = bcoo_concatenate([bcoo_concatenate([M,eye(n)*-1], dimension=1),
                          bcoo_concatenate([eye(n)*mu,eye(n)*nu], dimension=1)], dimension=0)
    return BCSR.from_bcoo(G)

def lcp_fischer(M, q,*, tol=1e-6, maxit=10, w0=None)->tuple[Array, Array]:
    '''
    Solve the problem Mx+q=y y'x=0, y>=0, x>=0
    w=[x,y]
    Fischer (1995) Newton method
    '''
    jax.config.update("jax_enable_x64", True)
    n = M.shape[0]
    M = M.astype(jnp.float64)
    w = jnp.zeros((n,2)) if w0 is None else w0
    k = 0
    lamdba = 1e-5
    while jnp.max(jnp.abs(T(M,q,w)))>tol and k<maxit:
        mu = jnp.where(jnp.all(w!=0, axis=1), -1+w[:,0]/jnp.sqrt(w[:,0]**2+w[:,1]**2), -1)
        nu = jnp.where(jnp.all(w!=0, axis=1), -1+w[:,1]/jnp.sqrt(w[:,0]**2+w[:,1]**2), 0)
        G = build_G(M, mu, nu)
        w_hat = spsolve(G.data, G.indices, G.indptr, G@w.flatten('F')-T(M,q, w))
        w_hat = w_hat.reshape(n,2,order='F')

        for j in range(100):
            t_k = 0.5**j
            w_next = w+t_k*(w_hat-w)
            if jnp.linalg.norm(T(M,q, w_next))  <= (1-lamdba*t_k)*jnp.linalg.norm(T(M,q,w)):
                w = w_next
                k+=1
                break
    if k == maxit:
        raise ValueError("Max iterations reached")
    else:
        return w[:,0],w[:,1]


