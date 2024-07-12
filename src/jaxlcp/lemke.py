import jax.numpy as jnp
from jax import Array
from jax import jit, vmap
from jax.experimental.sparse import BCOO, BCSR, bcoo_concatenate, eye
from jax.experimental.sparse.linalg import spsolve
import jax

def lcp(M: Array, q: Array, algorithm:str = None, **kwargs)->tuple[Array, Array]:
    if algorithm is None:
        if isinstance(M, BCOO) or isinstance(M, BCSR):
            algorithm = 'fischer'
        else:
            algorithm = 'lemke'

    if algorithm == 'lemke':
        return _lcp_lemke(M,q)
    if algorithm == 'fischer':
        return _lcp_fischer(M,q,**kwargs)
    else:
        raise ValueError('Algorithm must be either "lemke" or "fischer"')


@jit
def T(M,q,w):
    return jnp.concatenate([M @ w[:,0]+ q-w[:,1], jnp.sqrt(w[:,0]**2+w[:,1]**2)-w[:,0]-w[:,1]])


def build_G(M, mu, nu):
    n = M.shape[0]
    G = bcoo_concatenate([bcoo_concatenate([M,eye(n)*-1], dimension=1),
                          bcoo_concatenate([eye(n)*mu,eye(n)*nu], dimension=1)], dimension=0)
    return BCSR.from_bcoo(G)

def _lcp_fischer(M, q,*, tol=1e-6, maxit=10, w0=None)->tuple[Array, Array]:
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
        print(jnp.max(jnp.abs(T(M,q,w))))
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



@jit
def _gauss_jordan_pivot(tableau: Array, k: int, j: int) -> Array:
    '''
    Pivot row k w.r.t column j 
    - Multiply the pivoting row k by 1/tableau[k,j]
    - Subtract the pivoting row k from all other rows i != k
    '''
    pivot_row = tableau[k, :] / tableau[k, j]

    tableau = tableau - tableau[:,[j]]*pivot_row
    tableau = tableau.at[k, :].set(pivot_row)
   
    return tableau

def _lcp_lemke(M:Array, q:Array, maxits=10000)->tuple[Array, Array]:
    '''
    Solves the linear complementarity problem (LCP) using the Lemke algorithm.
    w-Mz = q with w,z >= 0 and w'z = 0
    '''
    if len(q) != M.shape[0] or M.shape[0] != M.shape[1]:
        raise ValueError("Matrices are not compatible")

    n = len(q)
    it = 0

    if jnp.all(q>0):
        w, z = q, jnp.zeros_like(q)
        return w, z
   
    tableau = jnp.block([[jnp.eye(n), -M, -jnp.ones((n, 1)), q.reshape(-1,1)]])
    basis = jnp.arange(n)
    k = jnp.argmin(tableau[:,-1])
    tableau = _gauss_jordan_pivot(tableau, k=k, j= -2 ) #initialize tableau
    
    basis = basis.at[k].set(2*n)
    enter_var = k+n
    while max(basis) == 2*n and it < maxits:
        print(it)
        exit_var = jnp.argmin(jnp.where(tableau[:, enter_var]>0, tableau[:, -1]/tableau[:, enter_var], jnp.inf))
        tableau = _gauss_jordan_pivot(tableau, exit_var, enter_var)
        basis = basis.at[exit_var].set(enter_var)
        enter_var = exit_var+n
        it += 1

    vars = jnp.zeros(2*n)
    vars = vars.at[basis].set(tableau[:,-1])
    w, z = vars[:n], vars[n:]
    return w,z



