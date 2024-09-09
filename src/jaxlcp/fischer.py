import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO, BCSR, bcoo_concatenate
from jax.experimental import sparse
from jax.experimental.sparse.linalg import spsolve
import jax
import warnings
jax.config.update("jax_enable_x64", False)

@jax.jit
def T(M:Array,q:Array,w:Array)->Array:
    return jnp.concatenate([M @ w[:,0]+ q-w[:,1], jnp.sqrt(w[:,0]**2+w[:,1]**2)-w[:,0]-w[:,1]])


@jax.jit
def build_G(M:Array, mu:Array, nu:Array)->Array:
    n = M.shape[0]
    data = jnp.concatenate([M.data, -jnp.ones(n), mu, nu])
    indices = jnp.vstack([M.indices, 
                             jnp.column_stack([jnp.arange(n), n+jnp.arange(n)]),
                             jnp.column_stack([n+jnp.arange(n), jnp.arange(n)]),
                             jnp.column_stack([n+jnp.arange(n), n+jnp.arange(n)])])
    G =  BCOO((data, indices), shape=(2*n, 2*n))
    return BCSR.from_bcoo(G)


def lcp_fischer(M, q, *, tol=1e-6, maxit=100, w0=None, spsolve_tol:float=1e-6):
    '''
    Solve the problem Mx+q=y y'x=0, y>=0, x>=0
    w=[x,y]
    Fischer (1995) Newton method
    '''
    n = M.shape[0]
    w_init = jnp.column_stack([jnp.ones(n),jnp.zeros(n)]) if w0 is None else w0
    lambd = 1e-5
    def cond_fun(val):
        '''
        Stop when T(M,q,w) is small enough or maxit is reached
        '''
        w, k = val
        return jnp.logical_and(jnp.max(jnp.abs(T(M, q, w))) > tol, k < maxit)

    def body_fun(val):
        w, k = val
        tau = 1/2*jnp.minimum(jnp.maximum(w[:,0],0), jnp.linalg.norm(T(M,q,w)))
        w = w.at[:,1].set(jnp.where(3*jnp.abs(w[:,1])<=tau,
                                    w[:,1]+jnp.sign(M@w[:,0]-w[:,1]+q)*tau,
                                    w[:,1]))
        mu = jnp.where(jnp.all(w != 0, axis=1), -1 + w[:, 0] / jnp.sqrt(w[:, 0]**2 + w[:, 1]**2), -1)
        nu = jnp.where(jnp.all(w != 0, axis=1), -1 + w[:, 1] / jnp.sqrt(w[:, 0]**2 + w[:, 1]**2), 0)
        G = build_G(M, mu, nu)
        w_hat = spsolve(G.data, G.indices, G.indptr,
                        G @ w.flatten('F') - T(M, q, w),
                        tol = spsolve_tol)
        w_hat = w_hat.reshape(n, 2, order='F')

        def armijo_cond(t_k):
            w_next = w + t_k * (w_hat - w)
            return jnp.linalg.norm(T(M, q, w_next)) > (1 - lambd * t_k) * jnp.linalg.norm(T(M, q, w))


        t_k = jax.lax.while_loop(armijo_cond,
                                lambda t_k: t_k*1/2,
                                1/2)

        w_next = w + t_k * (w_hat - w)
        return w_next, k + 1

    w, k = jax.lax.while_loop(cond_fun, body_fun, (w_init, 0))


    if k == maxit:
        warnings.warn(f"Max iterations={maxit} reached. Error is {jnp.max(jnp.abs(T(M, q, w)))}")
    
    return w[:, 0], w[:, 1]




# def lcp_fischer(M, q,*, tol=1e-6, maxit=10, w0:Array=None, spsolve_tol:float=1e-6)->tuple[Array, Array]:
#     '''
#     Solve the problem Mx+q=y y'x=0, y>=0, x>=0
#     w=[x,y]
#     Fischer (1995) Newton method
#     '''
#     n = M.shape[0]
#     M = M.astype(jnp.float64)
#     w = jnp.zeros((n,2)) if w0 is None else w0
#     k = 0
#     lamdba = 1e-5
#     while jnp.max(jnp.abs(T(M,q,w)))>tol and k<maxit:
#         tau = 1/2*jnp.minimum(jnp.max(w[0],0), jnp.linalg.norm(T(M,q,w)))
#         mu = jnp.where(jnp.all(w!=0, axis=1), -1+w[:,0]/jnp.sqrt(w[:,0]**2+w[:,1]**2), -1)
#         nu = jnp.where(jnp.all(w!=0, axis=1), -1+w[:,1]/jnp.sqrt(w[:,0]**2+w[:,1]**2), 0)
#         G = build_G(M, mu, nu)
#         w_hat = spsolve(G.data, G.indices, G.indptr, G@w.flatten('F')-T(M,q, w), tol=spsolve_tol)
#         w_hat = w_hat.reshape(n,2,order='F')

#         for j in range(100):
#             t_k = 0.5**j
#             w_next = w+t_k*(w_hat-w)
#             if jnp.linalg.norm(T(M,q, w_next))  <= (1-lamdba*t_k)*jnp.linalg.norm(T(M,q,w)):
#                 w = w_next
#                 k+=1
#                 break
#     if k == maxit:
#         raise ValueError("Max iterations reached")
#     else:
#         return w[:,0],w[:,1]



def lcp_fischer(M, q, *, tol=1e-6, maxit=10, w0=None, spsolve_tol:float=1e-6):
    n = M.shape[0]
    w_init = jnp.column_stack([jnp.ones(n),jnp.zeros(n)]) if w0 is None else w0
    lambd = 1e-5
    def cond_fun(val):
        '''
        Stop when T(M,q,w) is small enough or maxit is reached
        '''
        w, k = val
        return jnp.logical_and(jnp.max(jnp.abs(T(M, q, w))) > tol, k < maxit)

    def body_fun(val):
        w, k = val
        tau = 1/2*jnp.minimum(jnp.maximum(w[:,0],0), jnp.linalg.norm(T(M,q,w)))
        w = w.at[:,1].set(jnp.where(3*jnp.abs(w[:,1])<=tau,
                                    w[:,1]+jnp.sign(M@w[:,0]-w[:,1]+q)*tau,
                                    w[:,1]))
        mu = jnp.where(jnp.all(w != 0, axis=1), -1 + w[:, 0] / jnp.sqrt(w[:, 0]**2 + w[:, 1]**2), -1)
        nu = jnp.where(jnp.all(w != 0, axis=1), -1 + w[:, 1] / jnp.sqrt(w[:, 0]**2 + w[:, 1]**2), 0)
        G = build_G(M, mu, nu)
        w_hat = spsolve(G.data, G.indices, G.indptr, G @ w.flatten('F') - T(M, q, w), tol = spsolve_tol)
        w_hat = w_hat.reshape(n, 2, order='F')

        def armijo_cond(t_k):
            w_next = w + t_k * (w_hat - w)
            return jnp.linalg.norm(T(M, q, w_next)) > (1 - lambd * t_k) * jnp.linalg.norm(T(M, q, w))


        t_k = jax.lax.while_loop(armijo_cond,
                                lambda t_k: t_k*1/2,
                                1/2)

        w_next = w + t_k * (w_hat - w)
        return w_next, k + 1

    w, k = jax.lax.while_loop(cond_fun, body_fun, (w_init, 0))


    if k == maxit:
        warnings.warn("Max iterations reached")
    
    return w[:, 0], w[:, 1]




# def lcp_fischer(M,q,x0=None, max_iter:int=10, b_tol:float=1e-6):
#     tol = 1.0e-12
#     mu = 1e-3
#     mu_step = 5
#     mu_min = 1e-5
#     n = M.shape[0]
#     x = jnp.ones(n) if x0 is None else x0
#     psi, phi, J = FB(x, q, M)
#     new_x = True
#     for iter in range(max_iter):
#         if new_x:
#             bad  = jnp.maximum(jnp.abs(phi), x)<b_tol
#             psi = psi - 0.5*phi[bad]@phi[bad]
#             J = J[~bad][:,~bad]
#             phi = phi[~bad]
#             new_x = False
#             nx = x
#             nx = nx.at[bad].set(0) 
        
#         H = J.T @ J + mu * sparse.eye(J.shape[0])
#         Jphi = J.T @ phi
#         H = BCSR.from_bcoo(H)
#         d = -spsolve(H.data, H.indices, H.indptr, Jphi)
#         H = H.to_bcoo()
#         nx = x.at[~bad].set(x[~bad] + d)
#         npsi, nphi, nJ = FB(nx, q, M)
#         r = (psi - npsi) / -(Jphi.T @ d + 0.5 * d.T @ H @ d)
#         mu = jax.lax.cond(r<0.3, lambda mu: jnp.maximum(mu * mu_step, mu_min), lambda mu: mu, mu)
#         #if r < 0.3:
#         #    mu = jnp.maximum(mu * mu_step, mu_min)
#         if r > 0:
#             x = nx
#             psi = npsi
#             phi = nphi
#             J = nJ
#             new_x = True
#             if r > 0.8:
#                 mu = mu / mu_step * (mu > mu_min)
#         if psi < tol:
#             break
#     return x




def FB(x,q,M):
    n = x.shape[0]
    y = M @ x + q
    w = jnp.column_stack([x,y])
    s = jnp.sqrt(x**2 + y**2)
    phi = s - x - y
    psi = 0.5 * phi@phi
    mu = x/s - 1
    nu = y/s - 1
    J = sparse.eye(n)*mu + nu.reshape(-1,1)*M
    return psi, phi, J


