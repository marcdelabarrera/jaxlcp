from .fischer import lcp_fischer
#from lemke import _lcp_lemke
from jax import Array
from jax.experimental import sparse

def lcp(M: Array, q: Array, algorithm:str = None, **kwargs)->tuple[Array, Array]:
    if algorithm is None:
        if isinstance(M, sparse.BCOO) or isinstance(M, sparse.BCSR):
            algorithm = 'fischer'
        else:
            algorithm = 'lemke'

    if algorithm == 'lemke':
        return _lcp_lemke(M,q)
    if algorithm == 'fischer':
        return lcp_fischer(M,q,**kwargs)
    else:
        raise ValueError('Algorithm must be either "lemke" or "fischer"')


