import numpy as np
from scipy import sparse

Id = sparse.csr_matrix(np.eye(2))
Sx = (1/2)*sparse.csr_matrix([[0., 1.], [1., 0.]])
Sy = (1/2)*sparse.csr_matrix([[0., -1.j], [1.j, 0.]])
Sz = (1/2)*sparse.csr_matrix([[1., 0.], [0., -1.]])
Splus = sparse.csr_matrix([[0., 1.], [0., 0.]])
Sminus = sparse.csr_matrix([[0., 0.], [1., 0.]])


def singlesite_to_full(op, i, L):
    op_list = [Id]*L  # = [Id, Id, Id ...] with L entries
    op_list[i] = op
    full = op_list[0]
    for op_i in op_list[1:]:
        full = sparse.kron(full, op_i, format="csr")
    return full


def gen_sx_list(L):
    return [singlesite_to_full(Sx, i, L) for i in range(L)]

def gen_sy_list(L):
    return [singlesite_to_full(Sy, i, L) for i in range(L)]

def gen_sz_list(L):
    return [singlesite_to_full(Sz, i, L) for i in range(L)]


def gen_hamiltonian_periodic(sx_list, sz_list, g, J=1.):
    """ assumes periodic boundery conditions """
    L = len(sx_list)
    H = sparse.csr_matrix((2**L, 2**L))
    for j in range(L):
        H = H - J *( sx_list[j] * sx_list[(j+1)%L])
        H = H - g * sz_list[j]
    return H


def gen_hamiltonian(sx_list, sz_list, g, J=1.):
    """ assumes open boundary conditions """
    L = len(sx_list)
    H = sparse.csr_matrix((2**L, 2**L))
    for j in range(L-1):
        H = H - J *( sx_list[j] * sx_list[(j+1)%L])
        H = H - g * sz_list[j]
    H = H - g * sz_list[-1]
    return H


def gen_hamiltonian_lists(L, h, J=1):
    sx_list = gen_sx_list(L)
    sy_list = gen_sy_list(L)
    sz_list = gen_sz_list(L)
    H = sparse.csr_matrix((2 ** L, 2 ** L))
    H = H - J*(sx_list[0] * sx_list[1] + sy_list[0] * sy_list[1] + sz_list[0] * sz_list[1]) - h[0]*sz_list[0]
    for i in range(1, L-1):
        H += - J*(sx_list[i] * sx_list[i+1] + sy_list[i] * sy_list[i+1] + sz_list[i] * sz_list[i+1]) - h[i]*sz_list[i]
    return H

# fixme delete

# def gen_hamiltonian_random_h(L, W, J=1.):
#     """ assumes open boundary conditions """
#     sx_list = gen_sx_list(L)
#     sz_list = gen_sz_list(L)
#     H = sparse.csr_matrix((2**L, 2**L))
#     for j in range(L-1):
#         H = H - J *( sx_list[j] * sx_list[(j+1)%L])
#         H = H - np.random.uniform(-W, W) * sz_list[j]
#     H = H - np.random.uniform(-W, W) * sz_list[-1]
#     return H
