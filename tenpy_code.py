import tenpy.linalg.np_conserved as npc
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS

from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms import dmrg


def example_exact_diagonalization(L, Jz):
    xxz_pars = dict(L=L, Jxx=1., Jz=Jz, hz=0.0, bc_MPS='finite')
    M = XXZChain(xxz_pars)

    product_state = ["up", "down"] * (xxz_pars['L'] // 2)  # this selects a charge sector!
    psi_DMRG = MPS.from_product_state(M.lat.mps_sites(), product_state)
    charge_sector = psi_DMRG.get_total_charge(True)  # ED charge sector should match

    ED = ExactDiag(M, charge_sector=charge_sector, max_size=2.e6)
    ED.build_full_H_from_mpo()
    # ED.build_full_H_from_bonds()  # whatever you prefer
    print("start diagonalization")
    ED.full_diagonalization()  # the expensive part for large L
    E0_ED, psi_ED = ED.groundstate()  # return the ground state
    print("psi_ED =", psi_ED)


if __name__ == "__main__":
    example_exact_diagonalization(L=10, Jz=1.)