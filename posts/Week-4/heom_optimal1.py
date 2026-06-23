import time
import numpy as np
import scipy.sparse as sp
from math import factorial

from qutip import basis, sigmaz, qeye, spre, spost, liouvillian, state_number_enumerate, expect
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath

# --- 1. Get MPI Rank Early (so we can control prints) ---
try:
    from mpi4py import MPI
    comm_global = MPI.COMM_WORLD
    rank = comm_global.Get_rank()
except ImportError:
    rank = 0

# --- 2. System Setup ---
epsilon = 1.0
H    = 0.5 * epsilon * sigmaz()
rho0 = basis(2, 0) * basis(2, 0).dag()

lam   = 0.1
gamma = 0.5
T     = 1.0
Nk    = 8
Q     = sigmaz()

depth = 2
n     = 2
tlist = np.linspace(0, 20, 500)

# --- 3. QuTiP Reference ---
bath   = DrudeLorentzBath(Q=Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
solver = HEOMSolver(H, bath, max_depth=depth, options={"progress_bar": False})

t0 = time.perf_counter()
result_ref = solver.run(rho0, tlist)
t_ref = time.perf_counter() - t0
sz_ref = expect(sigmaz(), result_ref.states)

if rank == 0:
    print(f"QuTiP baseline: {t_ref:.3f} s")

# --- 4. Build RHS Matrix Serially ---
exponents = bath.exponents
if rank == 0:
    print(f"bath exponents: {len(exponents)}")
    for k, ex in enumerate(exponents):
        print(f"  k={k}  nu={ex.vk:.4f}  c={ex.ck:.4f}")

ado_labels   = solver.ados.labels
label_to_idx = {lbl: i for i, lbl in enumerate(ado_labels)}
n_ados       = len(ado_labels)
n_total      = n**2 * n_ados

if rank == 0:
    print(f"ADOs: {n_ados}   state-vector length: {n_total}")

Nsup   = n**2
L_sys  = sp.csr_matrix(liouvillian(H, []).full())
Q_l    = sp.csr_matrix(spre(Q).full())
Q_r    = sp.csr_matrix(spost(Q).full())
comm_Q = Q_l - Q_r

rows, cols, data = [], [], []

def add_block(i_block, j_block, mat):
    r0, c0 = i_block * Nsup, j_block * Nsup
    coo = sp.csr_matrix(mat).tocoo()
    for r, c, v in zip(coo.row, coo.col, coo.data):
        rows.append(r0 + r)
        cols.append(c0 + c)
        data.append(v)

for idx, label in enumerate(ado_labels):
    label = list(label)
    decay = sum(label[k] * exponents[k].vk for k in range(len(exponents)))
    add_block(idx, idx, L_sys - decay * sp.eye(Nsup, format="csr"))

    for k, ex in enumerate(exponents):
        ck = ex.ck
        nk = label[k]
        if nk >= 1:
            lbl_down = tuple(label[j] - int(j == k) for j in range(len(exponents)))
            if lbl_down in label_to_idx:
                add_block(idx, label_to_idx[lbl_down], -1j * nk * comm_Q)
        if sum(label) < depth:
            lbl_up = tuple(label[j] + int(j == k) for j in range(len(exponents)))
            if lbl_up in label_to_idx:
                add_block(idx, label_to_idx[lbl_up], -1j * (ck * Q_l - np.conj(ck) * Q_r))

RHS_opt1 = sp.csr_matrix((data, (rows, cols)), shape=(n_total, n_total), dtype=complex)

if rank == 0:
    print(f"RHS shape: {RHS_opt1.shape}   nnz: {RHS_opt1.nnz}")
    RHS_qutip = sp.csr_matrix(solver.rhs(0).full())
    diff = RHS_opt1 - RHS_qutip
    print(f"max |RHS_opt1 - RHS_qutip| = {abs(diff).max():.2e}")

rho0_he = np.zeros(n_total, dtype=complex)
rho0_he[:n**2] = rho0.full().ravel("F")

# --- 5. PETSc TS integrator (Distributed ODE Solve) ---
USE_PETSC = True  

if USE_PETSC:
    import petsc4py
    petsc4py.init()
    from petsc4py import PETSc

    comm = PETSc.COMM_WORLD

    A = PETSc.Mat().createAIJ(size=(n_total, n_total), comm=comm)
    A.setUp()

    rstart, rend = A.getOwnershipRange()
    for r in range(rstart, rend):
        c_ = RHS_opt1.indices[RHS_opt1.indptr[r]:RHS_opt1.indptr[r+1]]
        v_ = RHS_opt1.data[RHS_opt1.indptr[r]:RHS_opt1.indptr[r+1]]
        A.setValues([r], c_, v_, addv=PETSc.InsertMode.INSERT_VALUES)
    A.assemblyBegin()
    A.assemblyEnd()

    x = A.createVecRight()
    f = A.createVecLeft()
    
    if rank == 0:
        x.setValues(range(n_total), rho0_he)
    x.assemblyBegin()
    x.assemblyEnd()

    def rhs_petsc(ts, t, x, f):
        A.mult(x, f)

    ts = PETSc.TS().create(comm=comm)
    ts.setType(PETSc.TS.Type.RK)
    ts.setRHSFunction(rhs_petsc, f)
    ts.setTime(tlist[0])
    ts.setMaxTime(tlist[-1])
    ts.setTimeStep(tlist[1] - tlist[0])
    ts.setTolerances(rtol=1e-8, atol=1e-10)
    ts.setFromOptions()

    t0 = time.perf_counter()
    ts.solve(x)
    t_petsc = time.perf_counter() - t0

    if rank == 0:
        y_final = np.array(x.getArray())
        rho_f = y_final[:n**2].reshape((n, n), order="F")
        sz_final = rho_f[0, 0].real - rho_f[1, 1].real
        print(f"PETSc+MPI runtime : {t_petsc:.3f} s")
        print(f"<sz> at t_end     : {sz_final:.6f}")

# --- 6. SciPy Validation (Rank 0 Only!) ---
if rank == 0:
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    def rhs_fn(t, y):
        return RHS_opt1 @ y

    t0  = time.perf_counter()
    sol = solve_ivp(rhs_fn, t_span=(tlist[0], tlist[-1]), y0=rho0_he, t_eval=tlist, method="RK45", rtol=1e-8, atol=1e-10)
    t_sc = time.perf_counter() - t0

    def extract_sz(y_col):
        rho = y_col[:n**2].reshape((n, n), order="F")
        return rho[0, 0].real - rho[1, 1].real

    sz_opt1 = np.array([extract_sz(sol.y[:, i]) for i in range(len(tlist))])
    max_err = np.max(np.abs(sz_ref - sz_opt1))

    print(f"scipy runtime          : {t_sc:.3f} s")
    print(f"max |err| vs QuTiP ref : {max_err:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(tlist, sz_ref,  lw=2,            label="QuTiP ref")
    axes[0].plot(tlist, sz_opt1, lw=1.5, ls="--", label="Optimal 1 (dict RHS)")
    axes[0].set_xlabel("time"); axes[0].set_ylabel(r"$\langle\sigma_z\rangle$")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].semilogy(tlist, np.abs(sz_ref - sz_opt1) + 1e-16)
    axes[1].set_xlabel("time"); axes[1].set_ylabel("pointwise error")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("optimal1_validation.png", dpi=150, bbox_inches="tight")
    
    # We comment this out so it doesn't freeze the terminal waiting for you to close the window
    # plt.show()
