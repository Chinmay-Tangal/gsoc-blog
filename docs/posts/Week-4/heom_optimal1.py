import time
import numpy as np
import scipy.sparse as sp
from math import factorial
from itertools import product

from qutip import basis, sigmaz, qeye, spre, spost, liouvillian, expect
from qutip.solver.heom import DrudeLorentzBath, HEOMSolver

import petsc4py
petsc4py.init()
from petsc4py import PETSc

comm_petsc = PETSc.COMM_WORLD
rank = comm_petsc.Get_rank()
nproc = comm_petsc.Get_size()

epsilon = 1.0
H    = 0.5 * epsilon * sigmaz()
rho0 = basis(2, 0) * basis(2, 0).dag()

lam   = 0.1
gamma = 0.5
T     = 1.0
Nk    = 8
Q     = sigmaz()

depth = 8
n     = 2
tlist = np.linspace(0, 20, 500)

bath = DrudeLorentzBath(Q=Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
exponents = bath.exponents
K = len(exponents)

nu = np.array([e.vk for e in exponents])   # decay rates
ck = np.array([e.ck for e in exponents])   # complex amplitudes

if rank == 0:
    print(f"K = {K} exponential terms")
    for k in range(min(K, 3)): # Print first 3 to save screen space
        print(f"  k={k}  nu={nu[k]:.4f}  c={ck[k]:.4f}")
    if K > 3: print("  ...")

# --- 4. Build ADO Labels ---
def enumerate_ado_labels(K, depth):
    labels = []
    for label in product(range(depth + 1), repeat=K):
        if sum(label) <= depth:
            labels.append(label)
    labels.sort(key=lambda x: (sum(x), x))
    return labels

ado_labels   = enumerate_ado_labels(K, depth)
label_to_idx = {lbl: i for i, lbl in enumerate(ado_labels)}
n_ados       = len(ado_labels)
Nsup         = n**2
n_total      = Nsup * n_ados

if rank == 0:
    print(f"\nADOs: {n_ados}   total DOF: {n_total}")

H_mat = H.full()      
Q_mat = Q.full()      
Id    = np.eye(n, dtype=complex)

L_sys = -1j * (np.kron(Id, H_mat) - np.kron(H_mat.T, Id))
Q_l   = np.kron(Id,    Q_mat)        
Q_r   = np.kron(Q_mat.T, Id)         
comm_Q = Q_l - Q_r                   

L_sys_sp  = sp.csr_matrix(L_sys)
Q_l_sp    = sp.csr_matrix(Q_l)
Q_r_sp    = sp.csr_matrix(Q_r)
comm_Q_sp = sp.csr_matrix(comm_Q)


A = PETSc.Mat().createAIJ(size=(n_total, n_total), comm=comm_petsc)
A.setUp()
rstart, rend = A.getOwnershipRange()

local_rows, local_cols, local_data = [], [], []

for ado_idx, label in enumerate(ado_labels):
    label    = list(label)
    row_off  = ado_idx * Nsup     

    # --- diagonal block ---
    decay      = sum(label[k] * nu[k] for k in range(K))
    diag_block = L_sys_sp - decay * sp.eye(Nsup, format="csr")

    cx  = diag_block.tocsr()
    for local_r in range(Nsup):
        gr = row_off + local_r
        if rstart <= gr < rend:
            cols_r = cx.indices[cx.indptr[local_r]:cx.indptr[local_r+1]]
            vals_r = cx.data   [cx.indptr[local_r]:cx.indptr[local_r+1]]
            for c, v in zip(cols_r, vals_r):
                local_rows.append(gr)
                local_cols.append(ado_idx * Nsup + c)
                local_data.append(v)

    # --- off-diagonal blocks ---
    for k in range(K):
        nk = label[k]

        # Coupling Down
        if nk >= 1:
            lbl_down = tuple(label[j] - int(j == k) for j in range(K))
            if lbl_down in label_to_idx:
                col_off  = label_to_idx[lbl_down] * Nsup
                cx = (-1j * nk * comm_Q_sp).tocsr()
                for local_r in range(Nsup):
                    gr = row_off + local_r
                    if rstart <= gr < rend:
                        cols_r = cx.indices[cx.indptr[local_r]:cx.indptr[local_r+1]]
                        vals_r = cx.data   [cx.indptr[local_r]:cx.indptr[local_r+1]]
                        for c, v in zip(cols_r, vals_r):
                            local_rows.append(gr)
                            local_cols.append(col_off + c)
                            local_data.append(v)

        # Coupling Up
        if sum(label) < depth:
            lbl_up = tuple(label[j] + int(j == k) for j in range(K))
            if lbl_up in label_to_idx:
                col_off = label_to_idx[lbl_up] * Nsup
                cx = (-1j * (ck[k] * Q_l_sp - np.conj(ck[k]) * Q_r_sp)).tocsr()
                for local_r in range(Nsup):
                    gr = row_off + local_r
                    if rstart <= gr < rend:
                        cols_r = cx.indices[cx.indptr[local_r]:cx.indptr[local_r+1]]
                        vals_r = cx.data   [cx.indptr[local_r]:cx.indptr[local_r+1]]
                        for c, v in zip(cols_r, vals_r):
                            local_rows.append(gr)
                            local_cols.append(col_off + c)
                            local_data.append(v)

for gr, gc, gv in zip(local_rows, local_cols, local_data):
    A.setValue(gr, gc, gv, addv=PETSc.InsertMode.INSERT_VALUES)
A.assemblyBegin()
A.assemblyEnd()

print(f"Rank {rank} successfully built rows {rstart} to {rend-1} (Local nnz: {len(local_data)})")

x = A.createVecRight()
f = A.createVecLeft()

if rank == 0:
    rho0_he = np.zeros(n_total, dtype=complex)
    rho0_he[:n**2] = rho0.full().ravel("F")
    x.setValues(range(n_total), rho0_he)
x.assemblyBegin()
x.assemblyEnd()

def rhs_petsc(ts, t, x, f):
    A.mult(x, f)

ts = PETSc.TS().create(comm=comm_petsc)
ts.setType(PETSc.TS.Type.RK)
ts.setRHSFunction(rhs_petsc, f)
ts.setTime(tlist[0])
ts.setMaxTime(tlist[-1])
ts.setTimeStep(tlist[1] - tlist[0])
ts.setTolerances(rtol=1e-8, atol=1e-10)
ts.setFromOptions()

comm_petsc.barrier()
t0 = time.perf_counter()
ts.solve(x)
t_petsc = time.perf_counter() - t0

if rank == 0:
    y_final = np.array(x.getArray())
    rho_f = y_final[:n**2].reshape((n, n), order="F")
    sz_f  = rho_f[0, 0].real - rho_f[1, 1].real
    print(f"\nPETSc+MPI distributed runtime: {t_petsc:.6f} s")
    print(f"<sz> at t_end                : {sz_f:.6f}")

if rank == 0:
    if nproc == 1:
        print("\n--- Running Single-Core SciPy Validation ---")
        from scipy.integrate import solve_ivp
        
        RHS_local = sp.csr_matrix((local_data, (local_rows, local_cols)), shape=(n_total, n_total), dtype=complex)
        
        def rhs_fn(t, y): return RHS_local @ y
        
        t0  = time.perf_counter()
        sol = solve_ivp(rhs_fn, t_span=(tlist[0], tlist[-1]), y0=rho0_he, t_eval=tlist, method="RK45", rtol=1e-8, atol=1e-10)
        t_sc = time.perf_counter() - t0
        print(f"SciPy runtime: {t_sc:.3f} s")
        
        solver_ref  = HEOMSolver(H, bath, max_depth=depth, options={"progress_bar": False})
        sz_ref      = expect(sigmaz(), solver_ref.run(rho0, tlist).states)
        
        sz_opt2 = np.array([sol.y[:n**2, i].reshape((n,n), order="F")[0,0].real - sol.y[:n**2, i].reshape((n,n), order="F")[1,1].real for i in range(len(tlist))])
        print(f"Max |err| vs QuTiP ref: {np.max(np.abs(sz_ref - sz_opt2)):.2e}")
    else:
        print("\n[Note]: SciPy validation skipped because nproc > 1.")
        print("[Note]: In Optimal 2, Rank 0 only owns a fraction of the matrix, so SciPy cannot be run on it.")