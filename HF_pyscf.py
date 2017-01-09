import numpy as np
import scipy as sp
import scipy.linalg as spla
import pyscf as fci
from pyscf import gto, scf, ao2mo, fci

__idx2_cache = {}
def idx2(i,j):
    if (i,j) in __idx2_cache:
        return __idx2_cache[i,j]
    elif i>j:
        __idx2_cache[i,j] = int(i*(i+1)/2+j)
    else:
        __idx2_cache[i,j] = int(j*(j+1)/2+i)
    return __idx2_cache[i,j]

def idx4(i,j,k,l):
    """idx4(i,j,k,l) returns 2-tuple corresponding to (ij|kl) in
    square eri array (size n*(n-1)/2 square)"""
    return (idx2(i,j),idx2(k,l))

def RMSD(new_Dmatrix,old_Dmatrix):
        x = new_Dmatrix - old_Dmatrix
        x = np.power(RMS,2)
        x = np.sum(RMS)
        x = np.sqrt(RMS)
        return x

def G_matrix(old_Dmatrix,two_e_integrals):
    G = np.zeros(np.shape(old_Dmatrix))
    c=0
    for i in range(len(old_Dmatrix)):
        for j in range(len(old_Dmatrix)):
            for m in range(len(old_Dmatrix)):
                for n in range(len(old_Dmatrix)):
                    c += (old_Dmatrix[m,n])*((2.0*two_e_integrals[idx4(i,j,m,n)])-two_e_integrals[idx4(i,m,j,n)])
            G[i,j] = c
            c = 0
    return G

def F_matrix(Hcore,Gmatrix):
        F = np.zeros(np.shape(Hcore))
        for i in range(len(F)):
            for j in range(len(F)):
                F[i,j] = Hcore[i,j] + G[i,j]
        return F

def Eelec_calc(Dmatrix,Hcore,Fmatrix):
    Eelec = 0
    for i in range(len(Fmatrix)):
        for j in range(len(Fmatrix)):
            Eelec += Dmatrix[i,j]*(Hcore[i,j]+Fmatrix[i,j])
    return Eelec

def diagonalize(ortho, Fmatrix):
    F_prime = sp.dot(SOM,sp.dot(F,SOM))
    f_eval, f_evec = spla.eigh(F_prime)
    C = sp.dot(SOM,f_evec)
    return C

def D_matrix(Hcore,C):
    D = np.zeros(np.shape(Hcore))
    temp = 0
    for u in range(len(D)):
        for v in range(len(D)):
            for i in range(num_elec/2):
                temp += C[u,i] * C[v,i]
            D[u,v]=temp
            temp = 0
    return D

def orthogonalize(Smatrix):
    S_eval,S_evec = spla.eigh(Smatrix)
    #diagonalize the matrix
    S_eval_square=sp.zeros((7,7),dtype = np.float64)
    for i in range(len(S_eval)):
       S_eval_square[i,i] = S_eval[i]

    eval_change=sp.zeros((len(S_eval),len(S_eval)),dtype = np.float64)
    for i in range(len(eval_change)):
        eval_change[i,i]=1/sp.sqrt(S_eval[i])

    SOM = np.dot(S_evec,np.dot(eval_change,S_evec.T))   #This is the symmetric orthogonalization matrix. Change between atomic anc molecular basis.
    #Y = np.dot(np.transpose(SOM),np.dot(S,SOM))
    #print Y #This can be used to check the SOM, Y should be an identity matrix.
    return SOM

#### define the system to characterize using pyscf ####
mol = gto.M(
    atom = [['O', (0.000000000000,  -0.143225816552,   0.000000000000)],
            ['H', (1.638036840407,   1.136548822547,  -0.000000000000)],
            ['H', (-1.638036840407,   1.136548822547,  -0.000000000000)]],
    #basis = '6-31G',
    basis = 'sto-3g',
    verbose = 1,
    unit='b',
    symmetry=False
)

#### obtaining initial integrals from pyscf ####
myhf = scf.RHF(mol)
E_hf = myhf.kernel()
mo_coefficients = myhf.mo_coeff
S = myhf.get_ovlp()
Hcore =myhf.get_hcore()
eri = mol.intor('cint2e_sph',aosym=4)
Enuc = gto.mole.energy_nuc(mol)
num_elec = 10
true_value = E_hf
D=np.zeros((len(Hcore),len(Hcore)),dtype = np.float64)
k=0 #iteration counter
E_threshold = 1e-9  #energy convergence threshold
D_threshold = 1e-5  #density matrix convergence threshold
convergence = False
Etot_old  = 0
Etot_new  = 0



SOM = orthogonalize(S)
while (convergence == False):
    print '------- iteration',k,' -------'
    Dold = np.copy(D)

    #### G matrix ####
    G=G_matrix(Dold,eri)
    k+=1

    #### Fock matrix #####
    F = F_matrix(Hcore,G)

    #### Calculate total energy ####
    Etot_old = Etot_new
    Eelec = Eelec_calc(Dold,Hcore,F)
    Etot_new = Enuc + Eelec

    #### Find orbital coefficients ####
    C = diagonalize(SOM,F)

    #### Denisty matrix ####
    D = D_matrix(Hcore,C)

    #### RMS of D ####
    RMS = 0
    RMS=RMSD(D,Dold)
    print 'RMS: \t',RMS

    #### energy difference ####
    E_diff = np.abs(Etot_new - Etot_old)

    if (E_diff < E_threshold) and (RMS < D_threshold):
        convergence = True
    print 'E_diff: ', E_diff
print '-------  final results -------'
print 'final (Ha): \t', Etot_new
print 'error (Ha): \t', np.abs(Etot_new - true_value)
print 'final step count: \t', k
