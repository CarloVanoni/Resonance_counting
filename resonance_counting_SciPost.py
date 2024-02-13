from tabnanny import check
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.stats import lognorm
from scipy import sparse
from sympy import harmonic
from scipy import sparse
import sys, os
import numpy.linalg as linalg

rng = np.random.default_rng()

# Number of cores used by the code
nProc = 1

os.environ["MKL_NUM_THREADS"] = str(nProc)
os.environ["NUMEXPR_NUM_THREADS"] = str(nProc)
os.environ["OMP_NUM_THREADS"] = str(nProc)

def find_nearest(array, value, k):
    array = np.asarray(array)
    idx = np.argpartition(np.abs(array - value), k)
    return idx

def create_upper_matrix(values, size):
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 1)] = values
    return(upper)

#################  DRESSED HOPPINGS   #################

def compute_v(hop_vec, eigvec):
    v =[]
    for n in range(len(eigvec[:,0])):
        v.append(np.dot(hop_vec,eigvec[n,:]))
    return v

#################  ENERGY DIFFERENCES   #################

def compute_omega(eigvals,eps):
    return eigvals - eps

#################  CONTRIBUTION FROM ERGODIC BUBBLE   #################

def HaarS(Omega):
    return harmonic(Omega/2) - harmonic(1/2)

#################  SELF-CONSISTENT CRITERION  #################

def self_consistent_criterion(phi_an,N):
    phi_an = np.sort(phi_an)
    n = len(phi_an)

    stails = -np.sum(phi_an[:n]*np.log(phi_an[:n]))/n
    mean = 0
    for i in range(n):
        # Computes the running resonance probability, support set volume and wavefunction's head weight
        P = (i+1)/n
        Omega = 1+N*(1-P)
        mean = (mean*i + phi_an[i+1])/(i+1)
        C = 1-(N+1-Omega)*mean

        # Checks when the condition is first satisfied and stops if it is 
        if phi_an[i] > C/Omega:
            k = i+1

            stails = -np.sum(phi_an[:k]*np.log(phi_an[:k]))/k

            break

    return C, Omega, (N+1-Omega)*stails

#################  MATRIX GENERATORS  #################

def create_GRP(N,gamma,W):

    #Off_diag
    r = (N**(-gamma/2))*np.random.normal(0,1,N*(N-1)//2)
    c = create_upper_matrix(r, N)

    #Diagonal
    D = rng.uniform(-W,W,N)

    H = c + c.T + np.diag(D)

    return H

def create_BRP(N,K,W):

    #Off_diag
    r = np.random.binomial(1, np.min([K/N,1]), N*(N-1)//2) + N**(-2)*np.random.normal(0,1,N*(N-1)//2)
    c = create_upper_matrix(r, N)

    #Diagonal
    D = rng.uniform(-W,W,N)

    H = c + c.T + np.diag(D)

    return H

def create_BRP_sparse(N,K,W):

    #Off_diag
    r = np.random.binomial(1, np.min([K/N,1]), N*(N-1)//2)
    c = create_upper_matrix(r, N)

    #Diagonal
    D = rng.uniform(-W,W,N)

    H = c + c.T + np.diag(D)

    sH = sparse.csr_matrix(H)

    return sH

def create_LNRP(N,gamma,p,W):

    r = abs(N**(-gamma/2)*np.multiply((1-2*np.random.randint(0,2,size=N*(N-1)//2)) , lognorm.rvs(np.sqrt(gamma * p * np.log(N) / 2.), size=N*(N-1)//2)))

    
    # Fills the upper triangular matrix
    c = create_upper_matrix(r, N)

    #Diagonal
    D = rng.uniform(-W,W,size=N)

    # Full matrix
    H = c + c.T + np.diag(D)

    return H

N = int( sys.argv[1] )              # Matrix size
dis_num = int( sys.argv[2] )        # Number of realizations
gamma = float( sys.argv[3] )        # Value of gamma (parameter of GRP and LNRP)
W = float( sys.argv[4] )            # Value of onsite disorder (valid for BRP and LNRP only, W=1 for GRP)
model = int( sys.argv[5] )          # 0 => GRP, 1 => BRP, 2 => LNRP
size_phi = int( sys.argv[6] )       # Number of eigestates around E=0 to be considered
use_sparse = int( sys.argv[7] )     # 0 => full diagonalization methods, 1 => sparse diagonalization (valid only for BRP, always dense for GRP and LNRP)


nS_vec = np.zeros((dis_num))

phi_an = np.zeros((dis_num,size_phi))

for dis in range(dis_num):

    #GRP
    if model == 0:

        W = 1

        H0 = create_GRP(N,gamma,W)
        hop_vec = (N**(-gamma/2))*np.random.normal(0,1,N)
        eps = np.random.uniform(-W,W)

    #BRP
    elif model == 1:

        K=3     #Average connectivity

        # Full matrix
        if use_sparse == 0:
            H0 = create_BRP(N,K,W)
        
        # Sparse matrix
        else:
            H0 = create_BRP_sparse(N,K,W)
        hop_vec = np.random.binomial(1, np.min([K/N,1]), N) + N**(-2)*np.random.normal(0,1,N)
        eps = np.random.uniform(-W,W)
    
    #LNRP
    elif model == 2:

        p=1     # Log-normal parameter

        H0 = create_LNRP(N,gamma,p,W)
        hop_vec = abs(N**(-gamma/2)*np.multiply((1-2*np.random.randint(0,2,size=N)) , lognorm.rvs(np.sqrt(gamma * p * np.log(N) / 2.), size=N)))
        eps = np.random.uniform(-W,W)

    # Exact diagonalization
        
    # Full matrix
    if use_sparse == 0:
        eigvals, eigvecs = linalg.eigh(H0)
        eigvecs = eigvecs.T

        indx = np.argsort(np.abs(eigvals))
        indx = indx[:size_phi]
        eigvals = eigvals[indx]
        eigvecs = eigvecs[indx,:]
    
    # Sparse matrix
    else:
        eigvals, eigvecs = eigsh(H0, k=size_phi, sigma=0, which='LM')
        eigvecs = eigvecs.T

    #Dressed hoppings
    v = np.array(compute_v(hop_vec,eigvecs))

    #Energy difference
    omega_an = compute_omega(eigvals,eps)

    #Ratio for resonance condition
    phi_an[dis,:] = np.divide(v**2,omega_an**2)

   
    #################  Numerical participation entropy  #################

    S = 0
    for k in range(len(eigvals)):
        ev2 = np.square(eigvecs[k])
        ev2 = np.where(ev2 < 1e-12, 1e-12, ev2)

        S += -np.sum( np.multiply(ev2,np.log(ev2)))
    
    nS_vec[dis] = S/len(eigvals)

nS_av = np.mean(nS_vec)

#################  SELF-CONSISTENT RESONANCE CONDITION  #################
phi_an = phi_an.flatten()
c, Omega_av, stails = self_consistent_criterion(phi_an,N)

S_av = stails + (c*HaarS(Omega_av) - c*np.log(c))

#################  SAVE TO FILE  #################

toSave = np.array([c,stails,Omega_av,S_av,nS_av])

if model == 0:
    filename = "Results_res/res_GRP_N%d_gamma%.2f_dis%d.txt"%(N,gamma,dis_num)
    np.savetxt(filename,toSave)
elif model == 1:
    filename = "Results_res/res_BRP_N%d_W%.2f_dis%d.txt"%(N,W,dis_num)
    np.savetxt(filename,toSave)
elif model == 2:
    filename = "Results_res/res_LN_N%d_W%.2f_dis%d.txt"%(N,W,dis_num)
    np.savetxt(filename,toSave)
