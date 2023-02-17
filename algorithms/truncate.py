import numpy as np

def truncate(U, S, Vt, num_sv):
    U_trunc = U[:, :num_sv]
    S_trunc = np.diag(S[:num_sv])
    Vt_trunc = Vt[:num_sv, :]

    return U_trunc, S_trunc, Vt_trunc