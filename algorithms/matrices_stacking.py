from algorithms import truncate
import numpy as np


# To stack the truncated matrices for each color channel, you can simply concatenate them along the second axis (
# i.e., axis=1), since the first axis represents the rows of each matrix. Here's an example of how to do this for the
# U matrices:
def stack_matrices(U_r, U_g, U_b, num_sv):
    # Truncate the matrices for each color channel
    U_r_trunc, _, _ = truncate(U_r, None, None, num_sv)
    U_g_trunc, _, _ = truncate(U_g, None, None, num_sv)
    U_b_trunc, _, _ = truncate(U_b, None, None, num_sv)

    # Stack the truncated matrices horizontally
    Ur = np.hstack((U_r_trunc, np.zeros((U_r_trunc.shape[0], U_g_trunc.shape[1] + U_b_trunc.shape[1]))))
    Ug = np.hstack((np.zeros((U_g_trunc.shape[0], U_r_trunc.shape[1])), U_g_trunc,
                    np.zeros((U_g_trunc.shape[0], U_b_trunc.shape[1]))))
    Ub = np.hstack((np.zeros((U_b_trunc.shape[0], U_r_trunc.shape[1] + U_g_trunc.shape[1])), U_b_trunc))

    return Ur, Ug, Ub


# This function takes the truncated U matrices for each color channel, concatenates them horizontally using
# np.hstack, and pads the matrices with zeros so they have the same number of columns. The same approach can be used
# for the Vt and S matrices.
#
# To stack all of the matrices into one, you can concatenate the U, S, and Vt matrices along the first axis (i.e.,
# axis=0) using np.vstack. Here's an example:
def stack_all_matrices(U_r, U_g, U_b, S_r, S_g, S_b, Vt_r, Vt_g, Vt_b, num_sv):
    # Stack the matrices for each color channel
    Ur, Ug, Ub = stack_matrices(U_r, U_g, U_b, num_sv)
    Vtr, Vtg, Vtb = stack_matrices(Vt_r, Vt_g, Vt_b, num_sv)
    Sr = np.vstack(
        (S_r[:num_sv, :num_sv], np.zeros((S_g.shape[0] - num_sv, num_sv)), np.zeros((S_b.shape[0] - num_sv, num_sv))))
    Sg = np.vstack(
        (np.zeros((num_sv, S_r.shape[1])), S_g[:num_sv, :num_sv], np.zeros((S_b.shape[0] - num_sv, S_g.shape[1]))))
    Sb = np.vstack((np.zeros((num_sv, S_r.shape[1] + S_g.shape[1])), S_b[:num_sv, :num_sv]))

    # Stack all matrices
    Uall = np.vstack((Ur, Ug, Ub))
    Vtall = np.vstack((Vtr, Vtg, Vtb))
    Sall = np.hstack((np.vstack((Sr, np.zeros((S_r.shape[0], S_g.shape[1] + S_b.shape[1])))),
                      np.vstack((np.zeros((S_g.shape[0], S_r.shape[1])), Sg, np.zeros((S_g.shape[0], S_b.shape[1])))),
                      np.vstack((np.zeros((S_b.shape[0], S_r.shape[1] + S_g.shape[1])), Sb))))

    return Uall, Sall, Vtall

# Ur, Ug, Ub = stack_matrices(U_r_trunc, U_g_trunc, U_b_trunc, num_sv)
# Vtr, Vtg, Vtb = stack_matrices(Vt_r_trunc, Vt_g_trunc, Vt_b_trunc, num_sv)
# Sr, Sg, Sb = S_r_trunc, S_g_trunc, S_b_trunc
#
# # Stack all matrices
# Uall, Sall, Vtall = stack_all_matrices(U_r_trunc, U_g_trunc, U_b_trunc, S_r_trunc, S_g_trunc, S_b_trunc, Vt_r_trunc, Vt_g_trunc, Vt_b_trunc, num_sv)
