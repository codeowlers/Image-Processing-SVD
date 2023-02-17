import numpy as np

def compress(U_truncate, S_truncate, Vt_truncate):
   
    compressed_channel =  np.dot(U_truncate, np.dot(S_truncate, Vt_truncate))
    return compressed_channel   