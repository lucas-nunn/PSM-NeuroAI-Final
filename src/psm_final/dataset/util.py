import scipy.io as sio

def shared_stimuli(nsd_mat):
    exp = sio.loadmat(nsd_mat)
    sharedix = exp["sharedix"].flatten()
    shared_ids = set((sharedix).tolist())

    return shared_ids