####### Import #################
import numpy as np
import scipy.io as spio

####### Loading MATLAB Data ###########
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


####### Data Loading for AAD (Bertrand) #########

def dir2label(direction):
    """
    Converts direction (either "L" or "R" into label 1 or 2
    """
    if direction == 'L':
        return 1

    else:
        return 2

def eval_weighted_env(X,W):
    """
    Linear combination of multi_channel output of the envelopes using weight

    X: T x Nc x 2
    W : 1 x 15

    Returns
    Y : list of [yL,yR] each with T x 1
    """

    # List of tracks : List of T x Nc matrices
    X = [X[:,:,0],X[:,:,1]]
    Y = []
    for i in range(len(X)):
        # Apply the transform
        Yi = X[i]@W.T
        Y.append(Yi.reshape((-1,1)))

    return Y
   
   
def AAD_data(path,k):
    """
    Loads up Bertrand .mat file for particular application

    path : Path for the AAD .mat file
    k : index of interest

    Output:
    Xk : EEG at trial k
    yk : List of Envelope channel at trial k [yL,yR]
    label : 1 or 2 for left or right
    """
    trial = loadmat(path)['preproc_trials'][k]
    Xk = trial.RawData.EegData
    wk = trial.Envelope.subband_weights
    ck = dir2label(trial.attended_ear)
    yk = trial.Envelope.AudioData # T x channel x 2 (binaual)

    # Evaluate weighted envelope
    yk = eval_weighted_env(yk,wk)

    return {'eeg':Xk, 'env':yk, 'lab':ck}


################## AAD Bertrand Helper Function ######################
def window_XY(X,Y,fs=32, Td=0.500,Tmax=30):
    """
    Window the EEG and the envelopes according to decoding length

    X : EEG (T x Nc_eeg)
    Y : Envelopes (list of Tx1)
    fs : Sample rate (hz)
    Td : Window size (Observation Window) in s
    Tmax : Maximum sample collected before shutdown (Decision Window) in s

    Output:
    Xw : EEG (T,Nc_eeg*Td)
    """

    Xw = []
    Yw = []
    delta = fs
    for t in range(Tmax*fs):

        # We will sample exactly the correct regions of interest
        # We sample from the middle
        Xw.append(X[delta+t:delta+t+int(Td*fs),:].flatten('F'))

    Yw.append(Y[0][delta:delta+int(Tmax*fs)])
    Yw.append(Y[1][delta:delta+int(Tmax*fs)])

    return np.array(Xw),Yw


def init_params_AAD(paths,cxy_mode = "rand",cxy0 = None,lambd = 0.5):
    """
    Initialize Cxx, Cxy and w using all the data from listed paths

    paths : List of paths, thus the matrices to consider to build our parameters
    cxy_mode : "rand" if randomly initialize, "det" if deterministically initialize
    cxy0 : Prior of not rand
    """

    # Initialize the Cxx
    cnt = 0
    # Sample one example for size
    data = AAD_data(paths[0],0)
    TdNc = data['eeg'].shape[1]
    Cxx = np.zeros((TdNc,TdNc))
    for path in paths:
        for k in range(20):
            data = AAD_data(path,k)
            # Get the sequential data
            Xw,Yw = Xwindow_XY(data['eeg'],data['env'])
            # Append to Cxx
            Cxx = Cxx + X.T @ X
            cnt += 1
    Cxx = Cxx / cnt

    # Initialize the Cxy
    if cxy_model == "rand":
        Cxy = lambd * np.random.rand((TdNc,1))
    else: # if det
        Cxy = cxy0

    # Initialize Decoder
    w0 = np.linalg.pinv(Cxx) @ Cxy

    return Cxx,Cxy,w0