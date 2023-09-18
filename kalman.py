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
    TdNc = 16*data['eeg'].shape[1] # 16 is Td in index
    Cxx = np.zeros((TdNc,TdNc))
    for path in paths:
        for k in range(20):
            data = AAD_data(path,k)
            # Get the sequential data
            Xw,_ = window_XY(data['eeg'],data['env'])
            # Append to Cxx
            Cxx = Cxx + Xw.T @ Xw
            cnt += 1
    Cxx = Cxx / cnt

    # Initialize the Cxy
    if cxy_mode == "rand":
        Cxy = lambd * np.random.rand(TdNc,1)
    else: # if det
        Cxy = cxy0

    # Initialize Decoder
    w0 = np.linalg.pinv(Cxx) @ Cxy

    return Cxx,Cxy,w0

def eval_corr(y,yhat):
    """
    Evaluates normalized correlation between y and yhat
    Returns a magnitude to indicate "strength" of resemblence
    
    y : Reference Signal (T,1)
    yhat : SOI (T,1)
    """
    
    # Step 1 : Scale down
    y = y/np.linalg.norm(y)
    yhat = yhat/np.linalg.norm(yhat)
    
    # Step 2 : Dot product
    return np.dot(y.flatten(),yhat.flatten())


def AAD_forward(paths,Cxx,Cxy0,Ntrial):
    """
    Evaluate single iteration of update for AAD Unsupervised Training
    
    paths : each path corresponding to one subject
    Cxx : True Cxx for paths used
    Cxy0 : Prior Cxy used for decoder evaluation for current iteration
    Ntrial : Number of trials used for this iteration

    Output
    w : New weight
    """
    Cxy = Cxy0
    w = np.linalg.pinv(Cxx) @ Cxy
    
    # Store labels used for 
    cnt = 0
    for path in paths:
        Cxy = np.zeros_like(Cxy)
        for k in range(Ntrial):
            # Get current data
            cur_data = AAD_data(path,k)
            Xw,Yw = window_XY(cur_data['eeg'],cur_data['env'])
            # Get prior prediction
            yprior = Xw@w
            # Return argmax of the two correlations
            rho = [eval_corr(y,yprior) for y in Yw]
            idx = rho.index(max(rho))
            ypost = Yw[idx]
            # Evaluate new cross correlation
            Cxy = Cxy + Xw.T @ ypost  
            cnt += 1
    
    Cxy = Cxy * 1/cnt
    # Update Decoder
    w = np.linalg.pinv(Cxx) @ Cxy

    return w,Cxy

def AAD_test(paths,w,Ntrial):
    score = 0
    cnt = 0
    for path in paths:
        for k in range(Ntrial):
            # Get current data
            cur_data = AAD_data(path,k)
            Xw,Yw = window_XY(cur_data['eeg'],cur_data['env'])
            label = dir2label(cur_data['lab'])

            # Get prediction env
            yprior = Xw@w

            # Return argmax
            rho = [eval_corr(y,yprior) for y in Yw]
            idx = rho.index(max(rho))+1
            if idx == label:
                score += 1
            cnt += 1

    return score/cnt
            
            
    
    
    
    
