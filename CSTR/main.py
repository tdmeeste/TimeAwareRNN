import os
import sys
import numpy as np
from time import time
import torch
GPU = torch.cuda.is_available()

parent = os.path.dirname(sys.path[0])#os.getcwd())
sys.path.append(parent)
from taho.model import MIMO, GRUCell, HOGRUCell, IncrHOGRUCell, HOARNNCell, IncrHOARNNCell
from taho.train import EpochTrainer
from taho.util import SimpleLogger, show_data

from tensorboard_logger import configure, log_value
import argparse
import os
import pickle
import sys
import traceback
import shutil


"""
potentially varying input parameters
"""
parser = argparse.ArgumentParser(description='Models for Continuous Stirred Tank dataset')

# model definition
methods = """
set up model
- model:
    GRU (compensated GRU to avoid linear increase of state; has standard GRU as special case for Euler scheme and equidistant data)
    GRUinc (incremental GRU, for baseline only)
    ARNN (compensated ARNN to avoid linear increase of state)
    ARNNinc (incremental ARNN)
- time_aware:
    no: ignore uneven spacing: for GRU use original GRU implementation; ignore 'scheme' variable
    input: use normalized next interval size as extra input feature
    variable: time-aware implementation
"""

temporals = """
'current': predict current output given current input and previous state
'next': predict next output given current input and current state (and possibly next interval length)
"""



parser.add_argument("--time_aware", type=str, default='variable', choices=['no', 'input', 'variable'], help=methods)
parser.add_argument("--model", type=str, default='GRU', choices=['GRU', 'GRUinc', 'ARNN', 'ARNNinc'])


parser.add_argument("--gamma", type=float, default=1.0, help="diffusion parameter ARNN model")
parser.add_argument("--step_size", type=float, default=1.0, help="fixed step size parameter in the ARNN model")
parser.add_argument("--temporal", type=str, default='next', choices=['current', 'next'], help=temporals)

# data
parser.add_argument("--missing", type=float, default=0.0, help="fraction of missing samples (0.0 or 0.5)")

# model architecture
parser.add_argument("--k_state", type=int, default=20, help="dimension of hidden state")

# in case method == 'variable'
RKchoices = ['Euler', 'Midpoint', 'Kutta3', 'RK4']
parser.add_argument("--scheme", type=str, default='Euler', choices=RKchoices, help='Runge-Kutta training scheme')

# training
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--epochs", type=int, default=4000, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--bptt", type=int, default=20, help="bptt")
parser.add_argument("--dropout", type=float, default=0., help="drop prob")
parser.add_argument("--l2", type=float, default=0., help="L2 regularization")


# admin
parser.add_argument("--save", type=str, default='results', help="experiment logging folder")
parser.add_argument("--eval_epochs", type=int, default=20, help="validation every so many epochs")
parser.add_argument("--seed", type=int, default=0, help="random seed")

# during development
parser.add_argument("--reset", action="store_true", help="reset even if same experiment already finished")

#parser.add_argument('--feature', dest='feature', action='store_true')
#parser.add_argument('--no-feature', dest='feature', action='store_false')
#parser.set_defaults(feature=True)


paras = parser.parse_args()

hard_reset = paras.reset
# if paras.save already exists and contains log.txt:
# reset if not finished, or if hard_reset
log_file = os.path.join(paras.save, 'log.txt')
if os.path.isfile(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        completed = 'Finished' in content
        if completed and not hard_reset:
            print('Exit; already completed and no hard reset asked.')
            sys.exit()  # do not overwrite folder with current experiment
        else:  # reset folder
            shutil.rmtree(paras.save, ignore_errors=True)



# setup logging
logging = SimpleLogger(log_file) #log to file
configure(paras.save) #tensorboard logging
logging('Args: {}'.format(paras))





"""
fixed input parameters
"""
frac_dev = 15/100
frac_test = 15/100

GPU = torch.cuda.is_available()
logging('Using GPU?', GPU)

# set random seed for reproducibility
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)
np.random.seed(paras.seed)


"""
Load data
"""

data = np.loadtxt('./data/cstr_normalized_missing_prob_%.2f.dat' % paras.missing)


t = np.expand_dims(data[:, 0], axis=1)  # (Nsamples, 1)
X = np.expand_dims(data[:, 1], axis=1)  # (Nsamples, 1)
Y = data[:, 2:4]  # (Nsamples, 2)
k_in = X.shape[1]
k_out = Y.shape[1]

dt = np.expand_dims(data[:, 4], axis=1)  # (Nsamples, 1)
logging('loaded data, \nX', X.shape, '\nY', Y.shape, '\nt', t.shape, '\ndt', dt.shape,
        '\ntime intervals dt between %.3f and %.3f wide (%.3f on average).'%(np.min(dt), np.max(dt), np.mean(dt)))



N = X.shape[0]  # number of samples in total
Ndev = int(frac_dev * N)
Ntest = int(frac_test * N)
Ntrain = N - Ntest - Ndev

logging('first {} for training, then {} for development and {} for testing'.format(Ntrain, Ndev, Ntest))

"""
evaluation function
RRSE error
"""
def prediction_error(truth, prediction):
    assert truth.shape == prediction.shape, "Incompatible truth and prediction for calculating prediction error"
    # each shape (sequence, n_outputs)
    # Root Relative Squared Error
    se = np.sum((truth - prediction) ** 2, axis=0)  # summed squared error per channel
    rse = se / np.sum((truth - np.mean(truth, axis=0))**2)  # relative squared error
    rrse = np.mean(np.sqrt(rse))  # square root, followed by mean over channels
    return 100 * rrse  # in percentage



# prepare input data:
if paras.temporal == 'current':
    Xtrain = X[:Ntrain, :]
    dttrain = dt[:Ntrain, :]
    Ytrain = Y[:Ntrain, :]
    ttrain = t[:Ntrain, :] #for visualization

    Xdev = X[Ntrain:Ntrain+Ndev, :]
    dtdev = dt[Ntrain:Ntrain+Ndev, :]
    Ydev = Y[Ntrain:Ntrain+Ndev, :]
    tdev = t[Ntrain:Ntrain+Ndev, :]

    Xtest = X[Ntrain+Ndev:, :]
    dttest = dt[Ntrain + Ndev:, :]
    Ytest = Y[Ntrain+Ndev:, :]
    ttest = t[Ntrain+Ndev:, :]

elif paras.temporal == 'next':
    Xtrain = X[:Ntrain, :]
    dttrain = dt[:Ntrain, :]
    Ytrain = Y[1:Ntrain+1, :]
    ttrain = t[1:Ntrain+1, :]

    Xdev = X[Ntrain:Ntrain+Ndev, :]
    dtdev = dt[Ntrain:Ntrain+Ndev, :]
    Ydev = Y[Ntrain+1:Ntrain+Ndev+1, :]
    tdev = t[Ntrain+1:Ntrain+Ndev+1, :]

    Xtest = X[Ntrain+Ndev:-1, :]
    dttest = dt[Ntrain+Ndev:-1, :]  #last value was added artificially in data_processing.py, but is not used.
    Ytest = Y[Ntrain+Ndev+1:, :]
    ttest = t[Ntrain + Ndev + 1:, :]

else:
    raise IOError


"""
- model:
    GRU (compensated GRU to avoid linear increase of state; has standard GRU as special case for Euler scheme and equidistant data)
    GRUinc (incremental GRU, for baseline only)
    ARNN (compensated ARNN to avoid linear increase of state)
    ARNNinc (incremental ARNN)
- time_aware:
    no: ignore uneven spacing: for GRU use original GRU implementation
    input: use normalized next interval size as extra input feature
    variable: time-aware implementation
"""


#time_aware options

if paras.time_aware == 'input':
    assert paras.temporal == 'next', 'For baseline method with input extended with next interval length, temporal = next needed.' \
                                     'makes no sense otherwise.'

    # expand X matrices with additional input feature, i.e., normalized duration dt to next sample
    dt_mean, dt_std = np.mean(dttrain), np.std(dttrain)
    dttrain_n = (dttrain - dt_mean) / dt_std
    dtdev_n = (dtdev - dt_mean) / dt_std
    dttest_n = (dttest - dt_mean) / dt_std

    Xtrain = np.concatenate([Xtrain, dttrain_n], axis=1)
    Xdev = np.concatenate([Xdev, dtdev_n], axis=1)
    Xtest = np.concatenate([Xtest, dttest_n], axis=1)

    k_in += 1

if paras.time_aware == 'no' or paras.time_aware == 'input':
    # in case 'input': variable intervals already in input X;
    # now set actual time intervals to 1 (else same effect as time_aware == 'variable')
    dttrain = np.ones(dttrain.shape)
    dtdev = np.ones(dtdev.shape)
    dttest = np.ones(dttest.shape)

if paras.time_aware == 'variable':
    assert paras.temporal == 'next', 'For time-aware GRU, require temporal = next (makes no sense otherwise).'

# set model:
if paras.model == 'GRU':
    cell_factory = GRUCell if paras.time_aware == 'no' else HOGRUCell
elif paras.model == 'GRUinc':
    cell_factory = IncrHOGRUCell
elif paras.model == 'ARNN':
    cell_factory = HOARNNCell
elif paras.model == 'ARNNinc':
    cell_factory = IncrHOARNNCell
else:
    raise NotImplementedError('unknown model type ' + paras.model)

#model = StatefulMIMO(k_in, k_out, paras.k_state, dropout=paras.dropout, cell_factory=cell_factory)

dt_mean = np.mean(dttrain)
model = MIMO(k_in, k_out, paras.k_state, dropout=paras.dropout, cell_factory=cell_factory,
                         meandt=dt_mean, train_scheme=paras.scheme, eval_scheme=paras.scheme,
                         gamma=paras.gamma, step_size=paras.step_size, interpol='constant')



if GPU:
    model = model.cuda()

params = sum([np.prod(p.size()) for p in model.parameters()])
logging('\nModel %s (time_aware: %s, scheme %s) with %d trainable parameters' % (paras.model, paras.time_aware, paras.scheme, params))
for n, p in model.named_parameters():
    p_params = np.prod(p.size())
    print('\t%s\t%d (cuda: %s)'%(n, p_params, str(p.is_cuda)))

logging('Architecture: ', model)
log_value('model/params', params, 0)

optimizer = torch.optim.Adam(model.parameters(), lr=paras.lr, weight_decay=paras.l2)


# prepare tensors for evaluation
Xtrain_tn = torch.tensor(Xtrain, dtype=torch.float).unsqueeze(0)  # (1, Ntrain, k_in)
Ytrain_tn = torch.tensor(Ytrain, dtype=torch.float).unsqueeze(0)  # (1, Ntrain, k_out)
dttrain_tn = torch.tensor(dttrain, dtype=torch.float).unsqueeze(0)  # (1, Ntrain, 1)
Xdev_tn = torch.tensor(Xdev, dtype=torch.float).unsqueeze(0)  # (1, Ndev, k_in)
Ydev_tn = torch.tensor(Ydev, dtype=torch.float).unsqueeze(0)  # (1, Ndev, k_out)
dtdev_tn = torch.tensor(dtdev, dtype=torch.float).unsqueeze(0)  # (1, Ndev, 1)
Xtest_tn = torch.tensor(Xtest, dtype=torch.float).unsqueeze(0)
Ytest_tn = torch.tensor(Ytest, dtype=torch.float).unsqueeze(0)
dttest_tn = torch.tensor(dttest, dtype=torch.float).unsqueeze(0)

if GPU:
    Xtrain_tn = Xtrain_tn.cuda()
    Ytrain_tn = Ytrain_tn.cuda()
    dttrain_tn = dttrain_tn.cuda()
    Xdev_tn = Xdev_tn.cuda()
    Ydev_tn = Ydev_tn.cuda()
    dtdev_tn = dtdev_tn.cuda()
    Xtest_tn = Xtest_tn.cuda()
    Ytest_tn = Ytest_tn.cuda()
    dttest_tn = dttest_tn.cuda()


def t2np(tensor):
    return tensor.squeeze().detach().cpu().numpy()

trainer = EpochTrainer(model, optimizer, paras.epochs, Xtrain, Ytrain, dttrain,
                       batch_size=paras.batch_size, gpu=GPU, bptt=paras.bptt)  #dttrain ignored for all but 'variable' methods

t00 = time()

best_dev_error = 1.e5
best_dev_epoch = 0
error_test = -1

max_epochs_no_decrease = 1000

try:  # catch error and redirect to logger

    for epoch in range(1, paras.epochs + 1):

        #train 1 epoch
        mse_train = trainer(epoch)

        if epoch % paras.eval_epochs == 0:
            with torch.no_grad():

                model.eval()
                # (1) forecast on train data steps
                Ytrain_pred, htrain_pred = model(Xtrain_tn, dt=dttrain_tn)
                error_train = prediction_error(Ytrain, t2np(Ytrain_pred))

                # (2) forecast on dev data
                Ydev_pred, hdev_pred = model(Xdev_tn, state0=htrain_pred[:, -1, :], dt=dtdev_tn)
                mse_dev = model.criterion(Ydev_pred, Ydev_tn).item()
                error_dev = prediction_error(Ydev, t2np(Ydev_pred))

                # report evaluation results
                log_value('train/mse', mse_train, epoch)
                log_value('train/error', error_train, epoch)
                log_value('dev/loss', mse_dev, epoch)
                log_value('dev/error', error_dev, epoch)

                logging('epoch %04d | loss %.3f (train), %.3f (dev) | error %.3f (train), %.3f (dev) | tt %.2fmin'%
                        (epoch, mse_train, mse_dev, error_train, error_dev, (time()-t00)/60.))
                show_data(ttrain, Ytrain, t2np(Ytrain_pred), paras.save, 'current_trainresults',
                               msg='train results (train error %.3f) at iter %d' % (error_train, epoch))
                show_data(tdev, Ydev, t2np(Ydev_pred), paras.save, 'current_devresults',
                               msg='dev results (dev error %.3f) at iter %d' % (error_dev, epoch))

                # update best dev model
                if error_dev < best_dev_error:
                    best_dev_error = error_dev
                    best_dev_epoch = epoch
                    log_value('dev/best_error', best_dev_error, epoch)

                    #corresponding test result:
                    Ytest_pred, _ = model(Xtest_tn, state0=hdev_pred[:, -1, :], dt=dttest_tn)
                    error_test = prediction_error(Ytest, t2np(Ytest_pred))
                    log_value('test/corresp_error', error_test, epoch)
                    logging('new best dev error %.3f'%best_dev_error)

                    # make figure of best model on train, dev and test set for debugging
                    show_data(tdev, Ydev, t2np(Ydev_pred), paras.save, 'best_dev_devresults',
                              msg='dev results (dev error %.3f) at iter %d' % (error_dev, epoch))
                    show_data(ttest, Ytest, t2np(Ytest_pred), paras.save,
                              'best_dev_testresults',
                              msg='test results (test error %.3f) at iter %d (=best dev)' % (error_test, epoch))

                    # save model
                    #torch.save(model.state_dict(), os.path.join(paras.save, 'best_dev_model_state_dict.pt'))
                    torch.save(model, os.path.join(paras.save, 'best_dev_model.pt'))

                    # save dev and test predictions of best dev model
                    pickle.dump({'t_dev': tdev, 'y_target_dev': Ydev, 'y_pred_dev': t2np(Ydev_pred),
                                 't_test': ttest, 'y_target_test': Ytest, 'y_pred_test': t2np(Ytest_pred)},
                                open(os.path.join(paras.save, 'data4figs.pkl'), 'wb'))

                elif epoch - best_dev_epoch > max_epochs_no_decrease:
                    logging('Development error did not decrease over %d epochs -- quitting.'%max_epochs_no_decrease)
                    break


    log_value('finished/best_dev_error', best_dev_error, 0)
    log_value('finished/corresp_test_error', error_test, 0)



    logging('Finished: best dev error', best_dev_error,
              'at epoch', best_dev_epoch,
              'with corresp. test error', error_test)


except:
    var = traceback.format_exc()
    logging(var)
