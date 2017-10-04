import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L

import numpy as np
import numpy.random as random

from hparams import hparams as hp

class PreNet(chainer.Chain):
  def __init__(self):
    super(PreNet, self).__init__()
    with self.init_scope():
      self.fc1 = L.Linear(None, 256)
      self.fc2 = L.Linear(None, 128)

  def __call__(self, e):
    act = F.relu
    e = F.dropout(act(self.fc1(e)))
    e = F.dropout(act(self.fc2(e)))
    return e

class CBHG(chainer.Chain):
  def __init__(self, in_size, bank_k, proj_filters1, proj_filters2):
    super(CBHG, self).__init__()
    with self.init_scope():
      self.conv1d_banks = [Conv1DwithBatchNorm(in_size, 128, i + 1) for i in range(bank_k)]
      self.conv1d_proj1 = Conv1DwithBatchNorm(128, proj_filters1, 3)
      self.conv1d_proj2 = Conv1DwithBatchNorm(proj_filters1, proj_filters2, 3)
      self.highways = [L.Highway(proj_filters2) for i in range(4)] # The parameters of the original paper are probably wrong.
      self.gru = L.NStepBiGRU(1, proj_filters2, 128, dropout=0)


  def __call__(self, e):
    act = F.relu
    inputs = e
    # Convolution bank: concatenate on the last axis to stack channels from all convolutions
    e = F.concat(np.array([act(conv1d(e)) for conv1d in self.conv1d_banks])) # uncertain

    # Maxpooling
    e = max_pooling1d(e, 2, stride=1) #uncertain ksize

    # Two projection layers:
    e = act(self.conv1d_proj1(e))
    e = act(self.conv1d_proj2(e))

    # Residual connection:
    e = e + inputs

    # 4-layer HighwayNet:
    for highway in self.highways:
      e = highway(e) # uncertain
    
    # Bidirectional RNN
    _, e = self.gru(None, e)

    return e

def get_encoder_cbhg():
  return CBHG(128, 16, 128, 128)

def get_decoder_cbhg():
  return CBHG(hp.num_mels, 16, 256, 128)

# which is better, max_pooling_nd or max_pooling2d
def max_pooling1d(input, kernel_size, stride=1, pad=0):
  return F.max_pooling_nd(input, ksize=(1,kernel_size), stride=(1,stride), pad=pad)

'''
def max_pooling1d(input, in_channels, kernel_size, stride=1, pad=0):
  return F.max_pooling_2d(input, (in_channels, kernel_size), (in_channels, stride), (0, pad))
'''

# Convoution1D->BatchNormalization
class Conv1DwithBatchNorm(chainer.Chain):
  def __init__(self, in_channels, out_channels, kernal_size):
    super(Conv1DwithBatchNorm, self).__init__()
    with self.init_scope():
      # Conv1d by Conv2d
      # uncertain
      self.conv1d = L.Convolution2D(1, out_channels, (in_channels, kernal_size))
      self.batch_norm = L.BatchNormalization(out_channels)

  def __call__(self, e):
    e = self.conv1d(e)
    e = self.batch_norm(e)
    return e


class Attention(chainer.Chain):

    """
    Attention module https://arxiv.org/abs/1409.0473
    """

    def __init__(self, hidden_size):
      super(Attention, self).__init__()
      with self.init_scope():
        self.w1 = L.Linear(hidden_size, hidden_size)
        self.w2 = L.Linear(hidden_size, hidden_size)
        self.v = Variable(random.randn(hidden_size))
        self.hidden_size = hidden_size

    def __call__(self, encoder_output, decoder_output):
      batch_size = encoder_output.data.shape[0]
      exp_us = []
      sum_exp = Variable(np.zeros((batch_size, 1), dtype='float32'))
      
      d = decoder_output
      # uncertain
      # TODO: change to batch_matmul
      for h in encoder_output:
        u = F.matmul(self.v, F.tanh(self.w1(h) + self.w2(d)), transb=True)
        exp_u = F.exp(u)
        
        exp_us.append(exp_u)
        sum_exp += exp_u
        
      att = Variable(np.zeros((batch_size, self.hidden_size), dtype='float32'))

      for exp_u, h in zip(exp_us, encoder_output):
        a = exp_u / sum_exp
        att += F.reshape(F.batch_matmul(h, a), (batch_size, self.hidden_size))

      return att