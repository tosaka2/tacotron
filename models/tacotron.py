import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from util import textinput
from util.infolog import log
from .helpers import TacoTestHelper, TacoTrainingHelper
from .rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper

import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L
import numpy as np

from .modules import get_encoder_cbhg, get_decoder_cbhg, PreNet, Attention
import hparams as hp

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Tacotron(chainer.Chain):

  def __init__(self, n_source_vocab, n_units):
    super(Tacotron, self).init()
    with self.init_scope():
      self.embed = L.EmbedID(n_source_vocab, n_units)
      self.encoder_prenet = PreNet()
      self.decoder_prenet = PreNet()
      self.encoder_cbhg = get_encoder_cbhg()
      self.decoder_cbhg = get_decoder_cbhg()
      # TODO: change to NStepGRU
      self.attention_rnn = L.GRU(hp.num_mels * hp.outputs_per_step, 256)
      self.attention = Attention(256)
      self.decoder_rnn1 = L.GRU(512, 256)
      self.decoder_rnn2 = L.GRU(256, 256)
      self.calc_mel = L.Linear(256, hp.num_mels * hp.outputs_per_step)

  # 入力から誤差を計算
  def __call__(self, input, t):
    mel_t, lin_t = t # uncertain
    
    # Embeddings
    embed_output = sequence_embed(self.embed, input) # [N, T_in, n_units]
    
    # Encoder
    encoder_prenet_output = self.encoder_prenet(embed_output) # [N, T_in, 128] # uncertain
    encoder_outputs = encoder_cbhg(encoder_prenet_output) # [N, T_in, 256]

    # Decoder
    e = Variable(np.zeros(mel_t.shape))
    out_mels = Variable(np.array([]))
    for i in range(mel_t.shape[1]):
      
      if i > 0:
        e = mel_t[:,i - 1] # uncertain

      prenet_output = self.decoder_prenet()
      _, rnn_output = self.attention_rnn(e) # [N, 256]
      # Attention
      context_vector = self.attention() # [N, 256]
      decoder_input = F.concat((rnn_output, context_vector)) # [N, 512]
      
      decoder_rnn1_output = self.decoder_rnn1(decoder_input) # [N, 256]
      decoder_rnn2_output = self.decoder_rnn2(decoder_rnn1_output) # [N, 256]
      decoder_output = decoder_rnn1_output + decoder_rnn2_output # [N, 256]

      mel = self.calc_mel(decoder_output) # [N, r * num_mels]
      out_mels = F.concat((out_mels, mel))
    
    out_lins = self.decoder_cbhg(out_mels)

    # L1 loss
    mel_loss = F.absolute_error(mel_t, out_mels)
    lin_loss = F.absolute_error(lin_t, out_lins)
    # TODO: calculate loss from griffin_lim

    return mel_loss + lin_loss
    
  spectrogram = Variable(np.array([]))

  # synthesize spectrogram
  def output(self, input, max_length=100):
    with chainer.no_backprop_mode():
      with chainer.using_config('train', False):
        # Embeddings
        embed_output = sequence_embed(self.embed, input) # [N, T_in, n_units]

        # Encoder
        encoder_prenet_output = self.encoder_prenet(embed_output) # [N, T_in, 128] # uncertain
        encoder_outputs = encoder_cbhg(encoder_prenet_output) # [N, T_in, 256]

        # Decoder
        e = Variable(np.zeros(mel_t.shape))
        out_mels = Variable(np.array([]))
        for i in range(max_length): #TODO: loop for output length (until A becomes 0 vector)

          if i > 0:
            e = mel_t[:,i - 1] # uncertain

          prenet_output = self.decoder_prenet()
          _, rnn_output = self.attention_rnn(h) # [N, 256]
          # Attention
          context_vector = self.attention() # [N, 256]
          decoder_input = F.concat((rnn_output, context_vector)) # [N, 512]

          decoder_rnn1_output = self.decoder_rnn1(decoder_input) # [N, 256]
          decoder_rnn2_output = self.decoder_rnn2(decoder_rnn1_output) # [N, 256]
          decoder_output = decoder_rnn1_output + decoder_rnn2_output # [N, 256]

          mel = self.calc_mel(decoder_output) # [N, r * num_mels]
          out_mels = F.concat((out_mels, mel))

        out_lins = self.decoder_cbhg(out_mels)
        self.spectrogram = out_lins # BAD
        return out_lins