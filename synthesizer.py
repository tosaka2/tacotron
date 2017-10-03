import io
import numpy as np
import chainer
from hparams import hparams
from models import create_model
from util import audio, textinput


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    self.model = create_model(model_name, hparams)
  
    print('Loading checkpoint: %s' % checkpoint_path)
    chainer.serializers.load_npz(checkpoint_path, self.model)
  
  def synthesize(self, text):
    with chainer.using_config('train', False):
      seq = textinput.to_sequence(
        text, force_lowercase=hparams.force_lowercase,
        expand_abbreviations=hparams.expand_abbreviations)

      spec = self.model.output(seq)
      out = io.BytesIO()
      audio.save_wav(audio.inv_spectrogram(spec.T), out)
      return out.getvalue()
    