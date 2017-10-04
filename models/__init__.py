from .tacotron import Tacotron


def create_model(name, hparams):
  if name == 'tacotron':
    return Tacotron(hparams.n_source_vocab, hparams.n_units)
  else:
    raise Exception('Unknown model: ' + name)
