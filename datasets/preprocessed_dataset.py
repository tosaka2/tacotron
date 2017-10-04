import numpy as np
import os
import random
import threading
import time
import traceback
from util import cmudict, textinput
from util.infolog import log
import chainer

_batches_per_group = 32
_p_cmudict = 0.5
_pad = 0

# https://github.com/chainer/chainer/blob/1ad6355f8bfe4ccfcf0efcfdb5bd048787069806/examples/imagenet/train_imagenet.py
class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, metadata_filename, hparams):
        self._hparams = hparams

        # Load metadata:
        self._datadir = os.path.dirname(metadata_filename)
        # with open(metadata_filename) as f:
        with open(metadata_filename, encoding="utf-8_sig") as f:
            self._metadata = [line.strip().split('|') for line in f]
            hours = sum((int(x[2]) for x in self._metadata)) * \
                hparams.frame_shift_ms / (3600 * 1000)
            log('Loaded metadata for %d examples (%.2f hours)' %
                (len(self._metadata), hours))

        # Load CMUDict: If enabled, this will randomly substitute some words in the training data with
        # their ARPABet equivalents, which will allow you to also pass ARPABet to the model for
        # synthesis (useful for proper nouns, etc.)
        if hparams.use_cmudict:
            cmudict_path = os.path.join(self._datadir, 'cmudict-0.7b')
            if not os.path.isfile(cmudict_path):
                raise Exception('If use_cmudict=True, you must download ' +
                                'http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b to %s' % cmudict_path)
            self._cmudict = cmudict.CMUDict(cmudict_path, keep_ambiguous=False)
            log('Loaded CMUDict with %d unambiguous entries' %
                len(self._cmudict))
        else:
            self._cmudict = None


    def _get_next_example(self, offset):
        '''Loads a single example (input, mel_target, linear_target, cost) from disk'''
        meta = self._metadata[offset]

        text = meta[3]
        if self._cmudict and random.random() < _p_cmudict:
            text = ' '.join([self._maybe_get_arpabet(word)
                             for word in text.split(' ')])

        input_data = np.asarray(textinput.to_sequence(text), dtype=np.int32)
        linear_target = np.load(os.path.join(self._datadir, meta[0]))
        mel_target = np.load(os.path.join(self._datadir, meta[1]))
        return (input_data, mel_target, linear_target, len(linear_target))

    # curriculum learning?
    def _maybe_get_arpabet(self, word):
        pron = self._cmudict.lookup(word)
        return '{%s}' % pron[0] if pron is not None and random.random() < 0.5 else word

    def _prepare_batch(batch, outputs_per_step):
        random.shuffle(batch)
        inputs = _prepare_inputs([x[0] for x in batch])
        input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
        mel_targets = _prepare_targets([x[1] for x in batch], outputs_per_step)
        linear_targets = _prepare_targets(
            [x[2] for x in batch], outputs_per_step)
        return (inputs, input_lengths, mel_targets, linear_targets)

    def _prepare_inputs(inputs):
        max_len = max((len(x) for x in inputs))
        return np.stack([_pad_input(x, max_len) for x in inputs])

    def _prepare_targets(targets, alignment):
        max_len = max((len(t) for t in targets)) + 1
        return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])

    def _pad_input(x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

    def _pad_target(t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=_pad)

    def _round_up(x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    # implimentation of DatasetMixin?
    def __len__(self):
        return len(self._metadata)

    # implimentation of DatasetMixin
    def get_example(self, i):
        input, mel, lin, _ = self._get_next_example(i)
        return input, (mel, lin)