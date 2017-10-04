import argparse
from datetime import datetime
import math
import numpy as np
import os
import subprocess
import time
import traceback

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from datasets.preprocessed_dataset import PreprocessedDataset
import hparams
from hparams import hparams, hparams_debug_string, parse_hparams
from models import create_model
from util import audio, infolog, plot, textinput, ValueWindow
log = infolog.log


def get_git_commit():
  subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
  log('Git commit: %s' % commit)
  return commit


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
  commit = get_git_commit() if args.git else 'None'
  input_path = os.path.join(args.base_dir, args.input)
  log('Loading training data from: %s' % input_path)
  log('Using model: %s' % args.model)
  log('GPU: {}'.format(args.gpu))
  log('# epoch: {}'.format(args.epoch))
  log(hparams_debug_string())

  # Set up model:
  model = create_model('tacotron', hparams)

  # Setup an optimizer
  optimizer = chainer.optimizers.Adam()
  optimizer.setup(model)


  # Set up dataset
  train = PreprocessedDataset(input_path, hparams)
  # train_iter = chainer.iterators.MultiprocessIterator(train, hparams.batch_size) #TODO:align input sizes (shuffle=False)
  train_iter = chainer.iterators.SerialIterator(train, hparams.batch_size) #TODO:align input sizes (shuffle=False)
  
  # Set up a trainer
  # TODO: change to ParallelUpdater
  updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
  trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

  
  trainer.extend(extensions.LogReport(log_name="log", trigger=(args.summary_interval, 'iteration')))
  trainer.extend(extensions.snapshot(), trigger=(args.checkpoint_interval,'iteration'))
  # trainer.extend(extensions.snapshot_object(optimizer,  'snapshot_{.updater.iteration}', trigger=(args.checkpoint_interval,'iteration')))
  
  trainer.extend(extensions.dump_graph('main/loss'))

  trainer.extend(extensions.LogReport())
  
  # Save two plot images to the result dir
  if extensions.PlotReport.available():
    trainer.extend(
      extensions.PlotReport(['main/loss', 'validation/main/loss'],
        'iteration', file_name='loss.png'))
    trainer.extend(
      extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'],
        'iteration', file_name='accuracy.png'))

  # Print selected entries of the log to stdout
  # Here "main" refers to the target link of the "main" optimizer again, and
  # "validation" refers to the default name of the Evaluator extension.
  # Entries other than 'epoch' are reported by the Classifier link, called by
  # either the updater or the evaluator.
  trainer.extend(extensions.PrintReport(
      ['epoch', 'main/loss', 'validation/main/loss',
       'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
  # Print a progress bar to stdout
  trainer.extend(extensions.ProgressBar())

  if args.resume:
    # Resume from a snapshot
    chainer.serializers.load_npz(args.resume, trainer)
    log('Resuming from checkpoint: %s at commit: %s' % (args.resume, commit), slack=True)
  
  # snapshot spectrogram # uncertain
  @training.make_extension(trigger=(args.checkpoint_interval, 'iteration'))
  def save_audio():
    # model instance has spectrogram data which was processed last
    spectrogram = model.spectrogram #TODO: change this specification
    waveform = audio.inv_spectrogram(spectrogram.T)
    audio.save_wav(waveform, os.path.join(log_dir, 'iteration_{.updater.iteration}-audio.wav'.format(trainer)))
    plot.plot_alignment(alignment, os.path.join(log_dir, 'iteration_{.updater.iteration}-align.png'.format(trainer)),
              info='%s, %s, %s, iteration_{.updater.iteration}, loss=%.5f'.format(args.model, commit, time_string(), trainer, loss))
    log('Input: %s' % textinput.to_string(input_seq))

  trainer.extend(save_audio)

  # TODO: more extensions to save mel loss and linear loss
  # TODO: send results to slack on a regular basis (No need?)
  # TODO: monitor the divergence of loss (No need?)

  # Train!
  trainer.run()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron'))
  parser.add_argument('--input', default='training/train.txt') # metadata
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
  parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
  # 追加
  parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of sweeps over the dataset to train')
  parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
  parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
  
  args = parser.parse_args()
  
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  parse_hparams(args.hparams)
  train(log_dir, args)


if __name__ == "__main__":
  main()
