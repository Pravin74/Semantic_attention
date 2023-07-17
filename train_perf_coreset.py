from models_performer_attention_opt import build_model #for performer
#from models_performer_fast_attention import build_model ## for transformer with fast attention (useless as its already covered in performer)
# from models import build_model #for transformer
from dictionaries import IndexDictionary
from optimizers import NoamOptimizer
from trainer_perf_coreset_PCA import EpochSeq2SeqTrainer
# from trainer_perf_TCL import EpochSeq2SeqTrainer
from utils.log import get_logger
from utils.pipe import input_target_collate_fn

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from argparse import ArgumentParser
from datetime import datetime
import json
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # to run on CPU

parser = ArgumentParser(description='Train Transformer')
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--out_dir', type=str, default='/fs/cfar-projects/pravin/codes/output_SATFormer_summ_attend/')
parser.add_argument('--save_config', type=str, default=None)
parser.add_argument('--save_checkpoint', type=str, default=None)
parser.add_argument('--save_log', type=str, default=None)

parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

parser.add_argument('--dataset_limit', type=int, default=None)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--save_every', type=int, default=1)

parser.add_argument('--vocabulary_size', type=int, default=None)
parser.add_argument('--positional_encoding', action='store_true')

parser.add_argument('--d_input', type=int, default=2048)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--num_clusters', type=int, default=25)
parser.add_argument('--layers_count', type=int, default=6)
parser.add_argument('--heads_count', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=2048) # default is 256
parser.add_argument('--dropout_prob', type=float, default=0.1)

parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--optimizer', type=str, default="Noam", choices=["Noam", "Adam"])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--clip_grads', action='store_true')

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--sequence_length', type=int, default=20550)  # for performers_fast attention


def run_trainer(config):

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    run_name_format = (
        "d_model={d_model}-"
        "num_clusters={num_clusters}-"
        "layers_count={layers_count}-"
        "heads_count={heads_count}-"
        "pe={positional_encoding}-"
        "optimizer={optimizer}-"
        "{timestamp}"
    )

    run_name = run_name_format.format(**config, timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    logger = get_logger(run_name, save_log=config['save_log'])
    logger.info(f'Run name : {run_name}')
    logger.info(config)

    # logger.info('Constructing dictionaries...')
    # source_dictionary = IndexDictionary.load(config['data_dir'], mode='source', vocabulary_size=config['vocabulary_size'])
    # target_dictionary = IndexDictionary.load(config['data_dir'], mode='target', vocabulary_size=config['vocabulary_size'])
    # logger.info(f'Source dictionary vocabulary : {source_dictionary.vocabulary_size} tokens')
    # logger.info(f'Target dictionary vocabulary : {target_dictionary.vocabulary_size} tokens')

    logger.info('Building model...')
    model = build_model(config)

    logger.info(model)
    logger.info('Encoder : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.encoder.parameters()])))
    # logger.info('Decoder : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.decoder.parameters()])))
    # logger.info('Total : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.parameters()])))

    logger.info('Loading datasets...')

    loss_function = torch.nn.CrossEntropyLoss()

    # accuracy_function = AccuracyMetric()

    if config['optimizer'] == 'Noam':
        optimizer = NoamOptimizer(model.parameters(), d_model=config['d_model'])
    elif config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=config['lr'])
    else:
        raise NotImplementedError()

    logger.info('Start training...')
    trainer = EpochSeq2SeqTrainer(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        logger=logger,
        run_name=run_name,
        save_config=config['save_config'],
        save_checkpoint=config['save_checkpoint'],
        config=config
    )

    trainer.run(config['epochs'])

    return trainer


if __name__ == '__main__':

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)

        default_config = vars(args)
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
    else:
        config = vars(args)  # convert to dictionary

    run_trainer(config)
