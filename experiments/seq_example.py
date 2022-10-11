# @Time   : 2022/10/11
# @Author : Guy Koren
# @Email  : guy.koren@intel.com

"""
sequential recommendation example
========================
Here is the sample code for running sequential recommendation benchmarks using RecBole.
it is based on session_based_rec_example.py script
args.dataset can be one of: amazon-beauty,ml-1m,steam
"""

import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRec', help='Model for sequential rec.')
    parser.add_argument('--dataset', '-d', type=str, default='retailrocket-sess', help='Benchmarks for sequential rec.')
    parser.add_argument('--gpu_id', '-gpu',type=int,default=0,help='gpu id [starting from 0]')
    parser.add_argument('--validation', action='store_true', help='Whether evaluating on validation set (split from train set), otherwise on test set.')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = get_args()

    # configurations initialization
    # for all possible parameters with their default values see recbole/properties/overall.yaml
    # some of the configurations below were inspired by recbole/properties/dataset/sample.yaml
    config_dict = {
        ############################
        # Environment settings - see https://recbole.io/docs/user_guide/config/environment_settings.html
        # see default values in recbole/properties/overall.yaml
        # commenting out default values. to change, uncomment
        'gpu_id': args.gpu_id,
        'use_gpu': True,
        # 'seed': 2020,
        # 'state': 'INFO',
        # 'reproducibility': True,
        # 'data_path': 'dataset/',
        # 'checkpoint_dir': 'saved',
        # 'show_progress': True,
        # 'save_dataset': False,
        # 'dataset_save_path': None,
        # 'save_dataloaders': False,
        # 'dataloaders_save_path': None,
        # 'log_wandb': False,
        # 'wandb_project': 'recbole',
     
        ############################
        # Model Settings
        # model architecture parameters: see https://recbole.io/docs/user_guide/model/sequential/sasrec.html or relevant algorithm
        # the below setup for SASRec is taken from recbole/properties/model/SASRec.yaml
        # commenting out default values. to change, uncomment
        # 'n_layers': 2,
        # 'n_heads': 2,
        # 'hidden_size': 64,
        # 'inner_size': 256,
        # 'hidden_dropout_prob': 0.5,
        # 'attn_dropout_prob': 0.5,
        # 'hidden_act': 'gelu',
        # 'layer_norm_eps': 1e-12,
        # 'initializer_range': 0.02,
        # 'loss_type': 'CE',

        ############################
        # Data Settings - dataset basic information and preprocessing
        # see https://recbole.io/docs/user_guide/config/data_settings.html for details
        # see default values in recbole/properties/dataset/sample.yaml
        # in teh following, only settings that are relevant to sequential model are added:

        # Sequential Model Needed
        # 'ITEM_LIST_LENGTH_FIELD': 'item_length',
        # 'LIST_SUFFIX': '_list',
        # 'MAX_ITEM_LIST_LENGTH': 50,
        # 'POSITION_FIELD': 'position_id',

        # Selectively Loading
        # Selectively Loading
        # 'load_col': {'inter': ['user_id', 'item_id']},
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
        # 'unload_col': None,
        # 'unused_col': None,
        # 'additional_feat_suffix': None,
        
        # Filtering
        # 'rm_dup_inter': None,
        # 'val_interval': None,
        # 'filter_inter_by_user_or_item': True,
        'user_inter_num_interval': "[5,inf)",
        'item_inter_num_interval': "[5,inf)",

        # Preprocessing
        # 'alias_of_user_id': None,
        # 'alias_of_item_id': None,
        'alias_of_item_id': ['item_id_list'],
        # 'alias_of_entity_id': None,
        # 'alias_of_relation_id': None,
        # 'preload_weight': None,
        # 'normalize_field': None,
        # 'normalize_all': None,

        # Benchmark .inter
        'benchmark_filename': ['train', 'test'],


        ################################
        # Training settings - see https://recbole.io/docs/user_guide/config/training_settings.html
        # see default values in recbole/properties/overall.yaml
        # defaults are commented out
        # 'epochs': 300,
        'train_batch_size': 2048,
        # 'learner': 'adam',
        # 'learning_rate': 0.001,
        # 'neg_sampling': None,
        # 'eval_step': 1,
        # 'stopping_step': 10,
        'stopping_step': 50,
        # 'clip_grad_norm': None,
        # # clip_grad_norm:  {'max_norm': 5, 'norm_type': 2}
        # 'weight_decay': 0.0,
        # 'loss_decimal_place': 4,
        # 'require_pow': False,
    


        ################################
        # Evaluation settings - see https://recbole.io/docs/user_guide/config/evaluation_settings.html
        # see default values in recbole/properties/overall.yaml
        # defaults are commented out
        'eval_args':{
            'split': {'LS': 'valid_and_test'},      # Leave-one-out splitting with valid and test splits (in addition to train)
            'order': 'TO',      # time ordered
            'mode': 'pop100'},
        'repeatable': False,
        'metrics': ['Hit','NDCG'],
        'topk': [1,5,10],
        'valid_metric': 'Hit@5',
        # 'valid_metric_bigger': True,
        'eval_batch_size': 4096,
        # 'metric_decimal_place': 4       
    }

    config = Config(model=args.model, dataset=f'{args.dataset}', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_dataset, test_dataset = dataset.build()
    if args.validation:
        train_dataset.shuffle()
        new_train_dataset, new_test_dataset = train_dataset.split_by_ratio([1 - args.valid_portion, args.valid_portion])
        train_data = get_dataloader(config, 'train')(config, new_train_dataset, None, shuffle=True)
        test_data = get_dataloader(config, 'test')(config, new_test_dataset, None, shuffle=False)
    else:
        train_data = get_dataloader(config, 'train')(config, train_dataset, None, shuffle=True)
        test_data = get_dataloader(config, 'test')(config, test_dataset, None, shuffle=False)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training and evaluation
    test_score, test_result = trainer.fit(
        train_data, test_data, saved=True, show_progress=config['show_progress']
    )

    logger.info(set_color('test result', 'yellow') + f': {test_result}')
