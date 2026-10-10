
import argparse
from src.utils.utils import *
import logging
import torch
from src.data.feature_utils import *
from src.evaluation.evaluation_utils import *
from src.training.train import *
from src.utils.data_utils import get_dataloaders
from src.models.models import EncodersExecutorConfig
from src.utils.model_utils import get_model_output_dir, PretrainedModelsCollection
from tqdm.contrib.logging import logging_redirect_tqdm

def main(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    if not args.load_pretrained_from_file:

        configs:list[EncodersExecutorConfig] = [EncodersExecutorConfig().from_json_file(fn) for fn in args.config_files]
        
        if args.config_files_dir or not args.do_not_sort_configs:
            logger.info('sorting the configs')
            EncodersExecutorConfig.sort_strategies(configs)


    for i in range(len(configs)):
        # set seed for the config to be used, do NOT set this again for the tries
        if args.seed:
            set_seeds(args.seed)
            logger.info(f'setting execution seed to {args.seed}')

        if not args.load_pretrained_from_file:
            config = configs[i]
 


        for n_try in range(1):
            # do not set seeds here to get some variability

            
            # get encoderExecutor
            if not args.load_pretrained_from_file:
                encodersExecutor = EncodersExecutor(config)
                encodersExecutor.to(args.device)
            

            total_params = sum(p.numel() for p in encodersExecutor.parameters())
            print(f'{config.strategy_name}')
            print(f'Total number of parameters: {total_params}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=None, 
                        help="seed to be used for the runs")
    
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='name to give to the experiment, should be "path safe"')
    parser.add_argument('--experiment_description', type=str, default=None,
                        help='textual description of the experiment')
    
    parser.add_argument('--feature_dir', type=str, required=True, 
                        help="directory for the the features (just the dir name, not the path)")
    parser.add_argument('--feature_set_name', type=str, required=True, 
                        help="name for the feature set file (not the path)")

    parser.add_argument('--data_augmentation_fn', type=str, required=False, default=None,
                        help="name of the filename (without path) of the data augmentation config to be used, if any")
    
    parser.add_argument('--wandb_project_name',type=str, default=None,
                        help='to use weights and biases set this')
    
    parser.add_argument('--hf_user_name',type=str, required=True, # TODO: change this to allow/manage local changes too
                        help='to use huggingface set this')
    parser.add_argument('--hf_tags',nargs='+', default=None,
                        help='useful to add tags to the current experiment runs')

    parser.add_argument('--device', type=str, default=None,
                        help="the device to be used, default will be 'cuda' if avaliable otherwise 'cpu'")

    parser.add_argument('--use_high_precision_uid', action='store_true', help='wether to use higher precision for the uid (needed only if the generation has an high frequency, e.g. immediately subsequent calls)')

    parser.add_argument('--train', action='store_true',help='wether to train the model')

    parser.add_argument('--load_pretrained_from_file', type=str, default=None,help='wether to load pretrained models form file, in this case train flag should not be set')

    # these args are meaningful only when training
    parser.add_argument('--config_files', nargs='+', help='list of strategy config file names to be used to load the models, pass the names without the directory', default=None)
    parser.add_argument('--config_files_dir', type=str, help='directory for the strategy config files, if config_files is empty all the config files in the directory will be used', default=None)
    parser.add_argument('--do_not_sort_configs', action='store_false', help='wether to prevent sorting the configuration files when using the list of config files')

    parser.add_argument('--batch_size', type=int, default=None)
    
    parser.add_argument('--num_workers', type=int, default=2) # this should be 0 for better reproducibility, possibly at some small performance cost
    
    parser.add_argument('--epochs', type=int, default=None)
    
    parser.add_argument('--lr', type=float, default=None)
    
    parser.add_argument('--number_of_tries', type=int, default=1,
                        help="number of times to repeat each experiment to take the average values")
    
    parser.add_argument('--loss_patience', type=int, default=25)
    parser.add_argument('--loss_delta', type=float, default=0.0001)

    parser.add_argument('--loss_margin', type=float, default=0.25)
    parser.add_argument('--loss_within_category_margin', type=float, default=None,
                        help='margin to be used for the elements within the same category, this can be used only when the flag use_categories_in_loss is set')
    
    parser.add_argument('--new_positive_samples', type=int, default=None,
                        help='number of new elements to be sampled from the same class of the anchor as new positives, in this case the margin is the same as the loss_margin')
    parser.add_argument('--allow_sampling_replacement_of_new_positives', action='store_true', help='whether to allow replacement when sampling the new positive within the same class, this is useful to deal with the case in which a given class has less than new_positive_samples-1 different members within a batch')
    
    parser.add_argument('--use_categories_in_loss', action='store_true', help='whether to use the categories information in the loss if available')
    parser.add_argument('--filter_out_hard_negatives', action='store_true', help='whether to filter out the hard negatives, keeping only the semi-hard ones')
    
    parser.add_argument('--second_loss_component_weight', type=float, default=None,
                        help='weight for the second component of the loss')
    parser.add_argument('--second_loss_component_weight_decay_factor', type=float, default=None,
                        help='decaying factor for the second loss component')
    

    parser.add_argument('--scheduler', type=str, default='step_lr',
                        choices=['step_lr'], 
                        help="scheduler to be used")
    parser.add_argument('--scheduler_step_size', type=int, default=27)
    parser.add_argument('--scheduler_gamma', type=float, default=0.75)
    
    parser.add_argument('--save_strategy', type=str, default='best',
                        choices=['best', 'last'], 
                        help="saving strategy for the model and config")
    parser.add_argument('--best_strategy', type=str, default='t2s-r1',
                        choices=BestModelManager.get_best_model_strategies(), 
                        help="strategy to be used for choosing the best model")
    

    args = parser.parse_args()

    if args.train:
        assert not args.load_pretrained_from_file, f'train preloaded models is not allowed'
        assert args.batch_size, f'batch_size is required when train flag is set'
        assert args.epochs, f'epochs is required when train flag is set'
        assert args.lr, f'lr is required when train flag is set'
        assert args.best_strategy, f'best_strategy is required when train flag is set'

        assert not args.second_loss_component_weight or (args.use_categories_in_loss or args.new_positive_samples), f'to use the second weight component for the loss there should be a second component'
        assert not args.new_positive_samples or args.use_categories_in_loss, f"to use the new positive samples you need to set the use_categories_in_loss flag too"

        if not args.use_categories_in_loss:
            assert args.loss_within_category_margin is None, f'when use_categories_in_loss flag is not set loss_within_category_margin is not used, you may want to act on loss_margin instead'

        if args.second_loss_component_weight:
            assert args.second_loss_component_weight>0 and args.second_loss_component_weight<1, f'second_loss_component_weight must be in (0,1), given: {args.second_loss_component_weight}' 
            if args.second_loss_component_weight_decay_factor:
                assert args.second_loss_component_weight_decay_factor>0 , f"second_loss_component_weight_decay_factor must be positive: {args.second_loss_component_weight_decay_factor}"

        assert not args.filter_out_hard_negatives or not args.use_categories_in_loss, f'to be rebalanced'
    else:
        assert args.data_augmentation_fn is None, f'data augmentation is not used when not training, please remove it to avoid confusion'

    # checks on args
    assert args.load_pretrained_from_file or ((args.config_files or args.config_files_dir) and not (args.config_files and args.config_files_dir)), "exactly one between config_files and config_files_dir has to be set when not loading a pretrained model"
    
    if args.config_files_dir:
        # load config file names in the config_files attribute
        args.config_files = []
        assert os.path.isdir(args.config_files_dir)
        for fn in os.listdir(args.config_files_dir):
            args.config_files.append(os.path.join(args.config_files_dir, fn))
    elif args.config_files:
        for fn in args.config_files:
            assert os.path.exists(fn) and os.path.isfile(fn), f"errors for configuration file {fn}"


    main(args)



# python main_count_params.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --seed 424242 --wandb_project_name bbb --hf_user_name aaa --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --use_categories_in_loss --loss_within_category_margin 0.15 --second_loss_component_weight 0.05 --hf_tags "new 18th 2nd w 0.05" --experiment_name "CN_TC_2nd_w_0.05" --experiment_description "experiemnt with 2 loss components inter and intra class, margins 0.25 and 0.15, varing the weight of the two components"
