
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

    # set seed first time, then set it again before each training procedure   
    if args.seed:
        set_seeds(args.seed)
        logger.info(f'setting execution seed to {args.seed}')
    
    use_wandb = args.wandb_project_name is not None
    logger.info(f'use wandb: {use_wandb}')

    if not args.device:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'using device: {args.device}')

    feature_set = DigitalExhibitionFeatureSet.loadFeatureSet(args.feature_dir, args.feature_set_name)
    dataset_name = feature_set.get_dataset_name()
    logger.info(f'loaded feature set {args.feature_set_name} form {args.feature_dir} ({dataset_name})')
    desc_feature_size, scene_feature_size, scene_image_art_feature_size, scene_video_art_feature_size = feature_set.get_feature_sizes() # used to check assertions

    exhibitionDataset = None
    if dataset_name == 'digitalExps':
        #todo implement this
        from src.data.datasets.artExhibitionDataset.utils import ArtExhibitionDataset
        exhibitionDataset = ArtExhibitionDataset(skip_description_and_artworks_loading=True)
    elif dataset_name == 'generatedDigitalExps':
        from src.data.datasets.digitalMuseumsSemArtVideoGenDataset.utils import DigitalMuseumsSemArtVideoGenDataset
        exhibitionDataset = DigitalMuseumsSemArtVideoGenDataset(skip_description_and_artworks_loading=True)
    elif dataset_name == 'museums3k':
        from src.data.datasets.museums3k.utils import Museums3kDataset
        exhibitionDataset = Museums3kDataset()
    else:
        assert False, f"unimplemented dataset choice: {dataset_name}"
    logger.info(f'exhibitions dataset loaded')

    if not args.load_pretrained_from_file:

        configs:list[EncodersExecutorConfig] = [EncodersExecutorConfig().from_json_file(fn) for fn in args.config_files]
        
        if args.config_files_dir or not args.do_not_sort_configs:
            logger.info('sorting the configs')
            EncodersExecutorConfig.sort_strategies(configs)
        iter_num = len(configs) 

    else:
        pretrainedCollection = PretrainedModelsCollection.load_pretrained_collection(args.load_pretrained_from_file)
        iter_num = pretrainedCollection.get_models_num()
    

    last_run_uses_flattened_hierarchy = None
    dataloaders = None

    experiments_uid = get_time_UID(high_precision=False)

    experiment_dir = os.path.join('outputs', 'experiments')
    os.makedirs(experiment_dir, exist_ok=True)
    experiment_name =  f'{experiments_uid}{("_" + args.experiment_name.replace(" ", "_")) if args.experiment_name else ""}'
    experiment_info = {
        'name': experiment_name,
        'uid':experiments_uid,
        'description':args.experiment_description,
        'runs': []
    }

    for i in range(iter_num):
        # set seed for the config to be used, do NOT set this again for the tries
        if args.seed:
            set_seeds(args.seed)
            logger.info(f'setting execution seed to {args.seed}')

        if not args.load_pretrained_from_file:
            config = configs[i]
        else:
            _, config = pretrainedCollection.load_ith_model(i)

        # sanity check on the input size
        config_desc_feature_size, config_scene_feature_size, config_scene_image_art_feature_size, config_scene_video_art_feature_size = config.get_input_feature_sizes()
        assert (config_desc_feature_size == desc_feature_size) if config_desc_feature_size else True, f"desc_feature_size does not match the input size of the model ({desc_feature_size} vs {config_desc_feature_size})" 
        assert (config_scene_feature_size == scene_feature_size) if config_scene_feature_size else True, f"scene_feature_size does not match the input size of the model ({scene_feature_size} vs {config_scene_feature_size})" 
        assert (config_scene_image_art_feature_size == scene_image_art_feature_size) if config_scene_image_art_feature_size else True, f"scene_image_art_feature_size does not match the input size of the model ({scene_image_art_feature_size} vs {config_scene_image_art_feature_size})" 
        assert (config_scene_video_art_feature_size == scene_video_art_feature_size) if config_scene_video_art_feature_size else True, f"scene_video_art_feature_size does not match the input size of the model ({scene_video_art_feature_size} vs {config_scene_video_art_feature_size})" 

        output_feature_size = config.get_output_size()

        ee_strategy_name = config.get_full_strategy_name()
        flatten_hierarchy = not config.is_hierarchical()
        ee_strategy_args = config.get_ee_args()
        #uses_audio_video = config.uses_audio_video()

        run_uid = get_time_UID(high_precision=args.use_high_precision_uid)
        if args.train:
            run_name = f"{run_uid}_{ee_strategy_name}"
        else:
            original_uid = pretrainedCollection.get_ith_uid(i)
            run_name = f"{run_uid}_{ee_strategy_name}_test_of_{original_uid}_on_{dataset_name}"

        logger.info(f'=============== RUN #{i}/{iter_num}: {run_name} ===============')

        # recreate datasets only if they differ in the hierarchy flatten mode, otherwise they will be the same
        if last_run_uses_flattened_hierarchy is None or (last_run_uses_flattened_hierarchy != flatten_hierarchy):
            last_run_uses_flattened_hierarchy = flatten_hierarchy
            logger.info('loading datasets and dataloaders')
            dataloaders = get_dataloaders(args, flatten_hierarchy, exhibitionDataset)
        else:
            logger.info('using previous dataloaders as they have not changed between base and hierarchical')

        save_output_data = []
        
        run_info = {
            'base_name':run_name,
            'n_tries':args.number_of_tries,
            'dataset_name':dataset_name,
            'hf_username':args.hf_user_name
        }
        if use_wandb:
            run_info['wandb_project_name']=args.wandb_project_name
        
        experiment_info['runs'].append(run_info)

        for n_try in range(args.number_of_tries):
            # do not set seeds here to get some variability

            full_run_name = f"{dataset_name}_{run_name}_{n_try}"
            logger.info(f"### {run_name}: try {n_try+1}/{args.number_of_tries} ###")

            # get encoderExecutor
            if not args.load_pretrained_from_file:
                encodersExecutor = EncodersExecutor(config)
                encodersExecutor.to(args.device)
            else:
                encodersExecutor, _ = pretrainedCollection.load_ith_model(i)
                encodersExecutor.to(args.device)

            if use_wandb:
                logged_config = {**{
                        "experiment_uid": experiments_uid,
                        "run":run_name, # useful to aggregate by run, alternative to the mere uid
                        "run_uid":run_uid, 
                        "approach_name": ee_strategy_name, # useful to aggregate by approach
                        "batch_size": args.batch_size,
                        "learning_rate": args.lr,
                        "epochs": args.epochs,
                        "feature_set_name":args.feature_set_name,
                        "feature_set_dir":args.feature_dir,
                        "config_files":args.config_files,
                        "save_Strategy":args.save_strategy,
                        "best_strategy":args.best_strategy,
                        "is_test_only":not args.train,
                        "approachSpecifics/uses_flattened_hierarchy":flatten_hierarchy,
                        #"approachSpecifics/uses_audio_video_features":uses_audio_video,
                        "use_categories_in_loss":args.use_categories_in_loss,
                        "loss_within_category_margin":args.loss_within_category_margin,
                        "new_positive_samples":args.new_positive_samples,
                        "second_loss_component_weight":args.second_loss_component_weight

                    },
                    **{f"architecture/{n}": v for n, v in encodersExecutor.getModelClassesDict().items()},
                    **{f"approachSpecifics/{n}": v for n, v in ee_strategy_args.items()}
                }



                wandb.init(
                    # set the wandb project where this run will be logged
                    project=args.wandb_project_name,
                    name=f"{run_name}_{n_try}",
                    # track hyperparameters and run metadata
                    config=logged_config,
                    tags = args.hf_tags
                )
                wandb.run.log_code(".")

            if args.train: # possibly true only when not loading a pretrained model
                # setup model params/gradients logging
                wandb.watch(encodersExecutor, log='all', log_freq=5, log_graph=True)

                # setup manager to track best model for the run (new for each run)
                bmm = BestModelManager(args.best_strategy)

                # train the model
                train(full_run_name, encodersExecutor, bmm, dataloaders, args, logger, use_wandb)

                # load back the best model (i.e. encoderExecutor)
                logger.info(f"loading back the best model")
                if args.hf_user_name:
                    encodersExecutor.from_pretrained(get_hf_full_name(args.hf_user_name, full_run_name), config=config)
            
            # test model

            output_description_test, output_pov_test, output_categories, _ = inference(encodersExecutor, None, output_feature_size, dataloaders['test'], args.batch_size, args.device, use_wandb, is_test=True, use_categories_in_loss=args.use_categories_in_loss)

            save_output_data_try = {}
            save_output_data_try['n_try'] = n_try
            save_output_data_try['output_description_test'] = output_description_test.tolist()
            save_output_data_try['output_pov_test'] = output_pov_test.tolist()
            save_output_data_try['output_categories'] = output_categories.tolist()
            save_output_data_try['ids_list'] = dataloaders['test'].dataset.split_ids # we know these are sorted

            evalRes = evaluate(
                output_description=output_description_test,
                output_scene=output_pov_test,
                scene_classes= output_categories,
                description_classes= output_categories)
            
            save_output_data_try['metrics'] = evalRes
            
            if use_wandb:
                wandb.log({
                    f"test/{k}":v
                    for k,v in evalRes.items()
                })
            
            
            save_output_data.append(save_output_data_try)

            if use_wandb:
                wandb.finish()

            output_dir = get_model_output_dir(full_run_name)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'test_output.json'), 'w') as f:
                json.dump(save_output_data, f, indent=4)
    
    with open(os.path.join(experiment_dir, f'{experiment_name}.json'), 'w') as f:
        json.dump(experiment_info, f, indent=4)

        


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

    with logging_redirect_tqdm():
        main(args)

