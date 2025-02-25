import argparse
import os
from copy import deepcopy
from src.models.models import EncodersExecutorConfig

# knowledge for the model parameters comes from the src.models.models file

def generate_config_files(args):

    strategies = [] # these will be some dict templates to later on create the actual 
    
    useLSTMinPlaceOfGRU = not args.gru # use gru first if requested

    if args.b_cnn1d:

        s = {
                'name':'rnn_cnn1d',
                'ee_args':
                {
                    'useLSTM':useLSTMinPlaceOfGRU,
                    'desc_output_feature_size':args.output_feature_size,
                    'desc_input_feature_size':args.desc_feature_size,
                    'desc_model_is_bidirectional':args. desc_rnn_is_bidir,
                    'scene_input_channel':args.scene_feature_size,
                    'scene_output_channels':args.cnn1d_inter_size,
                    'kernel_size':args.kernel_size,
                    'scene_output_feature_size':args.output_feature_size ,
                },
                'uses_flattened_hierarchy':True,
            }

        if args.no_image_video:
            strategies.append(s)

        if args.with_image_video:
            s_c = deepcopy(s) # deep copy the dictionary to avoid issues with the edit or the args 
            s_c['name'] += '_image_video'
            s_c['ee_args']["image_art_in_channels"]=args.scene_image_art_feature_size
            s_c['ee_args']["video_art_in_channels"]=args.scene_video_art_feature_size
        
            strategies.append(s_c)
        
        if args.with_video:
            s_c = deepcopy(s) # deep copy the dictionary to avoid issues with the edit or the args 
            s_c['name'] += '_video'
            s_c['ee_args']["video_art_in_channels"]=args.scene_video_art_feature_size
        
            strategies.append(s_c)
    
    if args.b_mean_pool:
        s = {
                'name':'rnn_meanPoolProc', # baseline
                'ee_args':{
                    'useLSTM':useLSTMinPlaceOfGRU,
                    'desc_output_feature_size':args.output_feature_size,
                    'desc_input_feature_size':args.desc_feature_size,
                    'desc_model_is_bidirectional':args. desc_rnn_is_bidir,
                    'scene_input_channel':args.scene_feature_size,
                    'scene_output_channels':args.output_feature_size,
                },
                'uses_flattened_hierarchy':True,
            }
        
        if args.no_image_video:
            strategies.append(s)

        if args.with_image_video:
            s_c = deepcopy(s) # deep copy the dictionary to avoid issues with the edit or the args 
            s_c['name'] += '_image_video'
            s_c['ee_args']["image_art_in_channels"]=args.scene_image_art_feature_size
            s_c['ee_args']["video_art_in_channels"]=args.scene_video_art_feature_size
        
            strategies.append(s_c)
        
        if args.with_video:
            s_c = deepcopy(s) # deep copy the dictionary to avoid issues with the edit or the args 
            s_c['name'] += '_video'
            s_c['ee_args']["video_art_in_channels"]=args.scene_video_art_feature_size
        
            strategies.append(s_c)

    if args.b_rnn:
        s = {
                'name':'rnn_rnn',
                'ee_args':{
                    'useLSTM':useLSTMinPlaceOfGRU,
                    'desc_output_feature_size':args.output_feature_size,
                    'desc_input_feature_size':args.desc_feature_size,
                    'desc_model_is_bidirectional':args. desc_rnn_is_bidir,
                    'scene_input_channel':args.scene_feature_size,
                    'scene_output_channels':args.output_feature_size,
                    'scene_model_is_bidirectional':args.scene_rnn_is_bidir,
                },
                'uses_flattened_hierarchy':True,
            }
            
        if args.no_image_video:
            strategies.append(s)
        
        if args.with_image_video:
            print(f'with_image_video not supported by b_rnn, skipping it!')
        
        if args.with_video:
            print(f'with_video not supported by b_rnn, skipping it!')

    if args.h_rnn:
        s = {
                'name':'hier_by_room_rnn_meanPoolProc',
                'ee_args':{
                    'useLSTM':useLSTMinPlaceOfGRU,
                    'desc_output_feature_size':args.output_feature_size,
                    'desc_input_feature_size':args.desc_feature_size,
                    'desc_model_is_bidirectional':args. desc_rnn_is_bidir,
                    'roomModelArgs':{
                        "scene_in_channels":args.scene_feature_size,
                        "out_channels":args.h_inter_feature_size
                    },
                    'scenesSequenceModelArgs':{
                        "num_features":args.h_inter_feature_size,
                        "hidden_size":args.output_feature_size,
                        "is_bidirectional":args.h_rnn_is_bidir,
                    }
                },
                'uses_flattened_hierarchy':False,
            }
            
        if args.no_image_video:
            strategies.append(s)

        if args.with_image_video:
            s_c = deepcopy(s) # deep copy the dictionary to avoid issues with the edit or the args 
            s_c['name'] += '_image_video'
            s_c['ee_args']['roomModelArgs']["image_art_in_channels"]=args.scene_image_art_feature_size
            s_c['ee_args']['roomModelArgs']["video_art_in_channels"]=args.scene_video_art_feature_size
        
            strategies.append(s_c)
        
        if args.with_video:
            s_c = deepcopy(s) # deep copy the dictionary to avoid issues with the edit or the args 
            s_c['name'] += '_video'
            s_c['ee_args']['roomModelArgs']["video_art_in_channels"]=args.scene_video_art_feature_size
        
            strategies.append(s_c)
    
    if args.h_mean_pool_hierart:
        s = {
                'name':'hier_by_room_rnn_2_meanPoolProc',
                'ee_args':{
                    'useLSTM':useLSTMinPlaceOfGRU,
                    'desc_output_feature_size':args.output_feature_size,
                    'desc_input_feature_size':args.desc_feature_size,
                    'desc_model_is_bidirectional':args. desc_rnn_is_bidir,
                    'roomModelArgs':{
                        "scene_in_channels":args.scene_feature_size,
                        "out_channels":args.h_inter_feature_size,
                        "skip_last_linear":True
                    },
                    'scenesSequenceModelArgs':{
                        "scene_in_channels":args.h_inter_feature_size,
                        "out_channels":args.output_feature_size,
                        "skip_last_linear":False
                    }
                },
                'uses_flattened_hierarchy':False,
            }
            
        if args.no_image_video:
            strategies.append(s)

        if args.with_image_video:
            s_c = deepcopy(s) # deep copy the dictionary to avoid issues with the edit or the args 
            s_c['name'] += '_image_video'
            s_c['ee_args']['roomModelArgs']["image_art_in_channels"]=args.scene_image_art_feature_size
            s_c['ee_args']['roomModelArgs']["video_art_in_channels"]=args.scene_video_art_feature_size
            s_c['ee_args']['scenesSequenceModelArgs']["scene_in_channels"]=args.h_inter_feature_size*3 # correction due to skip_last_linear=True for the 1st net
        
            strategies.append(s_c)
        
        if args.with_video:
            s_c = deepcopy(s) # deep copy the dictionary to avoid issues with the edit or the args 
            s_c['name'] += '_video'
            s_c['ee_args']['roomModelArgs']["video_art_in_channels"]=args.scene_video_art_feature_size
            s_c['ee_args']['scenesSequenceModelArgs']["scene_in_channels"]=args.h_inter_feature_size*2 # correction due to skip_last_linear=True for the 1st net
        
            strategies.append(s_c)

    

    if args.h_cnn1d:
        s = {
                'name':'hier_by_room_rnn_meanPoolProc_cnn1d',
                'ee_args':{
                    'useLSTM':useLSTMinPlaceOfGRU,
                    'desc_output_feature_size':args.output_feature_size,
                    'desc_input_feature_size':args.desc_feature_size,
                    'desc_model_is_bidirectional':args. desc_rnn_is_bidir,
                    'roomModelArgs':{
                        "scene_in_channels":args.scene_feature_size,
                        "out_channels":args.h_inter_feature_size
                    },
                    'scenesSequenceModelArgs':{
                        'in_channels':args.h_inter_feature_size,
                        'out_channels':args.cnn1d_inter_size,
                        'kernel_size':args.kernel_size,
                        'feature_size':args.output_feature_size,
                    }
                },
                'uses_flattened_hierarchy':False,
            }

        if args.no_image_video:
            strategies.append(s)

        if args.with_image_video:
            s_c = deepcopy(s) # deep copy the dictionary to avoid issues with the edit or the args 
            s_c['name'] += '_image_video'
            s_c['ee_args']['roomModelArgs']["image_art_in_channels"]=args.scene_image_art_feature_size
            s_c['ee_args']['roomModelArgs']["video_art_in_channels"]=args.scene_video_art_feature_size
        
            strategies.append(s_c)
        
        if args.with_video:
            s_c = deepcopy(s) # deep copy the dictionary to avoid issues with the edit or the args 
            s_c['name'] += '_video'
            s_c['ee_args']['roomModelArgs']["video_art_in_channels"]=args.scene_video_art_feature_size
        
            strategies.append(s_c)

    
    if args.gru and args.lstm: # repeat each element with lstm if needed
        new_strategies = []
        for s in strategies:
            s_c = deepcopy(s)
            s_c['ee_args']['useLSTM'] = True
            new_strategies.append(s_c)
        strategies += new_strategies

    configs = [
        EncodersExecutorConfig(
            strategy_name=s['name'],
            ee_args=s['ee_args'],
            uses_flattened_hierarchy=s['uses_flattened_hierarchy'],
        )
        for s in strategies
    ]

    # save strategies
    for c in configs:
        fn = os.path.join(args.out_dir, f'{c.strategy_name + ("_lstm" if c.ee_args["useLSTM"] else "")}.json')
        with open(fn, 'w') as f:
            f.write(c.to_json_string())



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--out_dir', type=str, required=True, 
                        help="directory for the output of the configurations")
    
    # image and video addition
    parser.add_argument('--no_image_video', action='store_true', 
                        help="wether to generate the configurations that do not take in input the image and video elements")
    parser.add_argument('--with_image_video', action='store_true', 
                        help="whether to generate the configurations that take also artworks images and videos as inputs")
    parser.add_argument('--with_video', action='store_true', 
                        help="whether to generate the configurations that take also artworks videos as inputs")

    ## network structure
    
    # base models
    parser.add_argument('--b_mean_pool', action='store_true', 
                        help="whether to generate the configurations that use the mean pool approach in a flattened fashion")
    parser.add_argument('--b_cnn1d', action='store_true', 
                        help="whether to generate the configurations that use the cnn1d approach in a flattened fashion")
    parser.add_argument('--b_rnn', action='store_true', 
                        help="whether to generate the configurations that use the rnn approach in a flattened fashion (available for pov_only)")
    
    # hierarchical models
    parser.add_argument('--h_cnn1d', action='store_true', 
                        help="whether to generate the configurations that use the cnn1d hierarchically")
    parser.add_argument('--h_rnn', action='store_true', 
                        help="whether to generate the configurations that use the rnn hierarchically")
    parser.add_argument('--h_rnn_is_bidir', action='store_true', 
                        help="whether to use a bidirectional rnn for the rnn for the visual part in the hierarchical setting")
    parser.add_argument('--h_mean_pool_hierart', action='store_true', 
                        help="whether to generate the configurations that use the two mean poll networks hierarchically as in hierart (final linear layer of the first net is suppressed)")
    

    # rnn choice
    parser.add_argument('--gru', action='store_true', 
                        help="whether to generate the configurations using gru, either this or --lstm has to be true")
    parser.add_argument('--lstm', action='store_true', 
                        help="whether to generate the configurations adding lstm, either this or --gru has to be true")
    parser.add_argument('--desc_rnn_is_bidir', action='store_true', 
                        help="whether to use a bidirectional rnn for the rnn for the textual part")
    parser.add_argument('--scene_rnn_is_bidir', action='store_true', 
                        help="whether to use a bidirectional rnn for the rnn for the scene part in b_rnn")
    

    ## other params
    parser.add_argument('--kernel_size', type=int, default=None, 
                        help="size of the kernel for the cnn1d")
    parser.add_argument('--feature_size', type=int, default=None, 
                        help="size of the features")
    
    # feature sizes
    parser.add_argument('--desc_feature_size', type=int, default=None, 
                        help="size of the features for the descriptions, overwrite not-null elements and is superseded by finer definitions")
    parser.add_argument('--scene_feature_size', type=int, default=None, 
                        help="size of the features for the scene visual images")
    parser.add_argument('--scene_image_art_feature_size', type=int, default=None, 
                        help="size of the features for the image art elements")
    parser.add_argument('--scene_video_art_feature_size', type=int, default=None, 
                        help="size of the features for the video art elements")

    parser.add_argument('--output_feature_size', type=int, required=True, 
                        help="size of the features")
    
    parser.add_argument('--cnn1d_inter_size', type=int, default=None, 
                        help="size of the intermediate layer for cnn1d")
    parser.add_argument('--h_inter_feature_size', type=int, default=None, 
                        help="size of the intermediate feature size for the hierarchical models")
    
    
    args = parser.parse_args()


    assert not (args.b_cnn1d or args.h_cnn1d) or ((args.kernel_size is not None) and (args.cnn1d_inter_size is not None)), "kernel_size and cnn1d_inter_size are required for cnn1d"
    assert not (args.h_rnn or args.h_cnn1d) or (args.h_inter_feature_size is not None), "h_inter_feature_size required for hierarchical models"

    in_fss = [args.scene_feature_size, args.desc_feature_size]
    if args.with_image_video:
        in_fss += [args.scene_image_art_feature_size, args.scene_video_art_feature_size]
    elif args.with_video:
        in_fss += [args.scene_video_art_feature_size]
    assert args.feature_size or all([el is not None for el in in_fss]), "when feature_size is not given all the necessary feature sized must be specified"

    # set missing features
    if args.desc_feature_size is None:
        args.desc_feature_size = args.feature_size
    if args.scene_feature_size is None:
        args.scene_feature_size = args.feature_size
    if args.with_image_video and args.scene_image_art_feature_size is None:
        args.scene_image_art_feature_size = args.feature_size
    if (args.with_image_video or args.with_video) and args.scene_video_art_feature_size is None:
        args.scene_video_art_feature_size = args.feature_size

    os.makedirs(args.out_dir, exist_ok=False)

    generate_config_files(args)


# python create_config_files.py --out_dir src/data/training_configs/basicConfigs_in512_out256_k3_256 --no_image_video --with_image_video --b_mean_pool --b_cnn1d --h_cnn1d --h_rnn --h_rnn_is_bidir --gru --lstm --desc_rnn_is_bidir --kernel_size 3 --feature_size 512 --output_feature_size 256 --cnn1d_inter_size 256 --h_inter_feature_size 256