# digitalArtExhibitionsRetrieval

This project aims at exploring some AI techniques in order to address the retrieval problem of complex 3d scenes based on a user textual query. Specifically the chosen scenario involves 3d scenes regarding exhibitions of art in the for of 2d images and videos.

## setup

You should include setup the following datasets as follows


### SemArt dataset

To complete the `src/data/datasets/SemArt` directory you should download the SemArt dataset you can find [here](https://researchdata.aston.ac.uk/id/eprint/380/).
Then unzip the `SemArt.zip` and copy its contents in that folder.

### SemArtVideoArtGen
this is a github repo TODO

### digitalMuseumsSemArtVideoGenDataset
this is a github repo TODO

### Museums3k dataset

To complete the `src/data/datasets/museums3k` directory you should download the museums3k  data form [here](https://github.com/aranciokov/HierArtEx-MMM2025/tree/main). Specifically you should copy the `tmp_museums` dir as it is and the `indices_museum_dataset.pkl` and place them at in `src/data/datasets/museums3k`.


## Usage

### feature extraction

First extract the videos as i CLIP4Clip as follows withing the AV_CLIP4Clip repository
```
torchrun --nproc_per_node=1 main_task_retrieval.py --do_featureExt --num_thread_reader=1 --n_display=50 --metadata_fn metadata --video_dir compressed_videos --split all_test --output_dir ckpts/av_featureExt_clip4clip --lr 2e-5 --max_frames 32 --datatype semart_generated_artistic_videos --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0 --slice_framepos 2 --loose_type --linear_patch 2d --sim_header meanP --pretrained_clip_name ViT-B/32 --best_model_strategy loss --use_caching --test_best_model --featureExtDir features/
```

Then complete the extraction with

```
python extract_features.py --feature_output_dir_name genMuseums --dataset generatedDigitalExps --feature_set_name base_by_sentence_clip_32_fs --text_split_strategy sentence_by_sentence --video_feature_fn artisticVideosFeature_CLIP4clip_32fps__base.pickle
```


### config generation

From the main folder execute 

```
python create_config_files.py --out_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_only --no_image_video --b_mean_pool --b_cnn1d --h_cnn1d --h_rnn --h_rnn_is_bidir --gru --desc_rnn_is_bidir --kernel_size 3 --feature_size 512 --output_feature_size 256 --cnn1d_inter_size 256 --h_inter_feature_size 256
```

```
python create_config_files.py --out_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --with_video --b_mean_pool --b_cnn1d --h_cnn1d --h_rnn --h_rnn_is_bidir --gru --desc_rnn_is_bidir --kernel_size 3 --feature_size 512 --output_feature_size 256 --cnn1d_inter_size 256 --h_inter_feature_size 256
```


recreate hierart configs extended
```
python create_config_files.py --out_dir src/data/training_configs/hierArt_in512_out256_k3_256_pov_only --no_image_video --b_mean_pool --b_cnn1d --b_rnn --h_cnn1d --h_rnn --h_rnn_is_bidir --scene_rnn_is_bidir --h_mean_pool_hierart --gru --desc_rnn_is_bidir --kernel_size 3 --feature_size 512 --output_feature_size 256 --cnn1d_inter_size 256 --h_inter_feature_size 256
```



### training and evaluation

```
python main.py --train --feature_dir museums3k --feature_set_name base_sentence_open_clip  --config_files_dir src/data/training_configs/hierArt_in512_out256_k3_256_pov_only --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 50 --lr 0.001 --number_of_tries 3 --save_strategy last --device cuda --hf_tags "hierart_comparison" --experiment_name hierArt_comparison --experiment_description "experiments to make a comparison of the proposed architectures with respect to those use in the "
```

```
python main.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --hf_tags --hf_tags "STL" "new 16th" --experiment_name "STL" --experiment_description "basic experiment with the standard triplet loss" 
```

```
python main.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --use_categories_in_loss --loss_within_category_margin 0.15 --second_loss_component_weight 0.75 --hf_tags "new 18th 2nd w 0.75" --experiment_name "CN_TC_2nd_w_0.75" --experiment_description "experiemnt with 2 loss components inter and intra class, margins 0.25 and 0.15, varing the weight of the two components"
```

```
python main.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --use_categories_in_loss --loss_within_category_margin 0.15 --second_loss_component_weight 0.5 --hf_tags "new 18th 2nd w 0.5" --experiment_name "CN_TC_2nd_w_0.5" --experiment_description "experiemnt with 2 loss components inter and intra class, margins 0.25 and 0.15, varing the weight of the two components"
```

```
python main.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --use_categories_in_loss --loss_within_category_margin 0.15 --second_loss_component_weight 0.25 --hf_tags "new 18th 2nd w 0.25" --experiment_name "CN_TC_2nd_w_0.25" --experiment_description "experiemnt with 2 loss components inter and intra class, margins 0.25 and 0.15, varing the weight of the two components"
```

```
python main.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --use_categories_in_loss --loss_within_category_margin 0.15 --second_loss_component_weight 0.1 --hf_tags "new 18th 2nd w 0.1" --experiment_name "CN_TC_2nd_w_0.1" --experiment_description "experiemnt with 2 loss components inter and intra class, margins 0.25 and 0.15, varing the weight of the two components"
```

```
python main.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --use_categories_in_loss --loss_within_category_margin 0.15 --second_loss_component_weight 0.05 --hf_tags "new 18th 2nd w 0.05" --experiment_name "CN_TC_2nd_w_0.05" --experiment_description "experiemnt with 2 loss components inter and intra class, margins 0.25 and 0.15, varing the weight of the two components"
```

```
python main.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --use_categories_in_loss --loss_within_category_margin 0.15 --second_loss_component_weight 0.01 --hf_tags "new 18th 2nd w 0.01" --experiment_name "CN_TC_2nd_w_0.01" --experiment_description "experiemnt with 2 loss components inter and intra class, margins 0.25 and 0.15, varing the weight of the two components"
```

```
python main.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --use_categories_in_loss --loss_within_category_margin 0.15 --second_loss_component_weight 0 --hf_tags "new 18th 2nd w 0" "new 17th" --experiment_name "CN_TC_2nd_w_0" --experiment_description "experiemnt with 2 loss components inter and intra class, margins 0.25 and 0.15, varing the weight of the two components, second loss weight to 0, equivalent to not using it"
```

```
python main.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_only --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --use_categories_in_loss --loss_within_category_margin 0.15 --second_loss_component_weight 0.05 --hf_tags "new 18th 2nd w 0.05" --experiment_name "CN_TC_2nd_w_0.05" --experiment_description "experiemnt with 2 loss components inter and intra class, margins 0.25 and 0.15, varing the weight of the two components"
```

```
python main.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --use_categories_in_loss --loss_within_category_margin 0.1 --second_loss_component_weight 0.05 --hf_tags "new 22th 2nd w 0.05 2nd m 0.1" --experiment_name "CN_TC_2nd_w_0.05_2nd_m_0.1" --experiment_description "experiemnt with 2 loss components inter and intra class, margins 0.25 and 0.1, varing the weight of the two components"
```

```
python main.py --train --feature_dir genMuseums --feature_set_name base_by_sentence_clip_32_fs --config_files_dir src/data/training_configs/basicConfigs_in512_out256_k3_256_pov_and_video --seed 424242 --wandb_project_name paperDigitalArtExhibitions --hf_user_name paperDigitalArtExhibitions --batch_size 64 --epochs 25 --scheduler_step_size 5 --lr 0.0005 --number_of_tries 3 --save_strategy last --device cuda --loss_margin 0.25 --use_categories_in_loss --loss_within_category_margin 0.2 --second_loss_component_weight 0.05 --hf_tags "new 22th 2nd w 0.05 2nd m 0.2" --experiment_name "CN_TC_2nd_w_0.05_2nd_m_0.2" --experiment_description "experiemnt with 2 loss components inter and intra class, margins 0.25 and 0.2, varing the weight of the two components"
```
