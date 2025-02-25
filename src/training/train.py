import torch
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import wandb
from src.models.models import EncodersExecutor, EncodersExecutorConfig
from src.utils.model_utils import cosine_sim, get_hf_full_name, get_model_output_dir
from src.evaluation.evaluation_utils import evaluate, BestModelManager
from src.training.loss_functions import *
from src.evaluation.inference import inference
from time import sleep
from collections import Counter

def train(full_run_name:str, encodersExecutor:EncodersExecutor, bmm:BestModelManager, dataloaders, args, logger,use_wandb):
    encodersExecutor.setTrain()

    loss_fn = LossContrastive(patience=args.loss_patience, delta=args.loss_delta, logger=logger, use_wandb=use_wandb)
    #loss_fn = LossContrastive(patience=25, delta=0.0001, logger=logger)

    optimizer = torch.optim.Adam(encodersExecutor.getModelParams(), lr=args.lr)
    
    # params that may need to be passed with command line args
    
    if args.scheduler == 'step_lr':
        step_size = args.scheduler_step_size # 27
        gamma = args.scheduler_gamma # 0.75
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        sched_name = StepLR
    else:
        assert False, f'unknown scheduler: {args.scheduler}'
    
    # TODO: possibly alternative schedulers
    ## #1
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0, last_epoch=-1)
    # sched_name = CosineAnnealingLR
    ## #2
    # num_training_steps = (len(train_dataset) * num_epochs) // batch_size
    # num_warmup_steps = int(num_training_steps * 0.1)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    # sched_name = "cosine_schedule_with_warmup"
    
    use_wandb = args.wandb_project_name is not None
    if use_wandb:
        wandb.config.update(
            {
                "scheduler/name": sched_name,
                **{f"scheduler/{n}": v for n, v in scheduler.__dict__.items()}
            }
        )
    
    output_feature_size = encodersExecutor.getConfig().get_output_size()

    loss_components_weights = [1-args.second_loss_component_weight, args.second_loss_component_weight] if args.second_loss_component_weight is not None else [0.5,0.5]

    for ep in tqdm(range(args.epochs)):
                        
        epoch_loss_train = train_one_epoch(encodersExecutor, loss_fn, args.loss_margin, scheduler, optimizer, dataloaders['train'], args.device, logger, use_wandb, args.use_categories_in_loss, args.loss_within_category_margin, args.new_positive_samples, args.allow_sampling_replacement_of_new_positives, loss_components_weights, args.filter_out_hard_negatives)

        # validation on the validation set
        output_description_val, output_pov_val, output_categories, epoch_loss_val = inference(encodersExecutor, loss_fn, output_feature_size, dataloaders['val'], args.batch_size, args.device, use_wandb, is_test=False, use_categories_in_loss=args.use_categories_in_loss)

        evalRes = evaluate(output_description=output_description_val,
                            output_scene=output_pov_val,
                            scene_classes= output_categories,
                            description_classes= output_categories)
      
        evalRes['epoch_loss'] = epoch_loss_val # adding loss to eval Res for the update function


        isNewBest = bmm.updateBest(evalRes)

        if isNewBest:
            logger.info(f"found new best at {bmm.get_best_value()}")
        
        saveModel = (args.save_strategy == 'best' and isNewBest) or (args.save_strategy == 'last' and ep == (args.epochs - 1))

        if saveModel:
            if args.hf_user_name: # remote save
                max_push_retrial = 6
                sleep_base_time = 60
                model_got_uploaded = False
                for i in range(max_push_retrial):
                    try:
                        encodersExecutor.getConfig().push_to_hub(get_hf_full_name(args.hf_user_name, full_run_name))
                        encodersExecutor.push_to_hub(get_hf_full_name(args.hf_user_name, full_run_name), config=encodersExecutor.getConfig())
                        model_got_uploaded = True
                        break
                    except Exception as e:
                        logger.warning(f"an exception as occurred trying to upload a model: {e}")
                        logger.info(f"retrying in {sleep_base_time*(i**2)}s")
                
                if not model_got_uploaded:
                    logger.warning(f"last go at trying to upload the model")
                    encodersExecutor.getConfig().push_to_hub(get_hf_full_name(args.hf_user_name, full_run_name))
                    encodersExecutor.push_to_hub(get_hf_full_name(args.hf_user_name, full_run_name), config=encodersExecutor.getConfig())

            local_save_dir = get_model_output_dir(full_run_name)
            encodersExecutor.getConfig().save_pretrained(local_save_dir)
            encodersExecutor.save_pretrained(local_save_dir, config=encodersExecutor.getConfig())
            
    
        if use_wandb:
            wandb.log({
                f'val/{k}':v
                for k,v in evalRes.items()

            })

        # Validation train set to get the metrics on the training dataset
        output_description_val_train, output_pov_val_train, output_categories, epoch_loss_val = inference(encodersExecutor, loss_fn, output_feature_size, dataloaders['train'], args.batch_size, args.device, use_wandb=False, is_test=False, use_categories_in_loss=args.use_categories_in_loss) # use_wandb=False to suppress logging on the training set eval

        
        # not sure about this
        with torch.no_grad():
            logger.info(f'Loss Train: {epoch_loss_train}')
            loss_fn.on_epoch_end(epoch_loss_train, train=True)
            logger.info(f'Loss Val: {epoch_loss_val}')
            loss_fn.on_epoch_end(epoch_loss_val, train=False)

        evalRes= evaluate(output_description=output_description_val_train,
                            output_scene=output_pov_val_train,
                            scene_classes= output_categories,
                            description_classes= output_categories)
        
        if use_wandb:
            wandb.log({
                f'train/{k}':v
                for k,v in evalRes.items()
            })

        if args.second_loss_component_weight_decay_factor:
            second_component = min(1, loss_components_weights[1]*args.second_loss_component_weight_decay_factor)
            loss_components_weights = [1-second_component, second_component]

def train_one_epoch(encodersExecutor, loss_fn, loss_margin, scheduler, optimizer, train_dataloader, device, logger, use_wandb, use_categories_in_loss, loss_within_category_margin, new_positive_samples, allow_sampling_replacement_of_new_positives, loss_components_weights, filter_out_hard_negatives):
    encodersExecutor.setTrain()
   
    total_loss_train = 0
    num_batches_train = 0
    
    
    for i, (data_desc_pov, data_pov, data_imageart, data_videoart, indexes, len_pov, len_imageart, len_videoart, categories) in enumerate(train_dataloader):
        
        counter = Counter(categories.tolist())
        if use_wandb:
            wandb.log({
                **{f'train/units_c{k}':v for k,v in counter.items()},
                'train/units_avg':sum(counter.values())/len(counter)})
        
        data_desc_pov = data_desc_pov.to(device)
        data_pov = data_pov.to(device)
        data_imageart = data_imageart.to(device)
        data_videoart = data_videoart.to(device)
        categories = categories.to(device)

        batch = (data_desc_pov, data_pov, data_imageart, data_videoart, indexes, len_pov, len_imageart, len_videoart)

        optimizer.zero_grad()

        # bsz, fts, no_room_times_no_imgs = data_pov.shape

        output_desc_pov, output_pov= encodersExecutor.forward(batch)

        multiplication_dp = cosine_sim(output_desc_pov, output_pov)

        loss_contrastive = loss_fn.calculate_loss(multiplication_dp, margin=loss_margin, within_category_margin = loss_within_category_margin, categories=categories if use_categories_in_loss else None, new_positive_samples =new_positive_samples, allow_sampling_replacement_of_new_positives=allow_sampling_replacement_of_new_positives, loss_components_weights=loss_components_weights, filter_out_hard_negatives=filter_out_hard_negatives)

        loss_contrastive.backward()
        optimizer.step()

        total_loss_train += loss_contrastive.item()
        num_batches_train += 1
        
        if use_wandb:
            wandb.log({
                "train/loss": loss_contrastive.item(), 
                "scheduler/lr": scheduler.get_last_lr()[0]
            })

    scheduler.step()
    logger.info(scheduler.get_last_lr())
    epoch_loss_train = total_loss_train / num_batches_train

    return epoch_loss_train