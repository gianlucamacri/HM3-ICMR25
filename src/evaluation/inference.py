import wandb
import torch
from src.utils.model_utils import cosine_sim

def inference(encodersExecutor, loss_fn, output_feature_size, dataloader, batch_size, device, use_wandb, is_test:bool, use_categories_in_loss:bool):
    total_loss = 0
    num_batches = 0
    epoch_loss = None

    output_description = torch.empty(len(dataloader.dataset), output_feature_size)
    output_pov = torch.empty(len(dataloader.dataset), output_feature_size)
    output_categories = torch.empty(len(dataloader.dataset))


    encodersExecutor.setEval()

    with torch.no_grad():
        for j, (data_desc_pov, data_pov, data_imageart, data_videoart, indexes, len_pov,  len_imageart, len_videoart, categories) in enumerate(dataloader):

            data_desc_pov = data_desc_pov.to(device)
            data_pov = data_pov.to(device)
            data_imageart = data_imageart.to(device)
            data_videoart = data_videoart.to(device)
            categories = categories.to(device)

            # bsz, fts, no_room_times_no_imgs = data_pov.shape

            batch = (data_desc_pov, data_pov, data_imageart, data_videoart, indexes, len_pov,  len_imageart, len_videoart)
            batch_output_desc_pov, batch_output_pov = encodersExecutor.forward(batch)

            initial_index = j * batch_size
            final_index = (j + 1) * batch_size
            if final_index > len(dataloader.dataset):
                final_index = len(dataloader.dataset)

            output_description[initial_index:final_index, :] = batch_output_desc_pov
            output_pov[initial_index:final_index, :] = batch_output_pov
            output_categories[initial_index:final_index] = categories

            if loss_fn:
                multiplication_dp = cosine_sim(batch_output_desc_pov, batch_output_pov)

                loss_contrastive = loss_fn.calculate_loss(multiplication_dp, categories=categories if use_categories_in_loss else None)

                total_loss += loss_contrastive.item()
                num_batches += 1
                if use_wandb:
                    wandb.log({
                        f"{'test' if is_test else 'val'}/batch_loss": loss_contrastive.item(), 
                    })

        if loss_fn:
            epoch_loss = total_loss / num_batches
            if use_wandb:
                wandb.log({
                    f"{'test' if is_test else 'val'}/epoch_loss": epoch_loss, 
                })

    
    return output_description, output_pov, output_categories, epoch_loss