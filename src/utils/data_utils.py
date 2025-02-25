from src.data.featureDataset import ArtExhibitionFeatureDataset
from torch.utils.data import DataLoader

def get_dataloader(split:str, args, flatten_hierarchy, exhibitionDataset):
    aefd = ArtExhibitionFeatureDataset(
        args.feature_dir,
        args.feature_set_name,
        split,
        flatten_hierarchy,
        exhibitionDataset
    )
    collate_fn = aefd.get_collate_fn()
    dataloader = DataLoader(aefd, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=(split=='train'), num_workers=args.num_workers) # her we may want to explicitly set the sampler for better reproducibility: sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
    return dataloader     

def get_dataloaders(args, flatten_hierarchy, exhibitionDataset):
    return {
        'train':get_dataloader('train', args, flatten_hierarchy, exhibitionDataset),
        'val':get_dataloader('val', args, flatten_hierarchy, exhibitionDataset),
        'test':get_dataloader('test', args, flatten_hierarchy, exhibitionDataset),
    }