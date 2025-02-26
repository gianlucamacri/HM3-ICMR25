import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import itertools


class Museums3kDataset(Dataset):

    availableSplits = {'train', 'val', 'test'}

    __package_dir = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        
        data_scene_path = "tmp_museums/open_clip_features_museums3k/images"
        data_description_path = "tmp_museums/open_clip_features_museums3k/descriptions/sentences"

        self.images_per_room  = 12

        indices = pd.read_pickle(os.path.join(Museums3kDataset.__package_dir,"indices_museum_dataset.pkl"))
        
        available_data = [im.strip(".pt") for im in os.listdir(os.path.join(Museums3kDataset.__package_dir,data_scene_path))]
        available_data = sorted(available_data)
        #available_data = {s:[available_data[ix] for ix in indices[s].tolist()] for s in Museums3kDataset.availableSplits}
        
        self.number_of_rooms_map = {}

        self.pov_images = [torch.load(os.path.join(Museums3kDataset.__package_dir, data_scene_path, f"{sm}.pt")) for sm in available_data]
        self.descs = [torch.load(os.path.join(Museums3kDataset.__package_dir, data_description_path, f"{sm}.pt")) for sm in available_data]

        self.id_to_idx_map = {}
        
        self.museumsPerSplit = {}
        for s in Museums3kDataset.availableSplits:
            self.museumsPerSplit[s] = []
            for idx in indices[s]:
                id = self.__class__.transform_id(available_data[int(idx)], s, int(idx))
                self.id_to_idx_map[id] = idx
                self.museumsPerSplit[s].append(id)
                pov_dim = self.pov_images[int(idx)].shape[0]
                self.number_of_rooms_map[id] = pov_dim//self.images_per_room
                assert pov_dim % self.images_per_room == 0, f'assuming {pov_dim} is a multiple of {self.images_per_room}'

    def __len__(self):
        return len(self.pov_images)

    def __getitem__(self, index):
        desc_tensor = self.descs[index]
        scene_img_tensor = self.pov_images[index]
        
        return desc_tensor, scene_img_tensor, index
    
    def id_to_idx(self, id):
        return self.id_to_idx_map[id]

    @classmethod
    def transform_id(cls, id:str, split:str, i:int)-> str:
            """function to adapt to the naming scheme convention of the other datasets
            """
            assert '_' not in id, f'assuming id does not contain "_", but found: {id}'
            return f"{id}_{split}_C_{i}" # each element will have a unique category

        
    def getMuseumList(self, split=None):
        if split != None:
            assert split in Museums3kDataset.availableSplits, f"split {split} does not match any of the avaliable chices: {Museums3kDataset.availableSplits}"
            return self.museumsPerSplit[split]
        else:
            return list(itertools.chain.from_iterable(self.museumsPerSplit.values()))
    

    def getMuseumArtWorksPerRoom(self, id):

        number_of_rooms = self.number_of_rooms_map[id]

        # no artworks information
        imageIds = [[] for _ in range(number_of_rooms)]
        videoIds = imageIds

        return imageIds, videoIds
    
