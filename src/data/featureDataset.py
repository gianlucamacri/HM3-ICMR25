
from src.data.feature_utils import DigitalExhibitionFeatureSet
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pickle
from itertools import chain
from typing import Optional
import json
import numpy as np
from tqdm.auto import tqdm
import math
import random


class ArtExhibitionFeatureDataset(Dataset):

    # to do add ignore the lobby for images and videos ? 

    def __init__(self, feature_path, featuresFn,  split, flatten_room_hierarchy:bool, exhibitionsDataset, data_augmentation_fn:Optional[str]=None, logger=None):

        if data_augmentation_fn is not None:
            # load the data augmentation config
            data_augmentation_path = f'src/data/data_augmentation/{data_augmentation_fn}'
            logger.info(f'setting data augmentation from {data_augmentation_path}')
            with open(data_augmentation_path, 'r') as f:
                data_augmentation_dict = json.load(f)
        else:
            logger.info(f'no data augmentation requested')
            data_augmentation_dict = {'name':None}
        self.data_augmentation = self.__class__.strategy_from_dict(data_augmentation_dict)
        assert isinstance(self.data_augmentation, self.DataAugmentationStrategy), f"data augmentation fn must be of type DataAugmentationStrategy, got {type(self.data_augmentation)}"

        feature_set = DigitalExhibitionFeatureSet.loadFeatureSet(feature_path, featuresFn)

        self.flatten_room_hierarchy = flatten_room_hierarchy
        
        #assert self.flatten_room_hierarchy, f"unflatteded hierarchy is still in developement"

        self.feature_set = feature_set

        self.description_path = feature_set.get_descriptions(full_path=True)['data_fn']
        self.data_scene_path = feature_set.get_povs(full_path=True)['data_fn']
        self.data_image_art_path = feature_set.get_image_art(full_path=True)['data_fn']
        self.data_video_art_path = feature_set.get_video_art(full_path=True)['data_fn']

        with open(self.description_path, 'rb') as f:
            description_feature_dict = pickle.load(f)

        with open(self.data_scene_path, 'rb') as f:
            scene_feature_dict = pickle.load(f)

        def get_museum_category_from_name(name:str) -> str:
            return '_'.join(name.split('_')[-2:])

        museum_types = sorted({get_museum_category_from_name(id) for id in exhibitionsDataset.getMuseumList(None)})
        self.all_museum_category_str_idx_map = {t:idx for idx,t in enumerate(museum_types)}
        
        self.split_ids = exhibitionsDataset.getMuseumList(split)

        # def flatten_chain(els):
        #     return list(chain(*els))
        
        description_feature_dict = {k:v for k, v in description_feature_dict.items() if k in self.split_ids}
        #scene_feature_dict = {k:(v if not flatten_room_hierarchy else flatten_chain(v)) for k, v in scene_feature_dict.items() if k in self.split_ids}

        self.split_id_idx_map = {}
        self.image_ids_scene_dict_hier = {} 
        self.video_ids_scene_dict_hier = {}
        for id in self.split_ids:
            image_ids, video_ids = exhibitionsDataset.getMuseumArtWorksPerRoom(id)
            self.image_ids_scene_dict_hier[id] = image_ids # if not flatten_room_hierarchy else flatten_chain(image_ids)
            self.video_ids_scene_dict_hier[id] = video_ids # if not flatten_room_hierarchy else flatten_chain(video_ids)
            self.split_id_idx_map[id] = self.all_museum_category_str_idx_map[get_museum_category_from_name(id)]

        # with open(self.data_image_art_path, 'rb') as f:
        #     self.image_artworks_feature_dict = pickle.load(f)

        # with open(self.data_video_art_path, 'rb') as f:
        #     self.video_artworks_feature_dict = pickle.load(f)['features']

        # an hacky conversion to hdf5 format to save memory
        import os, h5py
        if logger: logger.info(f"Converting the feature dict to hdf5 format to save memory")
        self.feature_dict_h5f_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp", "tmp_features.hdf5")
        if os.path.exists(self.feature_dict_h5f_fn):
            os.remove(self.feature_dict_h5f_fn) # remove old
        compression = None

        with h5py.File(self.feature_dict_h5f_fn, "w") as h5f:
            
            if logger: logger.info(f"  - image features from {self.data_image_art_path}")
            with open(self.data_image_art_path, 'rb') as f:
                image_data_dict = pickle.load(f)
            group = h5f.create_group('images')
            for key, value in tqdm(image_data_dict.items()):
                str_key = str(key)
                group.create_dataset(str_key, data=value, compression=compression)

            if logger: logger.info(f"  - video features from {self.data_video_art_path}")
            with open(self.data_video_art_path, 'rb') as f:
                video_data_dict = pickle.load(f)['features']
            group = h5f.create_group('videos')
            for key, value in tqdm(video_data_dict.items()):
                str_key = str(key)
                group.create_dataset(str_key, data=value, compression=compression)

            if logger: logger.info(f"  - description hierarchical organization of {self.description_path}")
            descriptions_room_sentence_num = {
                k:[len(el) for el in v]
                for k,v in description_feature_dict.items()
            }
            group = h5f.create_group('descriptions_room_sentence_num')
            for key, value in tqdm(descriptions_room_sentence_num.items()):
                str_key = str(key)
                group.create_dataset(str_key, data=value, compression=None)

            if logger: logger.info(f"  - description features from {self.description_path}")
            group = h5f.create_group('descriptions')
            for key, value in tqdm(description_feature_dict.items()):
                str_key = str(key)
                group.create_dataset(str_key, data=torch.cat(value), compression=compression)
            
            if logger: logger.info(f"  - scene features from {self.data_scene_path}")
            group = h5f.create_group('scenes')
            for key, value in tqdm(scene_feature_dict.items()):
                str_key = str(key)
                group.create_dataset(str_key, data=value, compression=compression)

        self.feature_dict_h5f = h5py.File(self.feature_dict_h5f_fn, "r")

    def __len__(self):
        return len(self.split_ids)

    def __getitem__(self, index):

        id = self.split_ids[index]
        category = self.split_id_idx_map[id]

        # load descriptions reverting the flattening
        desc_tensor = torch.from_numpy(self.feature_dict_h5f['descriptions'][id][()])
        descriptions_room_sentence_num = self.feature_dict_h5f['descriptions_room_sentence_num'][id][()]
        idxs = np.hstack((0,descriptions_room_sentence_num.cumsum()))
        desc_tensor = [desc_tensor[idxs[i]:idxs[i+1]] for i in range(len(idxs)-1)]

        #scene_tensor = torch.from_numpy(self.feature_dict_h5f['scenes'][id][()])
        scene_tensor = tuple(el for el in torch.from_numpy(self.feature_dict_h5f['scenes'][id][()]))
        
        # todo fix missing images and videos case (is this "fixed"?)

        scene_image_art_tensor = [[torch.from_numpy(self.feature_dict_h5f['images'][id][()]).squeeze() for id in els] for els in self.image_ids_scene_dict_hier[id]]
        scene_image_art_tensor = [torch.stack(els) if len(els) != 0 else None for els in scene_image_art_tensor]

        scene_video_art_tensor = [[torch.from_numpy(self.feature_dict_h5f['videos'][id][()]) for id in els] for els in self.video_ids_scene_dict_hier[id]]
        scene_video_art_tensor = [torch.stack(els) if len(els) != 0 else None for els in scene_video_art_tensor]

        assert len(scene_tensor) == len(scene_image_art_tensor), f"scene_image_art_tensor dimensional mismatch error: {len(scene_tensor)} != {len(scene_image_art_tensor)}"
        assert len(scene_tensor) == len(scene_video_art_tensor), f"scene_video_art_tensor dimensional mismatch error: {len(scene_tensor)} != {len(scene_video_art_tensor)}"

        # we apply the data augmentation on the hierarchical version of the data
        # as e.g. room order shuffling requires to have the room information
        desc_tensor, scene_tensor, scene_image_art_tensor, \
            scene_video_art_tensor = self.data_augmentation.apply_augmentation(
                desc_tensor, scene_tensor, scene_image_art_tensor, scene_video_art_tensor
            )
        
        # description flattening
        desc_tensor = torch.cat(desc_tensor)
        
        # apply flattening if needed
        if self.flatten_room_hierarchy:
            scene_tensor = torch.cat(scene_tensor)
            scene_image_art_tensor = torch.cat([el for el in scene_image_art_tensor if el != None]) if len(self.image_ids_scene_dict_hier[id]) != 0 else None
            scene_video_art_tensor = torch.cat([el for el in scene_video_art_tensor if el != None])  if len(self.video_ids_scene_dict_hier[id]) != 0 else None

        return desc_tensor, scene_tensor, scene_image_art_tensor, scene_video_art_tensor, index, category
  
    def getFeatureSize(self):
        idx = 0
        datasetSize = len(self)

        # this could be imporved as now we are looking for a tuple with all the elements but a safer approach would be to loop component by componenet
        while True:
            assert idx < datasetSize, f"cannot find an element with all the features"

            desc_tensor, scene_tensor, scene_image_art_tensor, scene_video_art_tensor, _ = self[idx]

            desc_feature_size = desc_tensor.shape[1]
            if self.flatten_room_hierarchy:
                scene_feature_size = scene_tensor.shape[1]
                if scene_image_art_tensor is None:
                    idx += 1
                    continue # try next element
                scene_image_art_feature_size = scene_image_art_tensor.shape[1]
                if scene_video_art_tensor is None:
                    idx += 1
                    continue # try next element
                scene_video_art_feature_size = scene_video_art_tensor.shape[1]
            else:
                scene_feature_size = scene_tensor[0].shape[1]
                if scene_image_art_tensor[1] == [None]: # from 1 as the lobby is empty
                    idx += 1
                    continue # try next element
                scene_image_art_feature_size = scene_image_art_tensor[1].shape[1]
                if scene_video_art_tensor[1] is None:
                    idx += 1
                    continue # try next element
                scene_video_art_feature_size = scene_video_art_tensor[1].shape[1]
            
            break

        return desc_feature_size, scene_feature_size, scene_image_art_feature_size, scene_video_art_feature_size
    

    def get_collate_fn(self):

        desc_feature_size = self.feature_set.get_descriptions()['feature_size']
        scene_feature_size = self.feature_set.get_povs()['feature_size']
        scene_image_art_feature_size = self.feature_set.get_image_art()['feature_size']
        scene_video_art_feature_size = self.feature_set.get_video_art()['feature_size']

        def collate_fn(data):

            # data -> desc_tensor, scene_tensor, scene_image_art_tensor, scene_video_art_tensor, index, category

            # desc_tensor
            tmp_description_povs = [x[0] for x in data]
            tmp = pad_sequence(tmp_description_povs, batch_first=True)
            descs_pov = pack_padded_sequence(tmp,
                                            torch.tensor([len(x) for x in tmp_description_povs]),
                                            batch_first=True,
                                            enforce_sorted=False) # batch_size x number_of_scene_art_img x feature_size
            
            catagories = torch.tensor([x[5] for x in data])


            if self.flatten_room_hierarchy:
                # scene_img_tensor
                tmp_scene_pov = [x[1] for x in data]
                len_scene_pov = torch.tensor([len(x) for x in tmp_scene_pov])
                padded_scene_pov = pad_sequence(tmp_scene_pov, batch_first=True)

                # scene_image_art_tensor
                tmp_imageart = [x[2] if x[2] is not None else torch.zeros(1, scene_image_art_feature_size) for x in data]
                len_imageart = torch.tensor([len(x) for x in tmp_imageart])
                padded_imageart = pad_sequence(tmp_imageart, batch_first=True)

                # scene_video_art_tensor
                tmp_videoart = [x[3] if x[3] is not None else torch.zeros(1, scene_video_art_feature_size) for x in data]
                len_videoart = torch.tensor([len(x) for x in tmp_videoart])
                padded_videoart = pad_sequence(tmp_videoart, batch_first=True)

            else: # double padding is needed
                
                # scene_img_tensor
                tmp_scene_pov = [[xx for xx in x[1]] for x in data]
                maxRooms =  max([len(x) for x in tmp_scene_pov])
                len_scene_pov = torch.tensor([[len(xx) for xx in x]+ [0]*(maxRooms - len(x)) for x in tmp_scene_pov]) # also here no issues with size mismatch
                padded_scene_pov_by_room = [torch.stack(room) for room in tmp_scene_pov] # here no need to stack as the numebr of images per room is always the same
                padded_scene_pov = pad_sequence(padded_scene_pov_by_room, batch_first=True)

                # scene_image_art_tensor
                tmp_imageart = [[xx if xx is not None else torch.zeros(1, scene_image_art_feature_size) for xx in x[2]] for x in data]
                maxRooms =  max([len(x) for x in tmp_imageart])
                maxRoomSize = max([xx.shape[0] for x in tmp_imageart for xx in x])
                len_imageart = torch.tensor([[len(xx) for xx in x] + [0]*(maxRooms - len(x)) for x in tmp_imageart]) # to recover the higher padding number one can look at the first 0 in the sequence
                # Pad each tensor (room) in each list so that all tensors have the same first dimension
                padded_imageart_by_room = [
                    [torch.cat([xx, torch.zeros(maxRoomSize - xx.shape[0], scene_image_art_feature_size)], dim=0) for xx in x]
                    for x in tmp_imageart
                ]
                # Pad the lists of rooms so that all lists have the same number of rooms (maxRooms)
                padded_imageart_by_room = [
                    x + [torch.zeros(maxRoomSize, scene_image_art_feature_size)] * (maxRooms - len(x))
                    for x in padded_imageart_by_room
                ]
                # Convert the padded list of rooms into a tensor
                padded_imageart = torch.stack([torch.stack(room, dim=0) for room in padded_imageart_by_room], dim=0)

                # scene_video_art_tensor
                tmp_videoart = [[xx if xx is not None else torch.zeros(1, scene_video_art_feature_size) for xx in x[3]] for x in data]
                maxRooms =  max([len(x) for x in tmp_videoart])
                maxRoomSize = max([xx.shape[0] for x in tmp_videoart for xx in x])
                len_videoart = torch.tensor([[len(xx) for xx in x] + [0]*(maxRooms - len(x)) for x in tmp_videoart])
                padded_videoart_by_room = [
                    [torch.cat([xx, torch.zeros(maxRoomSize - xx.shape[0], scene_image_art_feature_size)], dim=0) for xx in x]
                    for x in tmp_videoart
                ]
                padded_videoart_by_room = [
                    x + [torch.zeros(maxRoomSize, scene_image_art_feature_size)] * (maxRooms - len(x))
                    for x in padded_videoart_by_room
                ]
                padded_videoart = torch.stack([torch.stack(room, dim=0) for room in padded_videoart_by_room], dim=0)
                
            
            indexes = [x[4] for x in data]
            
            return descs_pov, padded_scene_pov, padded_imageart, padded_videoart, indexes, len_scene_pov, len_imageart, len_videoart, catagories

        return collate_fn

    class DataAugmentationStrategy():
        def __init__(self, strategy_name:Optional[str]):
            self.strategy_name = strategy_name
        
        def apply_augmentation(self, desc_tensor, scene_tensor, scene_image_art_tensor, scene_video_art_tensor):
            raise NotImplementedError("This method should be overridden by subclasses")
        
        def to_dict(self) -> dict:
            """Return a JSON-serializable dict of this strategy."""
            return {"name": self.strategy_name}
        
        @classmethod
        def from_dict(cls, config: dict):
            """Recreate a strategy from a dict."""
            raise NotImplementedError("Must be implemented in subclass")
        
    class NoAugmentation(DataAugmentationStrategy):
        def __init__(self):
            super().__init__(None)

        def apply_augmentation(self, desc_tensor, scene_tensor, scene_image_art_tensor, scene_video_art_tensor):
            """
            method that does nothing
            """
            return desc_tensor, scene_tensor, scene_image_art_tensor, scene_video_art_tensor

        def to_dict(self) -> dict:
            base = super().to_dict()
            return base

        @classmethod
        def from_dict(cls, config: dict):
            return cls()

    class RoomShuffleAugmentation(DataAugmentationStrategy):
        def __init__(self, shuffle_prob: float = 1.0, fix_first_room: bool = False, fix_last_room: bool = False):
            super().__init__("RoomShuffleAugmentation")
            self.shuffle_prob = shuffle_prob
            self.fix_first_room = fix_first_room
            self.fix_last_room = fix_last_room

        def apply_augmentation(self, desc_tensors, scene_tensors, scene_image_art_tensors, scene_video_art_tensors):
            if torch.rand(1).item() < self.shuffle_prob:
                # compute a random permutation
                if not self.fix_first_room and not self.fix_last_room:
                    perm = torch.randperm(len(scene_tensors))
                elif self.fix_first_room and not self.fix_last_room:
                    perm = torch.cat((torch.tensor([0]), torch.randperm(len(scene_tensors)-1)+1))
                elif not self.fix_first_room and self.fix_last_room:
                    perm = torch.cat((torch.randperm(len(scene_tensors)-1), torch.tensor([len(scene_tensors)-1])))
                else: # both
                    if len(scene_tensors) < 2:
                        perm = torch.arange(len(scene_tensors)) # no shuffle possible
                    else:
                        perm = torch.cat((torch.tensor([0]), torch.randperm(len(scene_tensors)-2)+1, torch.tensor([len(scene_tensors)-1])))
                
                assert len(scene_tensors) == len(desc_tensors), f"mismatch between scenes and textual descriptions tensors, {len(scene_tensors)} vs {len(desc_tensors)}"
                desc_tensors = [desc_tensors[i] for i in perm]
                scene_tensors = [scene_tensors[i] for i in perm]
                if scene_image_art_tensors is not None:
                    assert len(scene_tensors) == len(scene_image_art_tensors), f"mismatch between scenes and image art tensors, {len(scene_tensors)} vs {len(scene_image_art_tensors)}"
                    scene_image_art_tensors = [scene_image_art_tensors[i] for i in perm]
                if scene_video_art_tensors is not None:
                    assert len(scene_tensors) == len(scene_video_art_tensors), f"mismatch between scenes and video art tensors, {len(scene_tensors)} vs {len(scene_video_art_tensors)}"
                    scene_video_art_tensors = [scene_video_art_tensors[i] for i in perm]
            return desc_tensors, scene_tensors, scene_image_art_tensors, scene_video_art_tensors

        def to_dict(self) -> dict:
            base = super().to_dict()
            base.update({"shuffle_prob": self.shuffle_prob})
            return base

        @classmethod
        def from_dict(cls, config: dict):
            return cls(shuffle_prob=config.get("shuffle_prob", 1.0), 
                       fix_first_room=config.get("fix_first_room", False), 
                       fix_last_room=config.get("fix_last_room", False))

    class PointOfViewFilteringFixedOffset(DataAugmentationStrategy):
        def __init__(self, first_room_id_to_keep:int, offset:int):
            super().__init__("PointOfViewFilteringFixedOffset")
            self.first_room_id_to_keep = first_room_id_to_keep
            self.offset = offset

        def apply_augmentation(self, desc_tensors, scene_tensors, scene_image_art_tensors, scene_video_art_tensors):
            """
            method that for each room keep 1 element every self.offset elements starting from the self.first_room_id_to_keep,
            it is implicitly assumed that the images can still represent the full room"""
            scene_tensors = [st[self.first_room_id_to_keep::self.offset] for st in scene_tensors]
            return desc_tensors, scene_tensors, scene_image_art_tensors, scene_video_art_tensors

        def to_dict(self) -> dict:
            base = super().to_dict()
            base.update({
                "first_room_id_to_keep": self.first_room_id_to_keep,
                "offset": self.offset
            })
            return base

        @classmethod
        def from_dict(cls, config: dict):
            return cls(
                first_room_id_to_keep=config["first_room_id_to_keep"],
                offset=config["offset"]
            )
    
    class PointOfViewFilteringFixedOffsetSampling(DataAugmentationStrategy):
        def __init__(self, offset:int):
            super().__init__("PointOfViewFilteringFixedOffsetSampling")
            self.offset = offset

        def apply_augmentation(self, desc_tensors, scene_tensors, scene_image_art_tensors, scene_video_art_tensors):
            """
            method that for each room keep 1 element every self.offset elements starting from the a random position
            betwee 0 and self.offset excluded,
            it is implicitly assumed that the images can still represent the full room"""
            first_room_id_to_keep = math.floor(np.random.random()*self.offset)
            scene_tensors = [st[first_room_id_to_keep::self.offset] for st in scene_tensors]
            return desc_tensors, scene_tensors, scene_image_art_tensors, scene_video_art_tensors

        def to_dict(self) -> dict:
            base = super().to_dict()
            base.update({
                "offset": self.offset
            })
            return base

        @classmethod
        def from_dict(cls, config: dict):
            return cls(
                offset=config["offset"]
            )
    
    class PointOfViewFilteringRandomSampling(DataAugmentationStrategy):
        def __init__(self, n:int):
            super().__init__("PointOfViewFilteringFixedOffsetSampling")
            self.n = n

        def apply_augmentation(self, desc_tensors, scene_tensors, scene_image_art_tensors, scene_video_art_tensors):
            """
            method that for each room keep self.n elements randomly sampled
            in this case the final elements may not represent the full room anymore"""
            scene_tensors = [st[random.sample(range(0, len(st)), self.n)] for st in scene_tensors]
            return desc_tensors, scene_tensors, scene_image_art_tensors, scene_video_art_tensors

        def to_dict(self) -> dict:
            base = super().to_dict()
            base.update({
                "n": self.n
            })
            return base

        @classmethod
        def from_dict(cls, config: dict):
            return cls(
                n=config["n"]
            )
    


    @classmethod
    def strategy_from_dict(cls, config: dict) -> DataAugmentationStrategy:
        strategy_registry = {
            None: cls.NoAugmentation,
            "RoomShuffleAugmentation": cls.RoomShuffleAugmentation,
            "PointOfViewFilteringFixedOffset": cls.PointOfViewFilteringFixedOffset,
            "PointOfViewFilteringFixedOffsetSampling": cls.PointOfViewFilteringFixedOffsetSampling,
            "PointOfViewFilteringRandomSampling": cls.PointOfViewFilteringRandomSampling
        }
        name = config["name"]
        if name not in strategy_registry:
            raise ValueError(f"Unknown strategy: {name}")
        return strategy_registry[name].from_dict(config)



