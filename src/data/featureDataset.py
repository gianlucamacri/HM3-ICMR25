
from src.data.feature_utils import DigitalExhibitionFeatureSet
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pickle
from itertools import chain


class ArtExhibitionFeatureDataset(Dataset):

    # to do add ignore the lobby for images and videos ? 

    def __init__(self, feature_path, featuresFn,  split, flatten_room_hierarchy:bool, exhibitionsDataset):

        feature_set = DigitalExhibitionFeatureSet.loadFeatureSet(feature_path, featuresFn)

        self.flatten_room_hierarchy = flatten_room_hierarchy
        
        #assert self.flatten_room_hierarchy, f"unflatteded hierarchy is still in developement"

        self.feature_set = feature_set

        self.description_path = feature_set.get_descriptions(full_path=True)['data_fn']
        self.data_scene_path = feature_set.get_povs(full_path=True)['data_fn']
        self.data_image_art_path = feature_set.get_image_art(full_path=True)['data_fn']
        self.data_video_art_path = feature_set.get_video_art(full_path=True)['data_fn']

        with open(self.description_path, 'rb') as f:
            self.description_feature_dict = pickle.load(f)

        with open(self.data_scene_path, 'rb') as f:
            self.scene_feature_dict = pickle.load(f)

        with open(self.data_image_art_path, 'rb') as f:
            self.image_artworks_feature_dict = pickle.load(f)

        with open(self.data_video_art_path, 'rb') as f:
            self.video_artworks_feature_dict = pickle.load(f)['features']

        def get_museum_category_from_name(name:str) -> str:
            return '_'.join(name.split('_')[-2:])

        museum_types = sorted({get_museum_category_from_name(id) for id in exhibitionsDataset.getMuseumList(None)})
        self.all_museum_category_str_idx_map = {t:idx for idx,t in enumerate(museum_types)}
        
        self.split_ids = exhibitionsDataset.getMuseumList(split)

        def flatten_chain(els):
            return list(chain(*els))
        
        self.description_feature_dict = {k:v for k, v in self.description_feature_dict.items() if k in self.split_ids}
        self.scene_feature_dict = {k:(v if not flatten_room_hierarchy else flatten_chain(v)) for k, v in self.scene_feature_dict.items() if k in self.split_ids}

        self.split_id_idx_map = {}
        self.image_ids_scene_dict = {} 
        self.video_ids_scene_dict = {}
        for id in self.split_ids:
            image_ids, video_ids = exhibitionsDataset.getMuseumArtWorksPerRoom(id)
            self.image_ids_scene_dict[id] = image_ids if not flatten_room_hierarchy else flatten_chain(image_ids)
            self.video_ids_scene_dict[id] = video_ids if not flatten_room_hierarchy else flatten_chain(video_ids)
            self.split_id_idx_map[id] = self.all_museum_category_str_idx_map[get_museum_category_from_name(id)]

    def __len__(self):
        return len(self.split_ids)

    def __getitem__(self, index):

        id = self.split_ids[index]
        category = self.split_id_idx_map[id]

        desc_tensor = self.description_feature_dict[id]

        if self.flatten_room_hierarchy:
            scene_tensor = torch.stack(self.scene_feature_dict[id])
            scene_image_art_tensor = torch.stack([self.image_artworks_feature_dict[id].squeeze() for id in self.image_ids_scene_dict[id]]) if len(self.image_ids_scene_dict[id]) != 0 else None
            scene_video_art_tensor = torch.stack([self.video_artworks_feature_dict[id] for id in self.video_ids_scene_dict[id]]) if len(self.video_ids_scene_dict[id]) != 0 else None
            
        else:
            scene_tensor = self.scene_feature_dict[id]
            
            # todo fix missing images and videos case

            scene_image_art_tensor = [[self.image_artworks_feature_dict[id].squeeze() for id in els] for els in self.image_ids_scene_dict[id]]
            scene_image_art_tensor = [torch.stack(els) if len(els) != 0 else None for els in scene_image_art_tensor]
            
            scene_video_art_tensor = [[self.video_artworks_feature_dict[id] for id in els] for els in self.video_ids_scene_dict[id]]
            scene_video_art_tensor = [torch.stack(els) if len(els) != 0 else None for els in scene_video_art_tensor]

            assert len(scene_tensor) == len(scene_image_art_tensor)
            assert len(scene_tensor) == len(scene_video_art_tensor)

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

 