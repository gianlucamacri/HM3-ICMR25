import json
import os
import inspect
from src.data.feature_extractor import TextSplitStrategy, TextExtractor, ImageExtractor, VideoExtractor

class DigitalExhibitionFeatureSet():

    FS_DIRNAME = 'featureSets'

    def __init__(self, dir_name, feature_set_fn, fn_dict=None):
        base_path =  os.path.join(self.__get_class_path(), 'features')
        if not feature_set_fn.endswith('.json'):
            feature_set_fn += '.json'
        self.fn_path = os.path.join(base_path, dir_name, __class__.FS_DIRNAME, feature_set_fn)
        #assert not os.path.exists(self.fn_path), f"path {self.fn_path} already exists, please use the load method instead"
        os.makedirs(os.path.dirname(self.fn_path), exist_ok=True)
        if fn_dict is None:
            self.fn_dict = {}
        else:
            self.fn_dict = fn_dict

    def get_image_art(self, full_path = False):
        try:
            data = self.fn_dict['image_art']
        except KeyError:
            return None
        data['data_fn'] = self.__get_out_path(data['data_fn'], full_path)
        return data

    def get_video_art(self, full_path = False):
        try:
            data = self.fn_dict['video_art']
        except KeyError:
            return None
        data['data_fn'] = self.__get_out_path(data['data_fn'], full_path)
        return data

    def get_povs(self, full_path = False):
        try:
            data = self.fn_dict['povs']
        except KeyError:
            return None
        data['data_fn'] = self.__get_out_path(data['data_fn'], full_path)
        return data

    def get_descriptions(self, full_path = False):
        try:
            data = self.fn_dict['descriptions']
        except KeyError:
            return None
        data['data_fn'] = self.__get_out_path(data['data_fn'], full_path)
        return data
    
    def get_feature_sizes(self):
        return  self.fn_dict['descriptions']['feature_size'],\
                self.fn_dict['povs']['feature_size'],\
                self.fn_dict['image_art']['feature_size'],\
                self.fn_dict['video_art']['feature_size']


    def add_and_store_image_art(self, fn, extractor:ImageExtractor):
        self.fn_dict['image_art'] = {
            'data_fn':fn,
            'extractor':extractor.name,
            'feature_size':extractor.value,
        }
        self.__store()
        
    def add_and_store_video_art(self, fn, extractor:VideoExtractor):
        self.fn_dict['video_art'] = {
            'data_fn':fn,
            'extractor':extractor.name,
            'feature_size':extractor.value,
        }
        self.__store()

    def add_and_store_povs(self, fn, extractor:ImageExtractor):
        self.fn_dict['povs'] = {
            'data_fn':fn,
            'extractor':extractor.name,
            'feature_size':extractor.value,
        }
        self.__store()

    def add_and_store_descriptions(self, fn, extractor:TextExtractor, split_strategy:TextSplitStrategy):
        self.fn_dict['descriptions'] = {
            'data_fn':fn,
            'extractor':extractor.name,
            'feature_size':extractor.value,
            'split_strategy':split_strategy.name,
        }
        self.__store()

    def set_dataset_name(self, dataset_name:str):
        assert dataset_name in {'digitalExps', 'generatedDigitalExps', 'museums3k'}
        self.fn_dict['dataset_name'] = dataset_name

    def get_dataset_name(self):
        return self.fn_dict['dataset_name']

    @classmethod
    def loadFeatureSet(cls, dir_name, feature_set_fn):
        feature_set = DigitalExhibitionFeatureSet(dir_name, feature_set_fn)
        with open(feature_set.fn_path) as f:
            fn_dict = json.load(f)
        feature_set.fn_dict = fn_dict
        return feature_set
        
    def __store(self):
        with open(self.fn_path, 'w') as f:
            json.dump(self.fn_dict, f, indent=4)
        
    def __get_out_path(self, file_fn:str, full_path:bool):
        if not full_path:
            return file_fn
        
        return os.path.join(os.path.dirname(self.fn_path), '..', file_fn)

    def __get_class_path(self):
        return os.path.dirname(inspect.getfile(self.__class__))



