import torch
import pickle
from tqdm.auto import tqdm
import os
import inspect
from enum import Enum, auto
import logging


class TextSplitStrategy(Enum):
    NONE = auto()
    BY_SENTENCE = auto()
    BY_MAX_LENGTH = auto()

class TextExtractor(Enum):
    CLIP_VIT_BASE_32 = 512
    OPEN_CLIP = 512

class ImageExtractor(Enum):
    CLIP_VIT_BASE_32 = 512
    OPEN_CLIP = 512
    NONE = 512

class VideoExtractor(Enum):
    C4C_VIT_BASE_32 = 512
    NONE = 512

class FeatureExtractor():
    
    CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
    RELATIVE_FEATURE_FOLDER = 'features'


    def __init__(self, feature_folder_name:str, text_extraction_model:TextExtractor, image_extraction_model:ImageExtractor, video_extraction_model:VideoExtractor, logger=None):

        if logger == None:
            logger = logging

        abs_feature_path = os.path.join(self.__get_class_path(), self.__class__.RELATIVE_FEATURE_FOLDER, feature_folder_name)
        if os.path.exists(abs_feature_path):
            assert os.path.isdir(abs_feature_path), f'path {abs_feature_path} is not a directory'
        else:
            logger.info(f'creating feature dir at: {abs_feature_path}')
            os.makedirs(abs_feature_path)
        self.feature_path = abs_feature_path 

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'        
        logger.info(f'using device: {self.device}')

        if text_extraction_model == TextExtractor.CLIP_VIT_BASE_32:
            self.__set_clip_text_extractor(TextExtractor.CLIP_VIT_BASE_32)
        else:
            raise ValueError(f"unknown text extractor: {text_extraction_model.name}")
        
        if image_extraction_model == ImageExtractor.CLIP_VIT_BASE_32:
            self.__set_clip_image_extractor(ImageExtractor.CLIP_VIT_BASE_32)
        else:
            raise ValueError(f"unknown text extractor: {image_extraction_model.name}")
        
        if video_extraction_model == VideoExtractor.C4C_VIT_BASE_32:
            self.videoExtractorFeatureSize=VideoExtractor.C4C_VIT_BASE_32.value
            self.videoExtractorName = 'C4C_VIT_BASE_32'
        else:
            raise ValueError(f"unknown text extractor: {image_extraction_model.name}")
        

        
    def get_feature_sizes(self):
        return {
            'text':self.textExtractorFeatureSize,
            'image':self.imageExtractorFeatureSize,
            'video':self.videoExtractorFeatureSize
        }
        

    def get_feature_path(self):
        return self.feature_path
    
    def __get_class_path(self):
        return os.path.dirname(inspect.getfile(self.__class__))


    def __set_clip_image_extractor(self, imageExtractorName:ImageExtractor):
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        imageExtractor = CLIPVisionModelWithProjection.from_pretrained(self.__class__.CLIP_MODEL_NAME)
        imageExtractor = imageExtractor.to(self.device)
        imagePreprocessor = CLIPImageProcessor.from_pretrained(self.__class__.CLIP_MODEL_NAME)
        
        self.imageExtractor = imageExtractor
        self.imagePreprocessor = imagePreprocessor
        self.imageExtractorName = imageExtractorName.name
        self.imageExtractorFeatureSize = imageExtractorName.value


    def __set_clip_text_extractor(self, textExtractorName:TextExtractor):
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        textExtractor = CLIPTextModelWithProjection.from_pretrained(self.__class__.CLIP_MODEL_NAME)
        textExtractor = textExtractor.to(self.device) 
        textTokenizer = AutoTokenizer.from_pretrained(self.__class__.CLIP_MODEL_NAME)
        
        self.textExtractor = textExtractor
        self.textTokenizer = textTokenizer
        self.textExtractorName = textExtractorName.name
        self.textExtractorFeatureSize = textExtractorName.value
        

    def extract_and_store_museum_descriptions_features(self, museumsDataset, split_strategy:TextSplitStrategy=None, stride:int = 0):

        descFeatures = self.extract_museum_descritption_features(museumsDataset, split_strategy=split_strategy, stride=stride)

        if split_strategy is None or split_strategy is TextSplitStrategy.NONE:
            splitStratName = ''
        elif split_strategy == TextSplitStrategy.BY_SENTENCE:
            splitStratName = 'SentenceLevelSplit'
        elif split_strategy == TextSplitStrategy.BY_MAX_LENGTH:
            splitStratName = f'MaxLenSplit_s{stride}'
        else:
            raise ValueError(f"Unknown implementation for TextSplitStrategy: {split_strategy.name}")

        out_fn = f"museumDesc{splitStratName}_{self.textExtractorName}.pickle"

        with open(os.path.join(self.feature_path, out_fn), 'wb') as f:
            pickle.dump(descFeatures, f)

        return out_fn
    

    def extract_museum_descritption_features(self, museumsDataset, split_strategy:TextSplitStrategy=None, stride:int = 0):
        
        # extract museum description data

        descs = museumsDataset.loadDesc()
        descFeatures = {}

        assert self.textExtractor is not None, "textExtractor has not been set"
        self.textExtractor.eval()
        with torch.no_grad():
            for id, desc in tqdm(descs.items()):
                if split_strategy is None:
                    inputs = self.textTokenizer(museumsDataset.flattenDescriptionList(desc), padding='max_length', truncation=True, return_tensors="pt")
                elif split_strategy == TextSplitStrategy.BY_SENTENCE:
                    inputs = self.textTokenizer(museumsDataset.flattenDescriptionList(desc), padding='max_length', truncation=True, return_tensors="pt")
                elif split_strategy == TextSplitStrategy.BY_MAX_LENGTH:
                    inputs = self.textTokenizer(' '.join(museumsDataset.flattenDescriptionList(desc)), padding='max_length', truncation=True,  return_tensors="pt" , return_overflowing_tokens=True, stride=stride)
                    _ = inputs.pop('overflow_to_sample_mapping')
                else:
                    raise ValueError(f"Unknown implementation for TextSplitStrategy: {split_strategy.name}")
                inputs=inputs.to(self.device)
                outputs = self.textExtractor(**inputs)
                descFeatures[id] = outputs['text_embeds'].detach().to('cpu')


        return descFeatures
    

    def extract_and_store_museum_image_features(self, museumsDataset):
        scenesFeatures = self.extract_museum_image_features(museumsDataset)
        out_fn = f'museumSceneFeatures_{self.textExtractorName}.pickle'

        with open(os.path.join(self.feature_path, out_fn), 'wb') as f:
            pickle.dump(scenesFeatures, f)

        return out_fn

    
    def extract_museum_image_features(self, museumsDataset):
        
        # extract museum description data

        scenesFeatures = {}
        self.imageExtractor.eval()
        with torch.no_grad():
            for id in tqdm(museumsDataset.getMuseumList()):
                museumscenesFeatures = []

                for roomScreens in museumsDataset.getMuseumScreenshots(id):
                    inputs = self.imagePreprocessor(roomScreens, return_tensors="pt").to(self.device)
                    outputs = self.imageExtractor(**inputs)
                    museumscenesFeatures.append(outputs['image_embeds'].detach().to('cpu'))

                scenesFeatures[id] = museumscenesFeatures

        return scenesFeatures
    

    def extract_and_store_semart_image_features(self, semartDataset):
        semartFeatures = self.extract_semart_image_features(semartDataset)
        out_fn = f'semartFeatures_{self.textExtractorName}.pickle'

        with open(os.path.join(self.feature_path, out_fn), 'wb') as f:
            pickle.dump(semartFeatures, f)
        
        return out_fn
    

    def extract_semart_image_features(self, semartDataset):
        
        # extract semart  image features
        semartFeatures = {}
        self.imageExtractor.eval()
        with torch.no_grad():
            for id in tqdm(semartDataset.getAllArtWorkIds()):

                inputs = self.imagePreprocessor(semartDataset.getArtWorkImage(id), return_tensors="pt").to(self.device)
                outputs = self.imageExtractor(**inputs)

                semartFeatures[id] = outputs['image_embeds'].detach().to('cpu')

        return semartFeatures
    
