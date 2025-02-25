import torch
import os
import json
from src.models.models import EncodersExecutor, EncodersExecutorConfig

def cosine_sim(im, s):
    '''cosine similarity between all the image and sentence pairs
    '''
    inner_prod = im.mm(s.t())
    im_norm = torch.sqrt((im ** 2).sum(1).view(-1, 1) + 1e-18)
    s_norm = torch.sqrt((s ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)
    return sim

def get_hf_full_name(hf_user_name, model_name):
    return f"{hf_user_name}/{model_name}"

def get_model_output_dir(model_name):
    """return the model output path wrt the base project dir (the main dir)"""
    return os.path.join('outputs', 'models', model_name)

class PretrainedModelsCollection():

    def __init__(self, data):
        self.type = data['type']
        assert self.type in {'hf', 'local'}
        self.models_locations = data['models_locations'] # wrt the base dir if local
    
    @classmethod
    def load_pretrained_collection(cls, fn):
        with open(fn) as f:
            data = json.load(f)
        assert 'type' in data.keys()
        assert 'models_locations' in data.keys()
        return PretrainedModelsCollection(data)
    
    def get_models_num(self):
        return len(self.models_locations)
    
    def load_ith_model(self, i):
        model_name = self.models_locations[i]
        config = EncodersExecutorConfig.from_pretrained(model_name)
        model = EncodersExecutor.from_pretrained(model_name, config=config)
        return model, config

    def get_ith_uid(self, i):
        return self.models_locations[i].split('/')[-1].split('_')[1]