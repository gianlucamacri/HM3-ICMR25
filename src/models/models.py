import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig

class ByRoomHierearchyModel(nn.Module): # baseline
    def __init__(self, roomModel, roomModelArgsDict, sequenceModel, sequenceModelArgsDict):
        super().__init__()
        self.roomModel = roomModel(**roomModelArgsDict)
        self.sequenceModel = sequenceModel(**sequenceModelArgsDict)

    def forward(self, x_scene, x_scene_list_length, x_image_art = None, x_image_art_list_len = None, x_video_art = None, x_video_list_len = None):
        #x_scene_list_length has shape: bs x max_n_rooms x img_per_room (4), e.g. [[4,4,4,0,0,0]]

        x_by_room = self.roomModel(x_scene, x_scene_list_length, x_image_art, x_image_art_list_len, x_video_art, x_video_list_len)

        #return x_by_room
        # padd to 0 the rooms that are not there (i.e. fake elements added by padded sequence)
        maxRoomN = x_scene_list_length.shape[-1]

        zero_mask = x_scene_list_length == 0
        x_room_length = zero_mask.int().argmax(dim=-1)
        x_room_length[zero_mask.sum(dim=-1) == 0] = maxRoomN  # Handle rows without any zero, assuming we never have empty inputs

        for item_idx in range(x_by_room.shape[0]):
            x_by_room[item_idx, x_room_length[item_idx]:, :] = 0

        if self.sequenceModel.__class__ in {OneDimensionalCNN, MeanPoolingProcessed}:
            x_museum = self.sequenceModel(x_by_room, x_room_length)
        elif self.sequenceModel.__class__ in {GRUNet, LSTMNet}:
            packed_x = torch.nn.utils.rnn.pack_padded_sequence(x_by_room, x_room_length, batch_first=True, enforce_sorted=False)
            x_museum = self.sequenceModel(packed_x)
        else:
            assert False, f'unknown model: {self.sequenceModel.__class__}'

        return x_museum

class OneDimensionalCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_size, image_art_in_channels=None, video_art_in_channels=None):
        super(OneDimensionalCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        
        linearInputSize = out_channels
        if image_art_in_channels is not None:
            self.imageConv = nn.Conv1d(in_channels=image_art_in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            linearInputSize += out_channels
        else:
            self.imageConv = None
        if video_art_in_channels is not None:
            self.videoConv = nn.Conv1d(in_channels=video_art_in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            linearInputSize += out_channels
        else:
            self.videoConv = None

        self.relu = nn.ReLU()
        self.fc = nn.Linear(linearInputSize, feature_size)

    def forward(self, x_scene, x_scene_list_length=None, x_image_art = None, x_image_art_list_len = None, x_video_art = None, x_video_list_len = None):

        def forwardConv(x, convLayer, x_list_length):
            x = x.to(torch.float32)
            x = x.transpose(1,2)
            x1 = convLayer(x)

            # remove the effect of the padding
            if x_list_length is not None:
                if len(x.shape) == 3:
                    for item_idx in range(x.shape[0]):
                        x1[item_idx, x_list_length[item_idx]:, :] = 0
                else:
                    assert False, f"shape {x.shape} not supported"

            x1 = self.relu(x1)
            
            den = x_list_length.to(x1.device).unsqueeze(-1) if x_list_length is not None else  x1.shape[-1]
            x1 = x1.sum(-1) / den # mean pooling across the number of images direction

            x1 = x1.view(x1.size(0), -1)

            return x1
          
        x1 = forwardConv(x_scene, self.conv , x_scene_list_length)

        if self.imageConv:
            x1_image_art = forwardConv(x_image_art, self.imageConv, x_image_art_list_len)
            x1 = torch.cat([x1, x1_image_art], dim=-1)
        
        if self.videoConv:
            x1_video_art = forwardConv(x_video_art, self.videoConv, x_video_list_len)
            x1 = torch.cat([x1, x1_video_art], dim=-1)

        x1_museum = self.fc(x1)
        return x1_museum


class MeanPoolingProcessed(nn.Module): # baseline
    def __init__(self, scene_in_channels, out_channels, image_art_in_channels = None, video_art_in_channels= None, skip_last_linear=False):
        super().__init__()
        self.trf_photo = nn.Linear(scene_in_channels, out_channels)
        self.relu = nn.ReLU()

        mean_in_channels = out_channels
        if image_art_in_channels:
            self.trf_image_art = nn.Linear(image_art_in_channels, out_channels)
            mean_in_channels += out_channels
        else:
            self.trf_image_art = None

        if video_art_in_channels:
            self.trf_video_art = nn.Linear(video_art_in_channels, out_channels)
            mean_in_channels += out_channels
        else:
            self.trf_video_art = None

        self.trf_mean = nn.Linear(mean_in_channels, out_channels)

        self.skip_last_linear = skip_last_linear

    def forward(self, x_scene, x_scene_list_length=None, x_image_art = None, x_image_art_list_len = None, x_video_art = None, x_video_list_len = None):
        
        def forwardAndPool(x, linearLayer, x_list_length):
            x = x.to(torch.float32)
            x1 = linearLayer(x)
            # remove the effect of the padding
            if x_list_length is not None:
                if len(x.shape) == 3:
                    for item_idx in range(x.shape[0]):
                        x1[item_idx, x_list_length[item_idx]:, :] = 0
                elif len(x.shape) == 4:
                    for item_idx in range(x.shape[0]):
                        for room_idx in range(x.shape[1]):
                            x1[item_idx, room_idx, x_list_length[item_idx, room_idx]:, :] = 0
                else:
                    assert False, f"shape {x.shape} not supported"

            x1_img = self.relu(x1)
            
            # bsz, max_n_imgs, ft_size = x1_img.shape
            # list_length_t = torch.tensor(x_list_length, device=x1_img.device) if isinstance(x_list_length, list) else x_list_length.to(x1_img.device)
            list_length_t = x_list_length.to(x1_img.device)
            
            x1_mean = x1_img.sum(-2) / list_length_t.unsqueeze(-1) # mean pooling across the number of images direction

            # set nan to 0
            x1_mean = torch.nan_to_num(x1_mean, nan=0.0)

            return x1_mean
        
        x1_mean = forwardAndPool(x_scene, self.trf_photo ,x_scene_list_length)

        if self.trf_image_art:
            x1_image_art_mean = forwardAndPool(x_image_art, self.trf_image_art, x_image_art_list_len)
            x1_mean = torch.cat([x1_mean, x1_image_art_mean], dim=-1)
        
        if self.trf_video_art:
            x1_video_art_mean = forwardAndPool(x_video_art, self.trf_video_art, x_video_list_len)
            x1_mean = torch.cat([x1_mean, x1_video_art_mean], dim=-1)

        if not self.skip_last_linear:
            x1_museum = self.trf_mean(x1_mean) # process the mean across another linear layer
        else:
            x1_museum = x1_mean

        return x1_museum


# GRU network
class GRUNet(nn.Module):
    def __init__(self, hidden_size, num_features, is_bidirectional=False):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True,
                          bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional

    def forward(self, x):
        x = x.to(torch.float32)
        _output, h_n = self.gru(x)   # h_t is the last t hidden state, _output has all the intermediate outputs as well, not needed
        if self.is_bidirectional:
            return h_n.mean(0)  # take the mean between the 2 directions
        return h_n.squeeze(0)   # batch_size x feature_emb
    
# LSTM network
class LSTMNet(nn.Module):
    def __init__(self, hidden_size, num_features, is_bidirectional=False):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True,
                          bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional

    def forward(self, x):
        x = x.to(torch.float32)
        _output, (h_n, _c_n) = self.lstm(x)  
        if self.is_bidirectional:
            return h_n.mean(0)  # take the mena between the 2 directions
        return h_n.squeeze(0)   # batch_size x feature_emb



class EncodersExecutorConfig(PretrainedConfig): # this inheritance is not technically correct but it is a quick and dirty way to be able to use the useful code that already exists for this class
    
    def __init__(self, strategy_name=None, ee_args=None, uses_flattened_hierarchy:bool=None, uses_audio_video_features:bool=None, **kwargs):
        
        self.strategy_name = strategy_name
        self.ee_args = ee_args
        self.uses_flattened_hierarchy = uses_flattened_hierarchy
        self.uses_audio_video_features = uses_audio_video_features
    
        super().__init__(**kwargs)
    
    # def from_json_file(self, fn):
    #     return (EncodersExecutorConfig) (super().from_json_file(fn))

    def get_strategy_name(self):
        return self.strategy_name
    
    def get_full_strategy_name(self):
        return f'{self.strategy_name}{"_lstm" if self.uses_lstm() else ""}'
    
    def is_hierarchical(self):
        return not self.uses_flattened_hierarchy

    def uses_audio_video(self):
        return self.uses_audio_video_features

    def get_ee_args(self):
        return self.ee_args
    
    def uses_additional_image_features(self):
        if self.is_hierarchical():
            return "image_art_in_channels" in self.get_ee_args()['roomModelArgs'].keys()
        else:
            return "image_art_in_channels" in self.get_ee_args().keys()
    
    def uses_additional_video_features(self):
        if self.is_hierarchical():
            return "video_art_in_channels" in self.get_ee_args()['roomModelArgs'].keys()
        else:
            return "video_art_in_channels" in self.get_ee_args().keys()
    
    def get_input_feature_sizes(self):
        args = self.get_ee_args()
        desc_feature_size = args["desc_input_feature_size"]
        if self.is_hierarchical():
            args = args["roomModelArgs"]
        scene_feature_size = args['scene_in_channels'] if self.is_hierarchical() else args['scene_input_channel']
        scene_image_art_feature_size = args['image_art_in_channels'] if self.uses_additional_image_features() else None
        scene_video_art_feature_size = args['video_art_in_channels'] if self.uses_additional_video_features() else None
        return desc_feature_size, scene_feature_size, scene_image_art_feature_size, scene_video_art_feature_size

    def get_output_size(self):
        return self.get_ee_args()['desc_output_feature_size']

    def uses_lstm(self):
        return self.get_ee_args()['useLSTM']

    def uses_cnn(self):
        if self.is_hierarchical():
            return 'kernel_size' in self.get_ee_args()['scenesSequenceModelArgs'].keys()
        else:
            return 'kernel_size' in self.get_ee_args().keys()

    @classmethod
    def sort_strategies(cls, strategies, reverse:bool=False):
        """sorts stably and in place the strategies

        Args:
            strategies (list[dict]): strategies to be sorted
            reverse (bool, optional): wether to reverse in the opposite order. Defaults to False.
        """
        def key(strategy):
            """extract the key from each strategy, it uses boolean lists as for binary numbers where True is 1 and False is 0, the sorting is stable and inplace
            """
            return (
                    strategy.get_output_size(),
                    strategy.is_hierarchical(),
                    strategy.uses_cnn(),
                    strategy.uses_lstm(),
                    strategy.uses_additional_image_features(),
                    strategy.uses_additional_video_features()
                )
        strategies.sort(key=key, reverse=reverse)



class EncodersExecutor(nn.Module, PyTorchModelHubMixin):

    strategy_names = {'rnn_cnn1d', 'rnn_cnn1d_image_video', 'rnn_meanPoolProc', 'rnn_meanPoolProc_image_video', 'hier_by_room_rnn_meanPoolProc', 'hier_by_room_rnn_meanPoolProc_image_video', 'hier_by_room_rnn_meanPoolProc_cnn1d', 'hier_by_room_rnn_meanPoolProc_cnn1d_image_video',
                      'rnn_cnn1d_video', 'rnn_meanPoolProc_video', 'hier_by_room_rnn_meanPoolProc_video', 'hier_by_room_rnn_meanPoolProc_cnn1d_video',
                      'hier_by_room_rnn_2_meanPoolProc', 'hier_by_room_rnn_2_meanPoolProc_image_video', 'hier_by_room_rnn_2_meanPoolProc_video',
                      'rnn_rnn'} # no pov available for 'rnn_rnn' as it is kinda weird to align videos and povs then

    def __init__(self, config:EncodersExecutorConfig):
        super().__init__()

        self.config = config

        # here i am not sure why the parameter type changes
        try:
            strategy_name = config.strategy_name
            strategy_args = config.ee_args
        except:
            strategy_name =  config['strategy_name']
            strategy_args =  config['ee_args']
        
        assert strategy_name in EncodersExecutor.strategy_names, f"strategy {strategy_name} not defined, choose among {EncodersExecutor.strategy_names}"

        self.strategy_name = strategy_name
        self.strategy_args = strategy_args

        if self.strategy_name == 'rnn_cnn1d':
            if strategy_args['useLSTM']:
                self.model_desc_pov = LSTMNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            else:
                self.model_desc_pov = GRUNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            self.model_pov = OneDimensionalCNN(in_channels=strategy_args['scene_input_channel'], out_channels=strategy_args['scene_output_channels'], kernel_size=strategy_args['kernel_size'], feature_size=strategy_args['scene_output_feature_size'])

        elif self.strategy_name == 'rnn_cnn1d_image_video':
            if strategy_args['useLSTM']:
                self.model_desc_pov = LSTMNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            else:
                self.model_desc_pov = GRUNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            self.model_pov = OneDimensionalCNN(in_channels=strategy_args['scene_input_channel'], out_channels=strategy_args['scene_output_channels'], kernel_size=strategy_args['kernel_size'], feature_size=strategy_args['scene_output_feature_size'], image_art_in_channels=strategy_args['image_art_in_channels'], video_art_in_channels=strategy_args['video_art_in_channels'])

        elif self.strategy_name == 'rnn_cnn1d_video':
            if strategy_args['useLSTM']:
                self.model_desc_pov = LSTMNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            else:
                self.model_desc_pov = GRUNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            self.model_pov = OneDimensionalCNN(in_channels=strategy_args['scene_input_channel'], out_channels=strategy_args['scene_output_channels'], kernel_size=strategy_args['kernel_size'], feature_size=strategy_args['scene_output_feature_size'], image_art_in_channels=None, video_art_in_channels=strategy_args['video_art_in_channels'])

                
        elif self.strategy_name == 'rnn_meanPoolProc':
            if strategy_args['useLSTM']:
                self.model_desc_pov = LSTMNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            else:
                self.model_desc_pov = GRUNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            self.model_pov = MeanPoolingProcessed(scene_in_channels=strategy_args['scene_input_channel'], out_channels=strategy_args['scene_output_channels'])
        
        elif self.strategy_name == 'rnn_meanPoolProc_image_video':
            if strategy_args['useLSTM']:
                self.model_desc_pov = LSTMNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            else:
                self.model_desc_pov = GRUNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            self.model_pov = MeanPoolingProcessed(scene_in_channels=strategy_args['scene_input_channel'], out_channels=strategy_args['scene_output_channels'], image_art_in_channels=strategy_args['image_art_in_channels'], video_art_in_channels=strategy_args['video_art_in_channels'])
        
        elif self.strategy_name == 'rnn_meanPoolProc_video':
            if strategy_args['useLSTM']:
                self.model_desc_pov = LSTMNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            else:
                self.model_desc_pov = GRUNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            self.model_pov = MeanPoolingProcessed(scene_in_channels=strategy_args['scene_input_channel'], out_channels=strategy_args['scene_output_channels'], image_art_in_channels=None, video_art_in_channels=strategy_args['video_art_in_channels'])   

        elif self.strategy_name == 'rnn_rnn':
            if strategy_args['useLSTM']:
                self.model_desc_pov = LSTMNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
                self.model_pov = LSTMNet(hidden_size=strategy_args['scene_output_channels'], num_features=strategy_args['scene_input_channel'], is_bidirectional=strategy_args['scene_model_is_bidirectional'])
            else:
                self.model_desc_pov = GRUNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
                self.model_pov = GRUNet(hidden_size=strategy_args['scene_output_channels'], num_features=strategy_args['scene_input_channel'], is_bidirectional=strategy_args['scene_model_is_bidirectional'])
            

        elif self.strategy_name in {'hier_by_room_rnn_meanPoolProc', 'hier_by_room_rnn_meanPoolProc_image_video', 'hier_by_room_rnn_meanPoolProc_video'}:
            
            if self.strategy_name == 'hier_by_room_rnn_meanPoolProc_video':
                assert "image_art_in_channels" not in strategy_args['roomModelArgs'], f"strategy {self.strategy_name} uses video only but 'image_art_in_channels' arg was given"

            
            if strategy_args['useLSTM']:
                self.model_desc_pov = LSTMNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
                self.model_pov = ByRoomHierearchyModel(
                    MeanPoolingProcessed,
                    strategy_args['roomModelArgs'],
                    LSTMNet,
                    strategy_args['scenesSequenceModelArgs'],
                )
            else:
                self.model_desc_pov = GRUNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
                self.model_pov = ByRoomHierearchyModel(
                    MeanPoolingProcessed,
                    strategy_args['roomModelArgs'],
                    GRUNet,
                    strategy_args['scenesSequenceModelArgs'],
                )


        elif self.strategy_name in {'hier_by_room_rnn_2_meanPoolProc', 'hier_by_room_rnn_2_meanPoolProc_image_video', 'hier_by_room_rnn_2_meanPoolProc_video'}:
            
            if self.strategy_name == 'hier_by_room_rnn_2_meanPoolProc_video':
                assert "image_art_in_channels" not in strategy_args['roomModelArgs'], f"strategy {self.strategy_name} uses video only but 'image_art_in_channels' arg was given"

            
            
            self.model_desc_pov = GRUNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            self.model_pov = ByRoomHierearchyModel(
                MeanPoolingProcessed,
                strategy_args['roomModelArgs'],
                MeanPoolingProcessed,
                strategy_args['scenesSequenceModelArgs'],
            )

        
        elif self.strategy_name in {'hier_by_room_rnn_meanPoolProc_cnn1d', 'hier_by_room_rnn_meanPoolProc_cnn1d_image_video', 'hier_by_room_rnn_meanPoolProc_cnn1d_video'}:
    
            self.model_desc_pov = LSTMNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional']) if strategy_args['useLSTM'] \
                else GRUNet(hidden_size=strategy_args['desc_output_feature_size'], num_features=strategy_args['desc_input_feature_size'], is_bidirectional=strategy_args['desc_model_is_bidirectional'])
            
            if self.strategy_name == 'hier_by_room_rnn_meanPoolProc_cnn1d_video':
                assert "image_art_in_channels" not in strategy_args['roomModelArgs'], f"strategy {self.strategy_name} uses video only but 'image_art_in_channels' arg was given"

            self.model_pov = ByRoomHierearchyModel(
                MeanPoolingProcessed,
                strategy_args['roomModelArgs'],
                OneDimensionalCNN,
                strategy_args['scenesSequenceModelArgs'],
            )

        
        
        
        else:
            assert False, f"unimplemented rule for {self.strategy_name}" 

    
    def getConfig(self):
        return self.config

    def getModelParams(self):
        return list(self.model_desc_pov.parameters()) + list(self.model_pov.parameters())

    def getModelParamsDict(self):
        return {'model_desc_pov':self.model_desc_pov.parameters(),
                'model_pov':self.model_pov.parameters()}

    def setTrain(self):
        self.model_desc_pov.train()
        self.model_pov.train()

    def setEval(self):
        self.model_desc_pov.eval()
        self.model_pov.eval()
    
            

    def getModelClassesDict(self):
        if self.strategy_name in {'rnn_cnn1d', 'rnn_cnn1d_image_video', 'rnn_cnn1d_video'}:
            return {
                "model_desc_pov": LSTMNet if self.strategy_args['useLSTM'] else GRUNet,
                "model_pov": OneDimensionalCNN,
            }
        elif self.strategy_name in {'rnn_meanPoolProc', 'rnn_meanPoolProc_image_video', 'rnn_meanPoolProc_video'} :
            return {
                "model_desc_pov": LSTMNet if self.strategy_args['useLSTM'] else GRUNet,
                "model_pov": MeanPoolingProcessed,
            }
        elif self.strategy_name in {'hier_by_room_rnn_meanPoolProc', 'hier_by_room_rnn_meanPoolProc_image_video', 'hier_by_room_rnn_meanPoolProc_cnn1d', 'hier_by_room_rnn_meanPoolProc_cnn1d_image_video', 'hier_by_room_rnn_meanPoolProc_video', 'hier_by_room_rnn_meanPoolProc_cnn1d_video', 'hier_by_room_rnn_2_meanPoolProc', 'hier_by_room_rnn_2_meanPoolProc_image_video', 'hier_by_room_rnn_2_meanPoolProc_video'}:
            return {
                "model_desc_pov": LSTMNet if self.strategy_args['useLSTM'] else GRUNet,
                "model_pov": ByRoomHierearchyModel,
            }
        elif self.strategy_name in {'rnn_rnn'}:
            return {
                "model_desc_pov": LSTMNet if self.strategy_args['useLSTM'] else GRUNet,
                "model_pov": LSTMNet if self.strategy_args['useLSTM'] else GRUNet,
            }
        else:
            assert False, f"unimplemented rule for {self.strategy_name}" 
        


    def forward(self, batch): # batch should be a tuple (descs_pov, padded_scene_pov, padded_imageart, padded_videoart, indexes, len_scene_pov, len_imageart, len_videoart)
        descs_pov, padded_scene_pov, padded_imageart, padded_videoart, indexes, len_scene_pov, len_imageart, len_videoart = batch

        if self.strategy_name in {'rnn_cnn1d', 'rnn_meanPoolProc', 'hier_by_room_rnn_meanPoolProc', 'hier_by_room_rnn_meanPoolProc_cnn1d', 'hier_by_room_rnn_2_meanPoolProc'}:
            output_desc_pov = self.model_desc_pov(descs_pov)
            output_pov = self.model_pov(padded_scene_pov, len_scene_pov)
        

        elif self.strategy_name in {'rnn_cnn1d_image_video', 'rnn_meanPoolProc_image_video','hier_by_room_rnn_meanPoolProc_image_video', 'hier_by_room_rnn_meanPoolProc_cnn1d_image_video', 'hier_by_room_rnn_2_meanPoolProc_image_video'}:
            output_desc_pov = self.model_desc_pov(descs_pov)
            output_pov = self.model_pov(padded_scene_pov, len_scene_pov, padded_imageart, len_imageart, padded_videoart, len_videoart)
        
        elif self.strategy_name in {'rnn_cnn1d_video', 'rnn_meanPoolProc_video','hier_by_room_rnn_meanPoolProc_video', 'hier_by_room_rnn_meanPoolProc_cnn1d_video', 'hier_by_room_rnn_2_meanPoolProc_video'}:
            output_desc_pov = self.model_desc_pov(descs_pov)
            output_pov = self.model_pov(padded_scene_pov, len_scene_pov, None, None, padded_videoart, len_videoart) # set to None unncecessary args
        
        elif self.strategy_name in {'rnn_rnn'}:
            output_desc_pov = self.model_desc_pov(descs_pov)
            packed_pov = torch.nn.utils.rnn.pack_padded_sequence(padded_scene_pov, len_scene_pov, batch_first=True, enforce_sorted=False)
            output_pov = self.model_pov(packed_pov)

        else:
            assert False, f"unimplemented rule for {self.strategy_name}" 

        return output_desc_pov, output_pov # description and museums output features
    


