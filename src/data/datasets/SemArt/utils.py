import pandas as pd
import os
import re
import html
from PIL import Image


class SemArtDataset():

    availableSplits = {'test','val','train'}

    __package_dir = os.path.dirname(os.path.abspath(__file__))

    __images_dir = 'Images'

    def __init__(self):
        
        self.idsPerSplit = {s:[] for s in SemArtDataset.availableSplits}
        dfs = []
        for s in SemArtDataset.availableSplits:
            df = pd.read_csv(os.path.join(SemArtDataset.__package_dir , f'semart_{s}.csv'), sep='\t', encoding='latin-1')
            dfs.append(df)
        self.data = pd.concat(dfs, ignore_index=True)

        # Apply the html.unescape to the relevant columns
        self.data['DESCRIPTION'] = self.data['DESCRIPTION'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)


    def getAllArtWorkIds(self):
        return list(self.data['IMAGE_FILE'])

    def getArtworkDescription(self, id):
        desc = self.data.loc[self.data['IMAGE_FILE']==id, 'DESCRIPTION'].iloc[0]
        #print(f'semart: {desc}')
        return desc
    
    def getArtWorkImage(self, id):
        fn = os.path.join(SemArtDataset.__package_dir, SemArtDataset.__images_dir, id)
        return Image.open(fn).convert("RGB")
