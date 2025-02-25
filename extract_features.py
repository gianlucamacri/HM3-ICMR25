import argparse
from src.data.feature_extractor import *
from src.data.feature_utils import DigitalExhibitionFeatureSet
from src.data.datasets.SemArt.utils import SemArtDataset

# from data.datasets.artistic_video_dataset.utils import ArtisticVideoDataset
# from data.datasets.SemArtVideoArtGen.utils import SemArtVideoArtGen

def main(args):
    
    textExtractor = TextExtractor.CLIP_VIT_BASE_32
    imageExtractor = ImageExtractor.CLIP_VIT_BASE_32
    videoExtractor = VideoExtractor.C4C_VIT_BASE_32

    extractor = FeatureExtractor(args.feature_output_dir_name, textExtractor, imageExtractor, videoExtractor)

    imageArtDataset = SemArtDataset()

    if args.dataset == 'digitalExps':
        from src.data.datasets.artExhibitionDataset.utils import ArtExhibitionDataset
        expositionsDataset = ArtExhibitionDataset('metadata_no_accents')
    elif args.dataset == 'generatedDigitalExps':
        from src.data.datasets.digitalMuseumsSemArtVideoGenDataset.utils import DigitalMuseumsSemArtVideoGenDataset
        expositionsDataset = DigitalMuseumsSemArtVideoGenDataset('metadata')
    
    feature_set = DigitalExhibitionFeatureSet(args.feature_output_dir_name, args.feature_set_name)
    feature_set.set_dataset_name(args.dataset)

    split_strategy = None
    if args.text_split_strategy == 'sentence_by_sentence':
        split_strategy = TextSplitStrategy.BY_SENTENCE
    elif args.text_split_strategy == 'max_len':
        split_strategy = TextSplitStrategy.BY_MAX_LENGTH
        assert args.text_stride >= 0, f'text_stride must be non negative, given {args.text_stride}'
    else:
        assert False, f'unknown {split_strategy}'
    descriptions_fn = extractor.extract_and_store_museum_descriptions_features(expositionsDataset, split_strategy, args.text_stride)
    feature_set.add_and_store_descriptions(descriptions_fn, textExtractor, split_strategy)
    
    povs_fn = extractor.extract_and_store_museum_image_features(expositionsDataset)
    feature_set.add_and_store_povs(povs_fn, imageExtractor)
    
    image_art_fn = extractor.extract_and_store_semart_image_features(imageArtDataset)
    feature_set.add_and_store_image_art(image_art_fn, imageExtractor)

    if args.video_feature_fn is not None:
        feature_set.add_and_store_video_art(args.video_feature_fn, videoExtractor)


    # for the videos the thing is a little bit trickier as it is using the code form either the c4c or the eclipse repository,
    # so it will stay the way it was for a while
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--feature_output_dir_name', type=str, required=True, 
                        help="Name of the output directory for the features")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['digitalExps', 'generatedDigitalExps'], 
                        help="Dataset to extract the features for")
    parser.add_argument('--feature_set_name', type=str, required=True, 
                        help="Filename for the feature_set output")
    parser.add_argument('--text_split_strategy', type=str, required=True,
                        choices=['sentence_by_sentence', 'max_len'])
    parser.add_argument('--text_stride', type=int, required=False, default=0,
                        help='number of overlapping tokens')
    
    parser.add_argument('--video_feature_fn', type=str, required=False, 
                        help="video feature filename")
    

    
    args = parser.parse_args()

    main(args)

    # todo: prevent overwriting of old features/check if they are already present with an attribute
