"""
This script runs inference on the taxonomy classification of events task.

Note subtaxonomy classification is only performed on taxonomy 173 at the moment.

"""

# imports
import config
import models
import dataset
import utils
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader


def inference():
    # load data
    print("Loading data...")
    data = pd.read_csv(config.INFERENCE_FILE, index_col=0, encoding='utf-8')
    print(f"Data loaded. Number of examples: {data.shape[0]}")

    ###########################
    # Taxonomy Classification #
    ###########################

    # cleaning and preprocessing for taxonomy model
    print("Preprocessing data...")
    preprocessed_data = utils.preprocessing_a1(data).reset_index(drop=True)
    # create torch dataset and dataloader for taxonomy model
    inference_set = dataset.DistilBertDataset(preprocessed_data, config.TOKENIZER, config.TAXONOMY_MAX_LEN)
    data_loader = DataLoader(inference_set, **config.PARAMS)
    print("Data preprocessed and loaded on Dataloader.")

    # load taxonomy model
    print("Classifying taxonomies...")
    taxonomy_model = models.DistilBERTClass()
    taxonomy_model.eval()
    if config.DEVICE == 'gpu':
        taxonomy_model.load_state_dict(torch.load(config.TAXONOMY_MODEL_PATH))
        taxonomy_model.to(torch.device(config.DEVICE))
    else:
        taxonomy_model.load_state_dict(torch.load(config.TAXONOMY_MODEL_PATH, map_location=torch.device(config.DEVICE)))

    # taxonomy inference
    taxonomies = utils.get_inference(data_loader, config.DEVICE, taxonomy_model)
    print("Taxonomy classification finished.")

    ############################################
    # Taxonomy 173: Subtaxonomy Classification #
    ############################################
    # get subset of the original data classified as taxonomy 173 (Encoded to 4)
    print("Filtering events classified as taxonomy 173...")
    indexes = np.where(taxonomies == 4)[0].tolist()
    preprocessed_data_subset = preprocessed_data.iloc[indexes]
    # create torch dataset and dataloader for subtaxonomy model
    inference_set = dataset.DistilBertDataset(preprocessed_data_subset, config.TOKENIZER, config.SUBTAXONOMY_MAX_LEN)
    data_loader = DataLoader(inference_set, **config.PARAMS)
    print("Events classified as taxonomy 173 loaded on Dataloader for subtaxonomy classification.")

    # load subtaxonomy model
    print("Classifying subtaxonomies...")
    subtaxonomy_model = models.DistilBERTClassSubtaxonomy()
    subtaxonomy_model.eval()
    if config.DEVICE == 'gpu':
        subtaxonomy_model.load_state_dict(torch.load(config.SUBTAXONOMY_MODEL_PATH))
        subtaxonomy_model.to(torch.device(config.DEVICE))
    else:
        subtaxonomy_model.load_state_dict(
            torch.load(config.SUBTAXONOMY_MODEL_PATH, map_location=torch.device(config.DEVICE))
        )

    # subtaxonomy inference
    subtaxonomies = utils.get_inference(data_loader, config.DEVICE, subtaxonomy_model).tolist()
    print("Subtaxonomy classification finished.")

    #######################
    # Put it all together #
    #######################

    inferences = dict()
    for idx, taxonomy in enumerate(taxonomies):
        if idx in indexes:
            inferences[idx] = {
                'taxonomy': utils.decode_taxonomy(taxonomy), 'subtaxonomy': utils.decode_subtaxonomy(subtaxonomies[0])
            }
            # remove first element
            subtaxonomies.pop(0)
        else:
            inferences[idx] = {'taxonomy': utils.decode_taxonomy(taxonomy), 'subtaxonomy': np.nan}

    print("Results:")
    print(inferences)
    return inferences


if __name__ == "__main__":
    inference()