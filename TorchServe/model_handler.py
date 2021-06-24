from abc import ABC
import json
import csv
import logging
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import DistilBertTokenizer

from model import DistilBERTClass

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s",transformers.__version__)

#################
# Torch dataset #
#################

class DistilBertDatasetInference(Dataset):
    def __init__(self, input_text, tokenizer, max_len):
        self.len = len(input_text)
        self.data = input_text
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.data[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long)
        }

    def __len__(self):
        return self.len


class DistilBERTMultiClassifierHandler(BaseHandler, ABC):
    """
    DistilBERT handler class for Multiclass Classification.
    """

    def __init__(self):
        super(DistilBERTMultiClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the DistilBERT model is loaded and
        initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_bin_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        # Loading the model and tokenizer from checkpoint and config files
        self.model = DistilBERTClass()
        self.model.load_state_dict(torch.load(model_bin_path, map_location=self.device))
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        self.model.eval()

        logger.info(
            "Transformer model from path %s loaded successfully", model_dir
        )

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning("Missing the index_to_name.json file.")
        self.initialized = True

    #################################################
    # text processing and cleaning helper functions #
    #################################################

    def is_ascii(self, w):
        try:
            w.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    def text_cleaning(self, text):
        """
        Clean text from symbols, punctuation, etc.

        Parameters:
        -----------
        text: string, text data

        Returns:
        --------
        cleaned_text: string, cleaned text data
        """
        # remove string formatting '\n' or '\t'
        tmp_text = re.sub(r'\n+', '. ', text)
        tmp_text = re.sub(r'\t+', '. ', text)
        # remove words with non-ascii characters
        tmp_text = " ".join([word for word in tmp_text.split() if self.is_ascii(word)])
        # remove email address
        tmp_text = " ".join([word for word in tmp_text.split() if not word.startswith("@")])
        # remove urls
        tmp_text = re.sub(r'http\S+', '', tmp_text, flags=re.MULTILINE)
        tmp_text = re.sub(r'www\S+', '', tmp_text, flags=re.MULTILINE)
        # remove punctuation but . (to split sentences)
        cleaned_text = re.sub('[^A-Za-z.,]+', ' ', tmp_text)
        # lowercase
        cleaned_text = cleaned_text.lower()

        return cleaned_text

    def text_preprocessing_a1(self, text):
        """
        Approach 1: Join first 2 sentences with last 2 sentences of the text.

        Parameters:
        -----------
        text: string, text data

        Returns:
        --------
        preprocessed_text: string, preprocessed text data
        """
        # sentence tokenize based on '. '
        sentences = text.split('. ')
        # get 2 first and 2 last sentences
        if len(sentences) >= 4:
            preprocessed_text = ". ".join(text.split('. ')[:2] + text.split('. ')[-2:])
            return preprocessed_text
        # if there are not 4 sentences, return full text
        else:
            preprocessed_text = text
            return preprocessed_text

    def preprocessing_a1(self, text):
        """
        Cleaning and preprocessing following approach 1.

        Parameters:
        -----------
        text: string, text data

        Returns:
        --------
        preprocessed_text: string, preprocessed text data
        """
        # clean description
        cleaned_text = self.text_cleaning(text)
        # preprocess description
        preprocessed_text = self.text_preprocessing_a1(cleaned_text)

        return preprocessed_text

    def preprocess(self, requests):
        """Text preprocessing.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            data_loader : The preprocess function returns a dataloader with processed text data ready for DistilBERT
        """
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')
        logger.info(input_text)
        # split sentences
        input_text = input_text.split('\n')
        max_length = 210
        logger.info(f"Received texts: {input_text}")
        logger.info(f"Input text type: {type(input_text)}")
        logger.info(f"Number of examples: {len(input_text)}")

        # preprocessing text
        preprocessed_input_text = [self.preprocessing_a1(text) for text in input_text]

        # torch dataset
        inference_set = DistilBertDatasetInference(preprocessed_input_text, self.tokenizer, max_length)
        # dataloader
        params = {'batch_size': 16,
                  'shuffle': False,
                  'num_workers': 4
                  }
        data_loader = DataLoader(inference_set, **params)

        return data_loader

    def inference(self, data_loader):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            data_loader : Torch dataloader with processed text data ready for DistilBERT
        Returns:
            list : It returns a list of the predicted value for the input text
        """

        # Handling inference for Multiclass classification
        outputs_list = []
        with torch.no_grad():
            for bi, d in enumerate(data_loader):
                ids = d['ids']
                mask = d['mask']

                # send them to the cpu
                ids = ids.to(self.device, dtype=torch.long)
                mask = mask.to(self.device, dtype=torch.long)

                outputs = self.model(
                    input_ids=ids,
                    attention_mask=mask
                )
                outputs_list.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy().tolist())

        print("This the output from the Multiclass classification model", outputs_list[0])

        inferences = np.argmax(outputs_list, axis=1)
        inferences = [self.mapping[str(pred)] for pred in inferences]

        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        logger.info(inference_output)
        return inference_output


_service = DistilBERTMultiClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e