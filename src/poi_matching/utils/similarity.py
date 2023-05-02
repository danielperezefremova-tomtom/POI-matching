import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

class SimilarityModel:
    """Class for computing similarity between two strings.
    It uses the method "encode" of the class SentenceTransformer.
    """

    def __init__(self, batch, model_name=MODEL_NAME) -> None:
        try:
            model = SentenceTransformer(model_name)
        except:
            raise ValueError(
                "Please provide a valid model available in SentenceTransformer pretrained models"
            )

        if torch.cuda.is_available():
            print('Cuda device available')
        else:
            print('No cuda device available')
        # Instantiate the client and replace with your endpoint.
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = batch

    def encode_string(self, strings_array: str):
        """Function that apply the method "encode" of the selected model to the string provided.
        It runs in cuda if available.

        :param strings_array: Array with the strings to encode.
        :type strings_array: Array
        :return: Response of the "encode" method of the class SentenceTransformer
        :rtype: Array
        """
        try:
            response = self.model.encode(
                strings_array,
                device=self.device,
                batch_size=self.batch,
                convert_to_tensor=True,
            )
        except:
            raise ValueError(
                "Please provide a valid model available in SentenceTransformer pretrained models"
            )

        return response

    def compute_similarity_strings(self, stringsA, stringsB):
        """Function to compare two strings using the class method "encode_string".

        :param stringsA: First array with the strings to compare
        :type stringsA: Array
        :param stringsB: Second array with the strings to compare
        :type stringsB: Array
        :return: Array with the value of similitude for each stringA-stringB pair
        :rtype: Array
        """
        stringsA_lower = np.array([string.lower() for string in stringsA])
        stringsB_lower = np.array([string.lower() for string in stringsB])

        tensor_A_encoded = self.encode_string(stringsA_lower)
        tensor_B_encoded = self.encode_string(stringsB_lower)

        cos_sim_instance = torch.nn.CosineSimilarity(dim=1)
        similarity = cos_sim_instance(tensor_A_encoded, tensor_B_encoded)

        return similarity.cpu().numpy()