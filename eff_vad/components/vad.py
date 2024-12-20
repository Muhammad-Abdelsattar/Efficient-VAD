import numpy as np
from typing import Any
from .base_component import BaseComponent
from .utils import softmax


class VAD(BaseComponent):
    """
    Represents a Voice Activity Detection (VAD) component that can be used to detect
    speech activity in audio data. This component uses an ONNX-based model to
    perform the VAD inference.
    """
    def __init__(self,
                 model_path:str,
                 confidence_threshold:float,
                 chunk_size:int):
        super().__init__(model_path=model_path)

        self.confidence_threshold = confidence_threshold
        self.chunk_size = int((chunk_size/16000)*100)

    def prepare_data(self,
                     input_data: np.ndarray):
        """
        Prepares the input data for the VAD inference by chunking the input data into smaller segments and batching them.
        
        Args:
            input_data (np.ndarray): The input audio data to be processed.
        
        Returns:
            dict[str, np.ndarray]: A dictionary containing the prepared input data for the VAD inference.
        """
        if(input_data.shape[-1]<self.chunk_size):
            return input_data

        number_of_chunks = input_data.shape[2]//self.chunk_size
        chunks = np.array_split(input_data,number_of_chunks,axis=-1)
        batched_inputs = np.zeros((len(chunks),chunks[0].shape[1],chunks[0].shape[2])).astype(np.float32)
        for idx,chunk in enumerate(chunks):
            batched_inputs[idx,:,:chunk.shape[2]] = chunk
        prepared_input = {self.session_input[0].name:batched_inputs}
        return prepared_input

    def perform_inference(self,
                          prepared_input: dict[str,np.ndarray]):

        raw_output = self.session.run(None,prepared_input)
        return raw_output[0]

    def postprocess(self,
                    raw_output: Any):
        """
        Postprocesses the raw output of the VAD inference to obtain the final VAD results as a boolean array.
        Args:
            raw_output (Any): The raw output of the VAD inference.

        Returns:
            np.ndarray: A boolean array indicating the presence of speech activity in each chunk of the input audio data.
        """

        return softmax(raw_output)[:,1] > self.confidence_threshold
