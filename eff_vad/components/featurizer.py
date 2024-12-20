import numpy as np
from typing import Any
from .base_component import BaseComponent


class  Featurizer(BaseComponent):
    """
    A featurizer component used to extract mel spectrogram features from audio signals.
    """
    def __init__(self,
                 model_path:str):
        super().__init__(model_path=model_path)


    def prepare_data(self,
                     input_data: np.ndarray):
        """
        Prepares the signal before being processed by the featurizer model.
        The featurizer model expects the signal to be in float32 dtype and between ( -1 & 1) or int16 dtype between (-32768 & 32767).

        Args:
            audio_signal (np.ndarray) of shape (N,num_samples): The audio signal to be processed

        Returns:
            np.ndarray: The prepared audio signal ready to be featurized.
        """

        prepared_input = {}
        dtype = input_data.dtype

        if(dtype == np.int16):
            #perform scaling
            input_data = self.int16_to_float32(audio_signal=input_data)

        elif(dtype == np.float32):
            #don't perorm scaling
            pass
        else:
            raise NotImplementedError("Data type of the input must be either int16 or float32")

        input_data = self.fix_dimentions(audio_sginal=input_data)
        prepared_input[self.session_input[0].name] = input_data
        return prepared_input


    def perform_inference(self,
                          prepared_input: dict[str,np.ndarray]):

        raw_output = self.session.run(None,prepared_input)
        return raw_output


    def postprocess(self,
                    raw_output: Any):

        return raw_output[0]


    def int16_to_float32(self,
                         audio_signal: np.ndarray):

        return (audio_signal / 2.0**15).astype(np.float32)


    def fix_dimentions(self,
                       audio_sginal: np.ndarray):

        shape = audio_sginal.shape

        if(len(shape)==2): #array is already in batch format (batch_size,num_audio_samples)
            return audio_sginal

        elif(len(shape)==1): #array is in single element format (num_audio_samples)
            return np.expand_dims(audio_sginal,axis=0)

        else:
            raise NotImplementedError('The input must be of shape (N,T) or (T) where N is the batch size and T is the number of samples.')
