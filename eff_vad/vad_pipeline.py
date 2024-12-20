import numpy as np
from .components import *

class VADPipeline:
    def __init__(self,
                 preprocessor_model_path:str,
                 vad_model_path:str,
                 confidence_threshold:float,
                 chunk_size:int):
        
        self.preprocessor = Featurizer(model_path=preprocessor_model_path)
        self.vad = VAD(model_path=vad_model_path,
                        confidence_threshold=confidence_threshold,
                        chunk_size=chunk_size)
        

    def __call__(self,input_signal: np.ndarray):
        """
        Performs voice activity detection on the input audio signal.
        Args:
            input_signal (np.ndarray): The input audio signal to be processed.

        Returns:
            np.ndarray: The output of the VAD model.
        """
        featurized_input = self.preprocessor(input_signal)
        vad_output = self.vad(featurized_input)
        return vad_output