import numpy as np
import os
from .components import *

class VADPipeline:
    def __init__(self,
                 confidence_threshold: float,
                 chunk_size: int,
                 preprocessor_model_path: str = None,
                 vad_model_path: str = None):
        
        base_dir = os.path.dirname(__file__)
        if preprocessor_model_path is None:
            preprocessor_model_path = os.path.join(base_dir, "artifcats", "preprocessor.onnx")
        if vad_model_path is None:
            vad_model_path = os.path.join(base_dir, "artifcats", "vad_ml.onnx")
        
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