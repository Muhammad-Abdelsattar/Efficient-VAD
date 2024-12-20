import numpy as np
import onnxruntime as ort
from typing import Any,Optional

class BaseComponent:
    def __init__(self,
                 model_path: str):
        
        self.session = ort.InferenceSession(model_path)
        self.session_input = self.session.get_inputs()
        
        
    def prepare_data(self,
                     input_data: np.ndarray):
        
        raise NotImplementedError()
    
    
    def perform_inference(self,
                    prepared_input: dict[str,np.ndarray]):
        
        raise NotImplementedError()
        
        
    def postprocess(self,
                    raw_output: Any):
        
        raise NotImplementedError()
        
        
    def __call__(self,
                 input_data: np.ndarray):
        
        prepared_input = self.prepare_data(input_data=input_data)
        raw_output = self.perform_inference(prepared_input=prepared_input)
        
        return self.postprocess(raw_output=raw_output)
        
        
        