import os
import pytest # Import pytest
from eff_vad import VADPipeline

# Determine the project root directory to construct absolute paths to artifacts
try:
    PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) if "__file__" in locals() else os.getcwd()
except NameError: 
    PROJECT_ROOT = os.getcwd()

def test_init_with_default_paths():
    """Test VADPipeline initialization with default model paths."""
    try:
        pipeline = VADPipeline(confidence_threshold=0.5, chunk_size=480)
        assert pipeline.preprocessor.session is not None
        assert pipeline.vad.session is not None
        print("pytest: test_init_with_default_paths PASSED") # pytest captures stdout by default with -s
    except Exception as e:
        pytest.fail(f"Initialization with default paths failed: {e}")

def test_init_with_explicit_paths():
    """Test VADPipeline initialization with explicit (correct) model paths."""
    preprocessor_path = os.path.join(PROJECT_ROOT, "eff_vad", "artifcats", "preprocessor.onnx")
    vad_model_path = os.path.join(PROJECT_ROOT, "eff_vad", "artifcats", "vad_ml.onnx")

    if not os.path.exists(preprocessor_path):
        pytest.fail(f"Preprocessor model not found at expected path for test: {preprocessor_path}")
    if not os.path.exists(vad_model_path):
        pytest.fail(f"VAD model not found at expected path for test: {vad_model_path}")

    try:
        pipeline = VADPipeline(
            preprocessor_model_path=preprocessor_path,
            vad_model_path=vad_model_path,
            confidence_threshold=0.5,
            chunk_size=480
        )
        assert pipeline.preprocessor.session is not None
        assert pipeline.vad.session is not None
        print("pytest: test_init_with_explicit_paths PASSED") # pytest captures stdout by default with -s
    except Exception as e:
        pytest.fail(f"Initialization with explicit paths failed: {e}")

