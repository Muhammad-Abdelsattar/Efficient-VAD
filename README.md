# Efficient Voice Activity Detection (eff_vad)

This project provides an efficient, ONNX-based Voice Activity Detection (VAD) system. It can be used to detect speech segments in an audio signal.

## Requirements

- Python (>= 3.8)
- NumPy (>= 1.22.0)
- ONNX Runtime (>= 1.11.0)

## Installation

You can install this package directly from GitHub or by cloning the repository and installing from source.

**From GitHub:**

If you have Git installed, you can install the package directly from this repository using pip:

```bash
pip install git+https://github.com/<USERNAME>/<REPONAME>.git
```
*Replace `<USERNAME>/<REPONAME>` with the actual GitHub username and repository name.*

**From local source:**

First, clone this repository:

```bash
git clone https://github.com/<USERNAME>/<REPONAME>.git
cd <REPONAME>
```
*Replace `<USERNAME>/<REPONAME>` with the actual GitHub username and repository name.*

Then, install the package using pip:

```bash
pip install .
```

This will install the `Efficient-VAD` package and its dependencies.

## Running Tests

To run the tests, first ensure you have installed the development dependencies:

```bash
pip install .[test]
```

Then, you can run the tests using `pytest` from the root of the project:

```bash
pytest
```

## Usage

The main way to use `eff_vad` is through the `VADPipeline` class.

First, import the necessary classes and NumPy:

```python
import numpy as np
from eff_vad import VADPipeline
# If running directly from the cloned repository *without installing* (e.g., for development):
# from eff_vad.vad_pipeline import VADPipeline 
```

### Initializing the Pipeline

The `VADPipeline` requires the following parameters:

- `preprocessor_model_path (str)`: (Optional) Path to the ONNX model for feature preprocessing. Defaults to `"eff_vad/artifcats/preprocessor.onnx"` relative to the package.
- `vad_model_path (str)`: (Optional) Path to the ONNX model for voice activity detection. Defaults to `"eff_vad/artifcats/vad_ml.onnx"` relative to the package.
- `confidence_threshold (float)`: A threshold between 0.0 and 1.0. If the model's confidence for speech in a chunk is above this, it's marked as speech. A common value is 0.5.
- `chunk_size (int)`: The size of each audio chunk to be processed, **specified in number of samples for 16kHz audio**. For example, for 30ms chunks, `chunk_size` would be `0.030 * 16000 = 480` samples.

Example initialization using default model paths:

```python
# Initialize the VAD pipeline using default model paths
pipeline = VADPipeline(
    confidence_threshold=0.5,
    chunk_size=480  # Corresponds to 30ms chunks at 16kHz
)
```

Example initialization with custom model paths:

```python
# Define paths to the ONNX models (assuming they are in the expected location)
preprocessor_path = "custom/path/to/preprocessor.onnx"
vad_model_path = "custom/path/to/vad_ml.onnx"

# Initialize the VAD pipeline
pipeline = VADPipeline(
    preprocessor_model_path=preprocessor_path,
    vad_model_path=vad_model_path,
    confidence_threshold=0.5,
    chunk_size=480  # Corresponds to 30ms chunks at 16kHz (0.030s * 16000 Hz = 480 samples)
)
```

### Processing Audio

The pipeline processes audio provided as a NumPy array. The audio signal should be:
- A 1D NumPy array (for a single audio signal) or a 2D NumPy array (for a batch of signals, shape: `(batch_size, num_samples)`).
- Sampled at 16kHz.
- Data type can be `np.int16` (values between -32768 and 32767) or `np.float32` (values between -1.0 and 1.0).

Example with a dummy audio signal:

```python
# Create a dummy audio signal (e.g., 1 second of silence at 16kHz, float32)
sample_rate = 16000
duration = 1  # seconds
dummy_audio_float32 = np.zeros(int(sample_rate * duration), dtype=np.float32)

# Or, for int16:
# dummy_audio_int16 = np.zeros(int(sample_rate * duration), dtype=np.int16)

# Process the audio
vad_output = pipeline(dummy_audio_float32)

print(f"VAD output: {vad_output}")
# Example output for 1s of silence with 30ms (480 sample) chunks:
# VAD output: [False False False ... False False] (length will be num_samples / chunk_size)
```

### Interpreting the Output

The output from the pipeline (`vad_output`) is a NumPy boolean array.
- Each boolean value corresponds to one chunk of the input audio.
- `True` indicates that speech was detected in that chunk.
- `False` indicates that no speech (or silence) was detected.

The number of elements in the output array will be `total_samples // chunk_size_samples` (integer division). Any trailing audio shorter than `chunk_size_samples` will be ignored.
The duration of each chunk is determined by the `chunk_size` parameter you provided during initialization. For example, if `chunk_size=480` (for 16kHz audio), each boolean in the output represents a 30ms segment of the audio.

You can use this output to find speech segments:

```python
# Assuming vad_output from the example above and chunk_size = 480 (30ms)
chunk_duration_ms = (480 / 16000) * 1000  # 30ms

for i, is_speech in enumerate(vad_output):
    start_time_ms = i * chunk_duration_ms
    end_time_ms = (i + 1) * chunk_duration_ms
    if is_speech:
        print(f"Speech detected from {start_time_ms:.0f}ms to {end_time_ms:.0f}ms")
```

## Model Details

This VAD system uses two models in the ONNX (Open Neural Network Exchange) format:

1.  **Preprocessor Model (`preprocessor.onnx`)**: This model takes the raw audio signal and converts it into spectral features (mel spectrogram) that are suitable for the VAD model.
2.  **VAD Model (`vad_ml.onnx`)**: This model takes the spectral features from the preprocessor and performs the voice activity detection, outputting probabilities for speech presence in each processed chunk.

The models are located in the `eff_vad/artifcats/` directory.

## Contributing

Contributions are welcome! If you'd like to contribute, please feel free to fork the repository, make your changes, and submit a pull request. You can also open an issue if you find a bug or have a feature request.

## License

This project does not currently have a license. It is recommended to add a `LICENSE` file to define how others can use, modify, and distribute the code. Common open-source licenses include MIT, Apache 2.0, and GPL.
