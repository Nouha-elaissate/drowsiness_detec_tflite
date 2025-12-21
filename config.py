try:
    import tflite_runtime.interpreter as tflite
    print("Running on Raspberry Pi (TFLite Runtime)")
except ImportError:
    import tensorflow.lite as tflite
    print("Running on PC (TensorFlow CPU)")