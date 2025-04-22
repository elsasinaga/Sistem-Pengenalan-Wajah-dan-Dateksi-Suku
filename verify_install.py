try:
    import streamlit
    import cv2
    import mtcnn
    import tensorflow as tf
    import keras
    import numpy as np
    import pandas as pd
    import sklearn
    from PIL import Image
    import altair
    print("Semua library terinstal!")
    print(f"Streamlit: {streamlit.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"Pillow: {Image.__version__}")
    print(f"Altair: {altair.__version__}")
except ImportError as e:
    print(f"Error: {e}. Instal library yang hilang.")