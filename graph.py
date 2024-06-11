# Python library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from collections import Counter
from flask import send_file



def plot_png():
    # Load dataset
    df1 = pd.read_csv("testYOLOv7 - Sheet1.csv")
    # Filter dataframe for only 'yolov7' models
    df_yolov7 = df1[df1['Model'].str.contains('yolov7')]

    # Create a color dictionary for each unique model
    colors = {model: plt.cm.rainbow(i/float(len(df_yolov7['Model'].unique())-1)) for i, model in enumerate(df_yolov7['Model'].unique())}

    # Create a line plot for each model
    for model in df_yolov7['Model'].unique():
        df_model = df_yolov7[df_yolov7['Model'] == model]
        plt.plot(df_model['Parameter'], df_model['Map 0.5'], marker='o', color=colors[model], label=model)

    plt.xlabel('Parameter (million)')
    plt.ylabel('Map 0.5')
    plt.title('Map 0.5 vs Parameter for each YOLOv7 Model')
    plt.legend()
    plt.grid(True)

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')



