# Run this to enable CSS types
from IPython.core.display import HTML

# Font stuff
font_to_use = "Barlow" # "Verdana"
fallback = "Verdana"
font_import_str = f"""
@import url('https://fonts.googleapis.com/css2?family={font_to_use.replace(' ', '+')}:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');
"""

def css_styling(verbose=True):
    styles = open("/kaggle/input/my-css-styles/kaggle_styles.css", "r").read().replace('Verdana', font_to_use) #+f", {fallback}")
    html_str = "<style>"+font_import_str+styles+"\nb{font-weight:bold;}\n"+"</style>"
    if verbose: print(html_str)
    return HTML(html_str)

css_styling(False)

# --- Cell Separator ---

# print("\n... PIP INSTALLS STARTING ...\n")
# print("\n... PIP INSTALLS COMPLETE ...\n")

print("\n... IMPORTS STARTING ...\n")
print("\n\tVERSION INFORMATION")

# Competition Specific Import
# TBD

import pandas as pd; pd.options.mode.chained_assignment = None; pd.set_option('display.max_columns', None); import pandas;
import numpy as np; print(f"\t\t– NUMPY VERSION: {np.__version__}");
import sklearn; print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}");
from sklearn.metrics import roc_curve, auc, roc_auc_score
import cv2; print(f"\t\t– CV2 VERSION: {cv2.__version__}");

# For modelling and dataset
import lightgbm as lgb
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import OrdinalEncoder

# Built-In Imports (mostly don't worry about these)
from typing import Iterable, Any, Callable, Generator
from kaggle_datasets import KaggleDatasets
from dataclasses import dataclass
from collections import Counter
from datetime import datetime
from zipfile import ZipFile
from glob import glob
import subprocess
import warnings
import requests
import textwrap
import hashlib
import imageio
import IPython
import urllib
import zipfile
import pickle
import random
import shutil
import string
import h5py
import json
import copy
import math
import time
import gzip
import ast
import sys
import io
import gc
import re
import os

# Visualization Imports (overkill)
from IPython.core.display import HTML, Markdown
import matplotlib.pyplot as plt
from matplotlib import animation, rc; rc('animation', html='jshtml')
from tqdm.notebook import tqdm; tqdm.pandas();
import plotly.graph_objects as go
import plotly.express as px
import plotly
import seaborn as sns
from PIL import Image, ImageEnhance, ImageColor; Image.MAX_IMAGE_PIXELS = 5_000_000_000;
import matplotlib; print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}");
from colorama import Fore, Style, init; init()
import PIL

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple.

    Args:
        hex_color (str): The hex color string, starting with '#'.

    Returns:
        tuple: A tuple of RGB values.
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def clr_print(text: str, color: str = "#42BFBA", bold: bool = True) -> None:
    """Print the given text with the specified color and bold formatting.

    Args:
        text (str): The text to format.
        color (str): The hex color code to apply. Defaults to "#752F55".
        bold (bool): Whether to apply bold formatting. Defaults to True.
    """
    _text = text.replace('\n', '<br>')
    rgb = hex_to_rgb(color)
    color_style = f"color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});"
    bold_style = "font-weight: bold;" if bold else ""
    style = f"{color_style} {bold_style}"
    display(HTML(f"<span style='{style}'>{_text}</span>"))

def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    
seed_it_all()

# Create a Seaborn color palette
nb_palette = sns.color_palette(palette='tab20')

# Create colors for class labels
LABELS = ["Benign", "Malignant"]
COLORS = ['#66c2a5', '#fc8d62']
CLR_MAP_I2C = {i:c for i,c in enumerate(COLORS)}
CLR_MAP_S2C = {l:c for l,c in zip(LABELS, COLORS)}
LBL_MAP_I2S = {i:l for i,l in enumerate(LABELS)}
LBL_MAP_S2I = {v:k for k,v in LBL_MAP_I2S.items()}

# Is this notebook being run on the backend for scoring re-submission
IS_DEBUG = False if os.getenv('KAGGLE_IS_COMPETITION_RERUN') else True
print(f"IS DEBUG: {IS_DEBUG}")

# Plot the palette
clr_print("\n... NOTEBOOK COLOUR PALETTE ...")
sns.palplot(nb_palette, size=0.5)
plt.show()

print("\n\n... IMPORTS COMPLETE ...\n")

# --- Cell Separator ---

"""2024 ISIC Challenge primary prize scoring metric

Given a list of binary labels, an associated list of prediction 
scores ranging from [0,1], this function produces, as a single value, 
the partial area under the receiver operating characteristic (pAUC) 
above a given true positive rate (TPR).
https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

(c) 2024 Nicholas R Kurtansky, MSKCC
"""


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80) -> float:
    '''
    2024 ISIC Challenge metric: pAUC
    
    Given a solution file and submission file, this function returns the
    the partial area under the receiver operating characteristic (pAUC) 
    above a given true positive rate (TPR) = 0.80.
    https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.
    
    (c) 2024 Nicholas R Kurtansky, MSKCC

    Args:
        solution: ground truth pd.DataFrame of 1s and 0s
        submission: solution dataframe of predictions of scores ranging [0, 1]

    Returns:
        Float value range [0, max_fpr]
    '''

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # check submission is numeric
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('Submission target column must be numeric')

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray(solution.values)-1)
    
    # flip the submissions to their compliments
    v_pred = -1.0*np.asarray(submission.values)

    max_fpr = abs(1-min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
        
    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    #     # Equivalent code that uses sklearn's roc_auc_score
    #     v_gt = abs(np.asarray(solution.values)-1)
    #     v_pred = np.array([1.0 - x for x in submission.values])
    #     max_fpr = abs(1-min_tpr)
    #     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    #     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    #     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    #     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return(partial_auc)

# --- Cell Separator ---

def flatten_l_o_l(nested_list):
    """ Flatten a list of lists into a single list.

    Args:
        nested_list (Iterable): 
            – A list of lists (or iterables) to be flattened.

    Returns:
        A flattened list containing all items from the input list of lists.
    """
    return [item for sublist in nested_list for item in sublist]


def print_ln(symbol="-", line_len=110, newline_before=False, newline_after=False):
    """ Print a horizontal line of a specified length and symbol.

    Args:
        symbol (str, optional): 
            – The symbol to use for the horizontal line
        line_len (int, optional): 
            – The length of the horizontal line in characters
        newline_before (bool, optional): 
            – Whether to print a newline character before the line
        newline_after (bool, optional): 
            – Whether to print a newline character after the line
            
    Returns:
        None; A divider with pre/post new-lines (optional) is printed
    """
    if newline_before: print();
    print(symbol * line_len)
    if newline_after: print();
        
        
def display_hr(newline_before=False, newline_after=False):
    """ Renders a HTML <hr>

    Args:
        newline_before (bool, optional): 
            – Whether to print a newline character before the line
        newline_after (bool, optional): 
            – Whether to print a newline character after the line
            
    Returns:
        None; A divider with pre/post new-lines (optional) is printed
    """
    if newline_before: print();
    display(HTML("<hr>"))
    if newline_after: print();


def wrap_text(text, width=88):
    """Wrap text to a specified width.

    Args:
        text (str): 
            - The text to wrap.
        width (int): 
            - The maximum width of a line. Default is 88.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width)


def wrap_text_by_paragraphs(text, width=88):
    """Wrap text by paragraphs to a specified width.

    Args:
        text (str): 
            - The text containing multiple paragraphs to wrap.
        width (int): 
            - The maximum width of a line. Default is 88.

    Returns:
        str: The wrapped text with preserved paragraph separation.
    """
    paragraphs = text.split('\n')  # Assuming paragraphs are separated by newlines
    wrapped_paragraphs = [textwrap.fill(paragraph, width) for paragraph in paragraphs]
    return '\n\n'.join(wrapped_paragraphs)

# --- Cell Separator ---

def load_img_from_hdf5(
    isic_id: str, 
    file_path: str = "/kaggle/input/isic-2024-challenge/train-image.hdf5", 
    n_channels: int = 3
):
    """
    Load an image from the HDF5 dataset file by specifying an ISIC ID.
    
    The ISIC ID is expected to be in the form 'ISIC_#######'.
    
    Args:
        isic_id (str): The ID of the image to load.
        file_path (str): The path to the HDF5 file.
        n_channels (int): Number of channels (3 for RGB, 1 for grayscale).
    
    Returns:
        np.ndarray: The loaded image.
    
    Raises:
        KeyError: If the ISIC ID is not found in the HDF5 file.
        ValueError: If the image data cannot be decoded.
    
    Example Usage:
        img = load_img_from_hdf5('ISIC_0000000')
    """
    
    # Handle the case where the isic_id is passed incorrectly
    if not isic_id.lower().startswith("isic"):
        isic_id = f"ISIC_{int(str(isic_id).split('_', 1)[-1]):>07}"
        
    # Open the HDF5 file in read mode
    with h5py.File(file_path, 'r') as hf:
        
        # Retrieve the image data from the HDF5 dataset using the provided ISIC ID
        try:
            image_data = hf[isic_id][()]
        except KeyError:
            raise KeyError(f"ISIC ID {isic_id} not found in HDF5 file.")

        # Convert the binary data to a numpy array
        image_array = np.frombuffer(image_data, np.uint8)

        # Decode the image from the numpy array
        if n_channels == 3:
            # Load the image as a color image (BGR) and convert to RGB
            image = cv2.cvtColor(cv2.imdecode(image_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        else:
            # Load the image as a grayscale image
            image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

        # If the image failed to load for some reason (problems decoding) ...
        if image is None:
            raise ValueError(f"Could not decode image for ISIC ID: {isic_id}")
        
        return image
    
plt.figure(figsize=(6,6))
plt.title("ISIC_0015670", fontweight="bold")
plt.imshow(load_img_from_hdf5("ISIC_0015670"))
plt.show()

METADATA_COL2DESC = {
    "isic_id": "Unique identifier for each image case.",
    "target": "Binary class label indicating if the lesion is benign (0) or malignant (1).",
    "patient_id": "Unique identifier for each patient.",
    "age_approx": "Approximate age of the patient at the time of imaging.",
    "sex": "Sex of the patient (male or female).",
    "anatom_site_general": "General location of the lesion on the patient's body (e.g., upper extremity, posterior torso).",
    "clin_size_long_diam_mm": "Maximum diameter of the lesion in millimeters.",
    "image_type": "Type of image captured, as defined in the ISIC Archive.",
    "tbp_tile_type": "Lighting modality of the 3D Total Body Photography (TBP) source image.",
    "tbp_lv_A": "Color channel A inside the lesion; related to the green-red axis in LAB color space.",
    "tbp_lv_Aext": "Color channel A outside the lesion; related to the green-red axis in LAB color space.",
    "tbp_lv_B": "Color channel B inside the lesion; related to the blue-yellow axis in LAB color space.",
    "tbp_lv_Bext": "Color channel B outside the lesion; related to the blue-yellow axis in LAB color space.",
    "tbp_lv_C": "Chroma value inside the lesion, indicating color purity.",
    "tbp_lv_Cext": "Chroma value outside the lesion, indicating color purity.",
    "tbp_lv_H": "Hue value inside the lesion, representing the type of color (e.g., red, brown) in LAB color space.",
    "tbp_lv_Hext": "Hue value outside the lesion, representing the type of color (e.g., red, brown) in LAB color space.",
    "tbp_lv_L": "Luminance value inside the lesion; related to lightness in LAB color space.",
    "tbp_lv_Lext": "Luminance value outside the lesion; related to lightness in LAB color space.",
    "tbp_lv_areaMM2": "Area of the lesion in square millimeters.",
    "tbp_lv_area_perim_ratio": "Ratio of the lesion's perimeter to its area, indicating border jaggedness.",
    "tbp_lv_color_std_mean": "Mean color irregularity within the lesion, calculated as the variance of colors.",
    "tbp_lv_deltaA": "Average contrast in color channel A between inside and outside the lesion.",
    "tbp_lv_deltaB": "Average contrast in color channel B between inside and outside the lesion.",
    "tbp_lv_deltaL": "Average contrast in luminance between inside and outside the lesion.",
    "tbp_lv_deltaLB": "Combined contrast between the lesion and its immediate surrounding skin.",
    "tbp_lv_deltaLBnorm": "Normalized contrast between the lesion and its immediate surrounding skin in LAB color space.",
    "tbp_lv_eccentricity": "Eccentricity of the lesion, indicating how elongated it is.",
    "tbp_lv_location": "Detailed anatomical location of the lesion, dividing body parts further (e.g., Left Arm - Upper).",
    "tbp_lv_location_simple": "Simplified anatomical location of the lesion (e.g., Left Arm).",
    "tbp_lv_minorAxisMM": "Smallest diameter of the lesion in millimeters.",
    "tbp_lv_nevi_confidence": "Confidence score (0-100) from a neural network classifier estimating the probability that the lesion is a nevus.",
    "tbp_lv_norm_border": "Normalized border irregularity score on a scale of 0-10.",
    "tbp_lv_norm_color": "Normalized color variation score on a scale of 0-10.",
    "tbp_lv_perimeterMM": "Perimeter of the lesion in millimeters.",
    "tbp_lv_radial_color_std_max": "Color asymmetry score within the lesion, based on color variance in concentric rings.",
    "tbp_lv_stdL": "Standard deviation of luminance within the lesion.",
    "tbp_lv_stdLExt": "Standard deviation of luminance outside the lesion.",
    "tbp_lv_symm_2axis": "Measure of asymmetry of the lesion's border about a secondary axis.",
    "tbp_lv_symm_2axis_angle": "Angle of the secondary axis of symmetry for the lesion's border.",
    "tbp_lv_x": "X-coordinate of the lesion in the 3D TBP model.",
    "tbp_lv_y": "Y-coordinate of the lesion in the 3D TBP model.",
    "tbp_lv_z": "Z-coordinate of the lesion in the 3D TBP model.",
    "attribution": "Source or institution responsible for the image.",
    "copyright_license": "Type of copyright license for the image.",
    "lesion_id": "Unique identifier for lesions that were manually tagged as lesions of interest.",
    "iddx_full": "Full classified diagnosis of the lesion.",
    "iddx_1": "First-level diagnosis of the lesion (e.g., Benign, Malignant).",
    "iddx_2": "Second-level diagnosis providing more specific details about the lesion.",
    "iddx_3": "Third-level diagnosis with further classification details.",
    "iddx_4": "Fourth-level diagnosis with additional specificity.",
    "iddx_5": "Fifth-level diagnosis, providing the most detailed classification.",
    "mel_mitotic_index": "Mitotic index of invasive malignant melanomas, indicating cell division rate.",
    "mel_thick_mm": "Thickness in millimeters of melanoma invasion.",
    "tbp_lv_dnn_lesion_confidence": "Lesion confidence score (0-100) from a deep neural network classifier."
}


METADATA_COL2NAME = {
    "isic_id": "Unique Case Identifier",
    "target": "Binary Lession Classification",
    "patient_id": "Unique Patient Identifier",
    "age_approx": "Approximate Age",
    "sex": "Sex",
    "anatom_site_general": "General Anatomical Location",
    "clin_size_long_diam_mm": "Clinical Size (Longest Diameter in mm)",
    "image_type": "Image Type",
    "tbp_tile_type": "TBP Tile Type",
    "tbp_lv_A": "Color Channel A Inside Lesion",
    "tbp_lv_Aext": "Color Channel A Outside Lesion",
    "tbp_lv_B": "Color Channel B Inside Lesion",
    "tbp_lv_Bext": "Color Channel B Outside Lesion",
    "tbp_lv_C": "Chroma Inside Lesion",
    "tbp_lv_Cext": "Chroma Outside Lesion",
    "tbp_lv_H": "Hue Inside Lesion",
    "tbp_lv_Hext": "Hue Outside Lesion",
    "tbp_lv_L": "Luminance Inside Lesion",
    "tbp_lv_Lext": "Luminance Outside Lesion",
    "tbp_lv_areaMM2": "Lesion Area (mm²)",
    "tbp_lv_area_perim_ratio": "Area-to-Perimeter Ratio",
    "tbp_lv_color_std_mean": "Mean Color Irregularity",
    "tbp_lv_deltaA": "Delta A (Inside vs. Outside)",
    "tbp_lv_deltaB": "Delta B (Inside vs. Outside)",
    "tbp_lv_deltaL": "Delta L (Inside vs. Outside)",
    "tbp_lv_deltaLB": "Delta LB (Contrast)",
    "tbp_lv_deltaLBnorm": "Normalized Delta LB (Contrast)",
    "tbp_lv_eccentricity": "Eccentricity",
    "tbp_lv_location": "Detailed Anatomical Location",
    "tbp_lv_location_simple": "Simplified Anatomical Location",
    "tbp_lv_minorAxisMM": "Smallest Diameter (mm)",
    "tbp_lv_nevi_confidence": "Nevus Confidence Score",
    "tbp_lv_norm_border": "Normalized Border Irregularity",
    "tbp_lv_norm_color": "Normalized Color Variation",
    "tbp_lv_perimeterMM": "Lesion Perimeter (mm)",
    "tbp_lv_radial_color_std_max": "Radial Color Standard Deviation",
    "tbp_lv_stdL": "Standard Deviation of Luminance (Inside)",
    "tbp_lv_stdLExt": "Standard Deviation of Luminance (Outside)",
    "tbp_lv_symm_2axis": "Symmetry (Second Axis)",
    "tbp_lv_symm_2axis_angle": "Symmetry Angle (Second Axis)",
    "tbp_lv_x": "X-Coordinate",
    "tbp_lv_y": "Y-Coordinate",
    "tbp_lv_z": "Z-Coordinate",
    "attribution": "Image Source",
    "copyright_license": "Copyright License",
    "lesion_id": "Unique Lesion Identifier",
    "iddx_full": "Full Diagnosis",
    "iddx_1": "First Level Diagnosis",
    "iddx_2": "Second Level Diagnosis",
    "iddx_3": "Third Level Diagnosis",
    "iddx_4": "Fourth Level Diagnosis",
    "iddx_5": "Fifth Level Diagnosis",
    "mel_mitotic_index": "Mitotic Index (Melanoma)",
    "mel_thick_mm": "Thickness of Melanoma (mm)",
    "tbp_lv_dnn_lesion_confidence": "Lesion Confidence Score"
}


# --- Cell Separator ---

# ROOT PATHS
WORKING_DIR = "/kaggle/working"
INPUT_DIR = "/kaggle/input"
COMPETITION_DIR = os.path.join(INPUT_DIR, "isic-2024-challenge")

# IMAGE DIRS
TRAIN_IMAGE_DIR = os.path.join(COMPETITION_DIR, "train-image", "image")
TEST_IMAGE_DIR = os.path.join(COMPETITION_DIR, "test-image", "image")

# FILE PATHS
TRAIN_METADATA_CSV = os.path.join(COMPETITION_DIR, "train-metadata.csv")
TEST_METADATA_CSV = os.path.join(COMPETITION_DIR, "test-metadata.csv")
TRAIN_IMAGE_HDF5 = os.path.join(COMPETITION_DIR, "train-image.hdf5")
TEST_IMAGE_HDF5 = os.path.join(COMPETITION_DIR, "test-image.hdf5")
SS_CSV_PATH = os.path.join(COMPETITION_DIR, "sample_submission.csv")


# DEFINE COMPETITION DATAFRAMES
clr_print("\n\n... SAMPLE SUBMISSION DATAFRAME ...\n\n")
ss_df = pd.read_csv(SS_CSV_PATH)
display(ss_df)

clr_print("\n\n... TRAIN METADATA DATAFRAME ...\n\n")
train_df = pd.read_csv(TRAIN_METADATA_CSV)
display(train_df)

clr_print("\n\n... TEST METADATA DATAFRAME ...\n\n")
test_df = pd.read_csv(TEST_METADATA_CSV)
display(test_df)

clr_print("\n\n... HDF5 (DATASET) PATHS ...\n\n")
print(f"\t--> {TRAIN_IMAGE_HDF5}")
print(f"\t--> {TEST_IMAGE_HDF5}\n")

for _c in train_df.columns:
    display_hr(True, True)
    clr_print(f"COLUMN NAME         : <code>'{_c}'</code>")
    clr_print(f"HUMAN READABLE NAME : <span style='color: black !important;'>'{METADATA_COL2NAME.get(_c)}'</span>")
    clr_print(f"COLUMN DESCRIPTION  : <span style='color: black !important;'>'{METADATA_COL2DESC.get(_c)}'</span>")
display_hr(True, True)

# --- Cell Separator ---

train_df.describe().T

# --- Cell Separator ---

def plot_nan_heatmap(
    df: pd.DataFrame, 
    figsize: tuple = (17, 8), 
    cmap: str = 'magma_r', 
    title: str = 'NaN Values in DataFrame',
    x_tick_rotation=60,
    show_cbar: bool = False, 
    show_yticklabels: bool = False
) -> None:
    """Create a heatmap to visualize NaN values in a DataFrame.

    Args:
        df (pd.DataFrame): 
            The input DataFrame to visualize.
        figsize (tuple[int], optional): 
            Figure size as a tuple of (width, height)
        cmap (str, optional): 
            Colormap to use for the heatmap
        title (str, optional): 
            Title for the heatmap.
        x_tick_rotation (int, optional): 
            Rotation angle for x-axis tick labels.
        show_cbar (bool, optional): 
            Whether to show the color bar.
        show_yticklabels (bool, optional): 
            Whether to show y-axis tick labels.

    Returns:
        None; 
            The function displays the plot using plt.show().
    """
    
    # Setup the figure
    plt.figure(figsize=figsize)
    
    # Create the heatmap
    sns.heatmap(df.isna(), cbar=show_cbar, yticklabels=show_yticklabels, cmap=cmap)
    
    # Update the title/labels
    plt.title(title, fontweight="bold")
    plt.xlabel('Columns', fontweight="bold")
    plt.ylabel('Rows', fontweight="bold")
    
    
    # Rotate x-axis labels
    plt.xticks(rotation=x_tick_rotation, ha='right')
    
    # Adjust the bottom margin to prevent label cutoff
    plt.tight_layout()
    
    # Render
    plt.show()

    # Print NaN counts per column
    nan_counts = df.isna().sum().sort_values(ascending=False)
    
    # clr_print()
    clr_print("NaN counts per column:")
    print(nan_counts[nan_counts])
    clr_print("Features with 0 NaN values:")
    print(_nan[_nan==0].index.tolist())

# --- Cell Separator ---

def plot_target_distribution(df: pd.DataFrame, log_y: bool = True) -> None:
    """Plot the distribution of the target variable.

    Args:
        df (pd.DataFrame): 
            The input dataframe containing the target column.
        log_y (bool, optional):
            Whether to log the y-axis (helpful for visualizing large class imbalance)

    Returns:
        None; 
            This function doesn't return anything, it displays a plot.
    """
    # Count the occurrences of each target value
    target_counts = df['target'].value_counts().sort_index()
    
    # Calculate percentages
    total = len(df)
    percentages = [f"{count/total:.3%}" for count in target_counts]
    
    # Create the bar plot
    fig = go.Figure(data=[
        go.Bar(
            x=LABELS,  # Assume we have access to this
            y=target_counts,
            text=percentages,
            textposition='auto',
            marker_color=COLORS  # Assume we have access to this
        )
    ])
    
    # Customize the layout
    fig.update_layout(
        title='<b>DISTRIBUTION OF BENIGN VS MALIGNANT LESIONS',
        xaxis_title='<b>Lesion Classification</b>', yaxis_title=f'<b>Count {"<sub>"+"(Log Scale)"+"</sub>" if log_y else ""}</b>',
        template='plotly_white', height=600, width=1200,
    )
    
    if log_y:
        fig.update_layout(yaxis=dict(type='log'))
    
    # Add annotation for total count
    fig.add_annotation(
        text=f"<b>TOTAL SAMPLES:  {total:,}</b>",
        xref="paper", yref="paper",
        x=0.98, y=1.05,
        showarrow=False,
        font=dict(size=12)
    )
    
    # Show the plot
    fig.show()

# Call the function
plot_target_distribution(train_df)

# --- Cell Separator ---

def plot_target_distribution(
    df: pd.DataFrame, 
    log_y: bool = True, 
    target_as_str: bool = True,
    target_col: str = "target", 
    color_sequence: list[str] | None = None,
    template_theme: str = "plotly_white"
) -> None:
    """Plot the distribution of the target variable.

    This function creates a histogram of the target variable distribution,
    with options for log scale, string labels, and custom color schemes.

    Args:
        df (pd.DataFrame): 
            The input dataframe containing the target column.
        log_y (bool, optional): 
            Whether to use log scale for y-axis.
        target_as_str (bool, optional): 
            Whether to convert target labels to strings.
        target_col (str, optional): 
            Name of the target column. 
        color_sequence (list[str], optional): 
            Custom color sequence for the bars. 
            Defaults to None, which uses the global COLORS.
        template_theme (str, optional): 
            Plotly template theme for visuals styling.
            The available templates are:
                - 'ggplot2', 'seaborn', 'simple_white', 
                - 'plotly', 'plotly_white', 'plotly_dark', 
                - 'presentation', 'xgridoff', 'ygridoff', 'gridon', 
                - 'none'

    Returns:
        None; 
            This function displays a plot and doesn't return anything.
    """
    
    # Prevent accidental edits to the original dataframe
    _df = df.copy()
    
    # Set default color sequence if not provided
    if not color_sequence:
        color_sequence = COLORS
    
    # Convert target labels to strings if requested
    if target_as_str:
        _df[target_col] = _df[target_col].map(LBL_MAP_I2S)
    
    # Create the histogram using Plotly Express
    fig = px.histogram(
        _df, x=target_col, color=target_col, 
        color_discrete_sequence=color_sequence, 
        log_y=log_y, height=500, width=1200, template=template_theme,
        title='<b>DISTRIBUTION OF BENIGN VS MALIGNANT LESIONS',
    )
    
    # Customize the layout
    fig.update_layout(
        bargap=0.1,  # Add space between bars
        xaxis_title='<b>Lesion Classification</b>', 
        yaxis_title=f'<b>Count {"<sub>(Log Scale)</sub>" if log_y else ""}</b>',
        showlegend=False  # Hide legend as color already differentiates categories
    )
    
    # Apply log scale to y-axis if requested
    if log_y:
        fig.update_layout(yaxis=dict(type='log'))
    
    # Add annotation for total sample count
    fig.add_annotation(
        text=f"<b>TOTAL SAMPLES: {len(_df):,}</b>",
        xref="paper", yref="paper",
        x=0.98, y=1.05,
        showarrow=False,
        font=dict(size=12)
    )
    
    # Display the plot
    fig.show()

plot_target_distribution(train_df)

# --- Cell Separator ---

def plot_categorical_feature_distribution(
    df: pd.DataFrame, 
    feature_col: str,
    target_col: str = "target",
    target_as_str: bool = True,
    log_y: bool = False, 
    color_sequence: list[str] | None = None,
    template_theme: str = "plotly_white",
    group_by_target: bool = True,
    stack_bars: bool = False
) -> None:
    """Plot the distribution of a feature, optionally grouped by the target variable.

    This function creates a histogram of the feature distribution,
    with options for log scale, custom color schemes, and grouping by target.

    Args:
        df (pd.DataFrame): 
            The input dataframe containing feature and target columns.
        feature_col (str): 
            Name of the feature column to plot.
        target_col (str, optional): 
            Name of the target column.
        target_as_str (bool, optional): 
            Whether to convert target labels to strings.
        log_y (bool, optional): 
            Whether to use log scale for y-axis.
        color_sequence (list[str], optional): 
            Custom color sequence for the bars.
        template_theme (str, optional): 
            Plotly template theme for visual styling.
            Available options include: 
                'ggplot2', 'seaborn', 'simple_white', 'plotly',
                'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff',
                'gridon', 'none'.
        group_by_target (bool, optional): 
            Whether to group bars by target.
        stack_bars (bool, optional): 
            Whether to stack bars when grouped.

    Returns:
        None: This function displays a plot and doesn't return anything.
    """
    # Prevent accidental edits to the original dataframe
    _df = df.copy().sort_values(by=[feature_col, target_col]).reset_index(drop=True)
        
    if target_as_str and group_by_target:
        _df[target_col] = _df[target_col].map(LBL_MAP_I2S)
        
    # Set default color sequence if not provided
    if not color_sequence:
        color_sequence = list(nb_palette.as_hex())
    
    # Prepare the histogram data
    if group_by_target:
        fig = px.histogram(
            _df, x=feature_col, color=target_col, 
            color_discrete_sequence=COLORS,  # Use target colors for grouping
            log_y=log_y, height=500, width=1200, template=template_theme,
            title=f'<b>DISTRIBUTION OF {feature_col.replace("_", " ").upper()} BY TARGET',
            barmode='group' if not stack_bars else 'stack'
        )
        
        # Add border to bars using the target colors
        for i, trace in enumerate(fig.data):
            trace.marker.line.color = COLORS[i]
            trace.marker.line.width = 1.5
    else:
        fig = px.histogram(
            _df, x=feature_col, color=feature_col, 
            color_discrete_sequence=color_sequence,
            log_y=log_y, height=500, width=1200, template=template_theme,
            title=f'<b>DISTRIBUTION OF {feature_col.replace("_", " ").upper()}',
        )
    
    # Customize the layout
    fig.update_layout(
        bargap=0.1,  # Add space between bars
        xaxis_title=f'<b>{feature_col.replace("_", " ").title()}</b>', 
        yaxis_title=f'<b>Count {"<sub>(Log Scale)</sub>" if log_y else ""}</b>',
        showlegend=group_by_target  # Show legend only when grouped by target
    )
    
    # Apply log scale to y-axis if requested
    if log_y:
        fig.update_layout(yaxis_type='log')
    
    # Display the plot
    fig.show()

plot_categorical_feature_distribution(train_df, 'anatom_site_general', group_by_target=True, stack_bars=False, log_y=True)
# plot_categorical_feature_distribution(train_df, 'anatom_site_general', group_by_target=False)

plot_categorical_feature_distribution(train_df, "sex", group_by_target=True, stack_bars=False, log_y=True)
# plot_categorical_feature_distribution(train_df, "sex", group_by_target=False)

# plot_categorical_feature_distribution(train_df, "age_approx", group_by_target=True, stack_bars=False, log_y=True)
plot_categorical_feature_distribution(train_df, "age_approx", group_by_target=False)

# --- Cell Separator ---

def plot_continuous_feature_distribution(
    df: pd.DataFrame, 
    feature_col: str,
    plot_style: str = "histogram",
    feature_readable_name: str | None = None,
    target_col: str = "target",
    target_as_str: bool = True,
    log_y: bool = False, 
    color_sequence: list[str] | None = None,
    template_theme: str = "plotly_white",
    group_by_target: bool = True,
    n_bins: int = 50
) -> None:
    """Plot the distribution of a continuous feature, optionally grouped by the target variable.
    
    This function creates either a histogram or a box plot of the continuous feature distribution,
    with options for log scale, custom color schemes, and grouping by target.
    
    Args:
        df (pd.DataFrame): 
            The input dataframe containing feature and target columns.
        feature_col (str): 
            Name of the continuous feature column to plot.
        plot_style (str, optional):
            Type of plot to create. Either "histogram" or "box".
        feature_readable_name (Optional[str]):
            An option to replace the column name with a readable name for title/axis.
        target_col (str, optional): 
            Name of the target column.
        target_as_str (bool, optional): 
            Whether to convert target labels to strings.
        log_y (bool, optional): 
            Whether to use log scale for y-axis.
        color_sequence (Optional[list[str]], optional): 
            Custom color sequence for the plots. 
            Defaults to None, which uses COLORS for target grouping.
        template_theme (str, optional): 
            Plotly template theme for visual styling.
        group_by_target (bool, optional): 
            Whether to group the plot by target.
        n_bins (int, optional):
            Number of bins for the histogram (only used if plot_style is "histogram").
    Returns:
        None; 
            This function displays a plot and doesn't return anything.
    """
    # Prevent accidental edits to the original dataframe
    _df = df.copy().sort_values(by=[feature_col, target_col]).reset_index(drop=True)
        
    if target_as_str:
        _df[target_col] = _df[target_col].map(LBL_MAP_I2S)
        
    # Set default color sequence if not provided
    if not color_sequence:
        if group_by_target:
            color_sequence = COLORS
        else:
            color_sequence = list(nb_palette.as_hex())
    
    if feature_readable_name is None:
        feature_readable_name = METADATA_COL2NAME.get(feature_col, feature_col.replace("_", " ").title())   
        
    if plot_style == "histogram":
        if group_by_target:
            fig = go.Figure()
            for i, target_value in enumerate(_df[target_col].unique()):
                subset = _df[_df[target_col] == target_value]
                fig.add_trace(go.Histogram(
                    x=subset[feature_col],
                    name=str(target_value),
                    marker_color=color_sequence[i % len(color_sequence)],
                    opacity=0.7,
                    nbinsx=n_bins
                ))
            
            fig.update_layout(
                barmode='overlay',
                title=f"<b>DISTRIBUTION OF '{feature_readable_name.upper()}' BY TARGET</b>",
                height=500, width=1200, template=template_theme
            )
        else:
            fig = px.histogram(
                _df, x=feature_col,
                color_discrete_sequence=[color_sequence[0]],
                log_y=log_y, height=500, width=1200, template=template_theme,
                title=f"<b>DISTRIBUTION OF '{feature_readable_name.upper()}'</b>",
                nbins=n_bins
            )
        
        # Customize the layout
        fig.update_layout(
            xaxis_title=f'<b>{feature_readable_name}</b>', 
            yaxis_title=f'<b>Count {"<sub>(Log Scale)</sub>" if log_y else ""}</b>',
            showlegend=group_by_target  # Show legend only when grouped by target
        )
        
    elif plot_style == "box":
        if group_by_target:
            fig = go.Figure()
            for i, target_value in enumerate(_df[target_col].unique()):
                subset = _df[_df[target_col] == target_value]
                fig.add_trace(go.Box(
                    y=subset[feature_col],
                    name=str(target_value),
                    marker_color=color_sequence[i % len(color_sequence)],
                    boxpoints='outliers',
                    boxmean=True
                ))
            
            fig.update_layout(
                title=f"<b>DISTRIBUTION OF '{feature_readable_name.upper()}' BY TARGET <sub>(includes likely outliers)</sub></b>",
                height=500, width=1200, template=template_theme
            )
        else:
            fig = px.box(
                _df, y=feature_col,
                color_discrete_sequence=color_sequence,
                height=500, width=1200, template=template_theme,
                title=f"<b>DISTRIBUTION OF '{feature_readable_name.upper()}'</b>",
                points='outliers',
            )
        
        # Customize the layout
        fig.update_layout(
            xaxis_title='<b>Target</b>' if group_by_target else '', 
            yaxis_title=f'<b>{feature_readable_name} {"<sub>(Log Scale)</sub>" if log_y else ""}</b>',
            showlegend=group_by_target  # Show legend only when grouped by target
        )
    
    else:
        raise ValueError("Invalid plot_style. Choose either 'histogram' or 'box'.")
    
    # Apply log scale to y-axis if requested (only for histogram)
    if log_y:
        fig.update_layout(yaxis_type='log')
    
    # Display the plot
    fig.show()
    
    
# Lesion Area
plot_continuous_feature_distribution(train_df, 'tbp_lv_areaMM2', plot_style="histogram", log_y=True, group_by_target=True, n_bins=100)

# Lesion Perimeter
plot_continuous_feature_distribution(train_df, 'tbp_lv_perimeterMM', plot_style="box", log_y=True, group_by_target=True, n_bins=100)

# Lesion Diameter
plot_continuous_feature_distribution(train_df, 'clin_size_long_diam_mm', plot_style="box", log_y=True, group_by_target=True)

# # Border Irregularity
# plot_continuous_feature_distribution(train_df, 'tbp_lv_norm_border', plot_style="histogram", log_y=True, group_by_target=True, n_bins=100)

# # Lesion Asymmetry
# plot_continuous_feature_distribution(train_df, 'tbp_lv_symm_2axis', plot_style="histogram", log_y=True, group_by_target=True, n_bins=100)

# # Color Contrast (Delta LB)
# plot_continuous_feature_distribution(train_df, 'tbp_lv_deltaLBnorm', log_y=True, group_by_target=True, n_bins=100)

# # Color Variation
# plot_continuous_feature_distribution(train_df, 'tbp_lv_norm_color, log_y=True, group_by_target=True, n_bins=100)


# # Nevus Confidence Score
# plot_continuous_feature_distribution(train_df, 'tbp_lv_nevi_confidence', log_y=True, group_by_target=True, n_bins=100)

# # Hue Inside Lesion
# plot_continuous_feature_distribution(train_df, 'tbp_lv_H', log_y=True, group_by_target=True, n_bins=100)

# # Luminance Inside Lesion
# plot_continuous_feature_distribution(train_df, 'tbp_lv_L', log_y=True, group_by_target=True, n_bins=100)

# # Color Standard Deviation Mean
# plot_continuous_feature_distribution(train_df, 'tbp_lv_color_std_mean', log_y=True, group_by_target=True, n_bins=100)

# # Lesion Eccentricity
# plot_continuous_feature_distribution(train_df, 'tbp_lv_eccentricity', log_y=True, group_by_target=True, n_bins=100)

# --- Cell Separator ---

def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    subset_percentage: float = 0.1,
    color_col: str = "target",
    target_as_str: bool = True,
    size_col: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    color_sequence: list[str] | None = None,
    template_theme: str = "plotly_white",
    log_x: bool = False,
    log_y: bool = False,
    hover_data: list[str] | None = None
) -> None:
    """Scatter plot to illustrate how two variables impact each other (and others)

    Args:
        df (pd.DataFrame): The input dataframe containing the data to plot.
        x_col (str): Name of the column to use for x-axis.
        y_col (str): Name of the column to use for y-axis.
        subset_percentage (float, optional): What percentage of the dataset to examine (helps with render)
        color_col (str, optional): Name of the column to use for color coding points.
        target_as_str (bool, optional): If the target column is given as a column, turn into str?
        size_col (str, optional): Name of the column to use for sizing points.
        x_label (str, optional): Custom label for x-axis. If None, uses x_col.
        y_label (str, optional): Custom label for y-axis. If None, uses y_col.
        title (str, optional): Title of the plot. If None, generates a default title.
        color_sequence (list[str], optional): Custom color sequence for color coding.
        template_theme (str, optional): Plotly template theme for visual styling.
        log_x (bool, optional): Whether to use log scale for x-axis.
        log_y (bool, optional): Whether to use log scale for y-axis.
        hover_data (list[str], optional): Additional columns to show in hover data.

    Returns:
        None: This function displays a plot and doesn't return anything.
    """
    # Prevent accidental edits to the original dataframe
    _df = df.copy().sort_values(by=[x_col, y_col, color_col]).reset_index(drop=True).sample(frac=subset_percentage)
        
    if target_as_str:
        _df["target"] = _df["target"].map(LBL_MAP_I2S)
    
    for _col in x_col, y_col, color_col, size_col:
        if _col and _df[_col].dtype=="float64":
            _df[_col] = _df[_col].fillna(_df[_col].mean())
    
    # Set default color sequence if not provided
    if not color_sequence:
        if color_col=="target":
            color_sequence = COLORS
        else:
            color_sequence = list(nb_palette.as_hex())
        
    # Set up the scatter plot
    fig = px.scatter(
        _df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        color_discrete_sequence=color_sequence,
        template=template_theme,
        log_x=log_x,
        log_y=log_y,
        hover_data=hover_data
    )
    
    x_col_str = x_label or METADATA_COL2NAME.get(x_col, x_col.replace('_', ' ').title())
    y_col_str = y_label or METADATA_COL2NAME.get(y_col, y_col.replace('_', ' ').title())
    
    
    # Customize the layout
    fig.update_layout(
        title=title or f"<b>Scatter Plot: '{y_col_str}' vs '{x_col_str}'</b>",
        xaxis_title=f"<b>{x_col_str}</b>",
        yaxis_title=f"<b>{y_col_str}</b>",
        height=600,
        width=1000
    )

    # Update axes to show log scale in label if applied
    if log_x:
        fig.update_xaxes(title=f"<b>{x_col_str} <sub>(Log Scale)</sub></b>")
    if log_y:
        fig.update_yaxes(title=f"<b>{y_col_str} <sub>(Log Scale)</sub></b>")

    # Display the plot
    fig.show()
    
plot_scatter(train_df, "tbp_lv_area_perim_ratio", "clin_size_long_diam_mm", color_col="anatom_site_general", size_col="tbp_lv_areaMM2", subset_percentage=0.001)

# --- Cell Separator ---

# Categorical columns
CAT_COLS = [
    # male      265546
    # female    123996
    # NaN        11517
    'sex',  
    
    # posterior torso    121902
    # lower extremity    103028
    # anterior torso      87770
    # upper extremity     70557
    # head/neck           12046
    # NaN                  5756
    'anatom_site_general', 
    
    # TBP tile: close-up    401059
    'image_type',
    
    # 3D: XP       285903
    # 3D: white    115156
    'tbp_tile_type', 

    # Torso Back Top Third       71112
    # Torso Front Top Half       63350
    # Torso Back Middle Third    46185
    # Left Leg - Lower           27428
    # Right Leg - Lower          25208
    # Torso Front Bottom Half    24360
    # Left Leg - Upper           23673
    # Right Leg - Upper          23034
    # Right Arm - Upper          22972
    # Left Arm - Upper           22816
    # Head & Neck                12046
    # Left Arm - Lower           11939
    # Right Arm - Lower          10636
    # Unknown                     5756
    # Torso Back Bottom Third     4596
    # Left Leg                    1974
    # Right Leg                   1711
    # Left Arm                    1593
    # Right Arm                    601
    # Torso Front                   60
    # Torso Back                     9
    'tbp_lv_location', 
    
    # Torso Back     121902
    # Torso Front     87770
    # Left Leg        53075
    # Right Leg       49953
    # Left Arm        36348
    # Right Arm       34209
    # Head & Neck     12046
    # Unknown          5756
    'tbp_lv_location_simple', 
    
    # Memorial Sloan Kettering Cancer Center                                             129068
    # Department of Dermatology, Hospital Clínic de Barcelona                            105724
    # University Hospital of Basel                                                       65218
    # Frazer Institute, The University of Queensland, Dermatology Research Centre        51768
    # ACEMID MIA                                                                         28665
    # ViDIR Group, Department of Dermatology, Medical University of Vienna               12640
    # Department of Dermatology, University of Athens, Andreas Syggros Hospital [...]    7976
    'attribution',
    
    # CC-BY       188812
    # CC-BY-NC    183582
    # CC-0         28665
    'copyright_license', 
    
    # NaN        401006
    # 0/mm^2         22
    # <1/mm^2        19
    # >4/mm^2         4
    # 1/mm^2          3
    # 3/mm^2          2
    # 2/mm^2          2
    # 4/mm^2          1
    'mel_mitotic_index'
]

# Basically these columns are hierarchical.
# The higher the number the more specific the diagnosis provided.
# The iddx_full column concatanates non NaN values in the 5 iddx columns with '::' between.
STR_COLS = [
    'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5'
]

NUM_COLS = [
    
    # 0    400666
    # 1       393
    'target',  # int

    # 55.0    58123
    # 65.0    54946
    # 60.0    54109
    # 50.0    47924
    # 70.0    39775
    # 40.0    31297
    # 75.0    30801
    # 45.0    23580
    # 80.0    21096
    # 35.0    11543
    # 30.0    10400
    # 85.0     8847
    # 25.0     3433
    # NaN      2798
    # 20.0     1742
    # 15.0      644
    # 5.0         1
    'age_approx',  # Should be an 'int' but the NaN casts it to float
    
    # 1758 unique
    'clin_size_long_diam_mm',  # float
    
    # Almost 400,000 unique for all of these
    'tbp_lv_A',   # float
    'tbp_lv_Aext',   # float 
    'tbp_lv_B',   # float
    'tbp_lv_Bext',   # float 
    'tbp_lv_C',   # float 
    'tbp_lv_Cext',   # float 
    'tbp_lv_H',   # float 
    'tbp_lv_Hext',   # float 
    'tbp_lv_L',   # float
    'tbp_lv_Lext',   # float 
    
    # 8000 or so unique
    'tbp_lv_areaMM2',  # float
    
    # 160,000 unique
    'tbp_lv_area_perim_ratio',  # float 
    
    # 26187 at 0.0 and the rest spread
    'tbp_lv_color_std_mean',  # float
    
    # nearly all unique
    'tbp_lv_deltaA',  # float
    'tbp_lv_deltaB',  # float
    'tbp_lv_deltaL',  # float
    'tbp_lv_deltaLB',  # float
    'tbp_lv_deltaLBnorm',  # float
    'tbp_lv_eccentricity',  # float
    
    # 80000 unique
    'tbp_lv_minorAxisMM',  # float 
    
    # nearly all unique (heavily stacked near 100%)
    'tbp_lv_nevi_confidence',  # float 
    
    # nearly all unique (but a weird bunch at exactly 10.0)
    'tbp_lv_norm_border',  # float 
    
    # nearly all unique (but a weird bunch at exactly 0.0 and 10.0)
    'tbp_lv_norm_color',  # float 
    
    # 9500 or so unique
    'tbp_lv_perimeterMM',  # float 
    
    # nearly all unique (but a huge chunk at 0.0 (40k))
    'tbp_lv_radial_color_std_max',  # float 
    
    # nearly all unique for both
    'tbp_lv_stdL',  # float 
    'tbp_lv_stdLExt',  # float 
    
    # 75000 unique
    'tbp_lv_symm_2axis',
    
    # 36 unique
    'tbp_lv_symm_2axis_angle',  # int
    
    # nearly all unique
    'tbp_lv_x',  # float
    'tbp_lv_y',  # float
    'tbp_lv_z',  # float
    
    # 400996 NaN (nearly all)
    'mel_thick_mm',  # float
    
    # A bunch stacked at 100 (and near 100) but mostly unique
    'tbp_lv_dnn_lesion_confidence'  # float
]

ID_COLS = ['isic_id', 'patient_id', 'lesion_id']
COLS_TO_IGNORE = ID_COLS + STR_COLS + ["copyright_license", "attribution", "image_type", "mel_thick_mm", "mel_mitotic_index"]
COLS_TO_ONE_HOT = [_c for _c in CAT_COLS if _c not in COLS_TO_IGNORE]
TARGET_COL = "target"

# --- Cell Separator ---

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataframe.

    For numerical columns, fill NaNs with the median value.
    For categorical columns, fill NaNs with the most frequent value (mode).

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    # Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Handle missing values in numerical columns
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Handle missing values in categorical columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df

def one_hot_encode(df: pd.DataFrame, columns_to_encode: list[str]) -> pd.DataFrame:
    """Perform one-hot encoding on specified categorical columns.

    Args:
        df (pd.DataFrame): Input dataframe
        columns_to_encode (list[str]): List of column names to one-hot encode

    Returns:
        pd.DataFrame: Dataframe with specified columns one-hot encoded
    """
    # Filter columns that are actually present in the dataframe
    columns_to_encode = [col for col in columns_to_encode if col in df.columns]

    # Perform one-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cols = encoder.fit_transform(df[columns_to_encode])
    
    # Create new column names for encoded features
    new_columns = encoder.get_feature_names_out(columns_to_encode)
    
    # Create a new dataframe with encoded features
    encoded_df = pd.DataFrame(encoded_cols, columns=new_columns, index=df.index)
    
    # Concatenate the original dataframe with the encoded features
    result_df = pd.concat([df.drop(columns_to_encode, axis=1), encoded_df], axis=1)
    
    return result_df

def scale_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical features using StandardScaler.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with scaled numerical features
    """
    # Identify numerical columns (excluding the target column)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [col for col in num_cols if col != TARGET_COL]

    # Scale numerical features
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

def handle_age_approx(df: pd.DataFrame) -> pd.DataFrame:
    """Handle the 'age_approx' column by rounding it to the nearest integer.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with 'age_approx' rounded to nearest integer
    """
    if 'age_approx' in df.columns:
        df['age_approx'] = df['age_approx'].round().astype('Int64')
    return df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input dataframe for machine learning tasks.

    This function performs the following steps:
    1. Handle missing values
    2. One-hot encode categorical variables
    3. Scale numerical features
    4. Handle the 'age_approx' column
    5. Drop unnecessary columns

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Preprocessed dataframe ready for machine learning tasks
    """
    # Just in case
    _df = df.copy()
    
    # Handle missing values
    clr_print("Handling missing values")
    _df = handle_missing_values(_df)

    # One-hot encode categorical variables
    clr_print("One hot encoding categorical variables")
    columns_to_encode = [col for col in COLS_TO_ONE_HOT if col in _df.columns]
    _df = one_hot_encode(_df, columns_to_encode)
    
    # Scale numerical features
    clr_print("Scaling numerical features")
    _df = scale_numerical_features(_df)

    # Drop unnecessary columns
    clr_print("Dropping unnecessary remaining columns")
    columns_to_drop = [col for col in COLS_TO_IGNORE+COLS_TO_ONE_HOT if col in _df.columns]
    return _df.drop(columns=columns_to_drop)

preprocessed_train_df = preprocess_df(train_df)

# --- Cell Separator ---

def plot_target_correlation(
    df: pd.DataFrame,
    target_col: str = 'target',
    n_top_features: int = 30,
    color_sequence: list[str] | None = None,
    template_theme: str = "plotly_white"
) -> None:
    """Create a correlation plot showing the top correlated features with the target variable.

    Args:
        df (pd.DataFrame): 
            The input dataframe containing feature and target columns.
        target_col (str, optional): 
            Name of the target column.
        n_top_features (int, optional): 
            Number of top correlated features to display.
        color_sequence (list[str], optional): 
            Custom color sequence for the plot.
            Defaults to None, which uses a blue to red color scale.
        template_theme (str, optional): 
            Plotly template theme for visual styling.

    Returns:
        None: This function displays a plot and doesn't return anything.
    """
    # Calculate correlations
    correlations = df.corr()[target_col]
    
    # Sort by absolute correlation value
    correlations_abs = correlations.abs().sort_values(ascending=False)
    
    # Select top correlated features (excluding the target itself)
    top_correlations = correlations[correlations_abs.index[1:n_top_features+1]]
    
    # Prepare data for plotting
    feature_names = top_correlations.index
    correlation_values = top_correlations.values
    
    # Set up color scale
    if color_sequence is None:
        color_sequence = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']
    
    # Create the bar plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=feature_names,
        x=correlation_values,
        orientation='h',
        marker=dict(
            color=correlation_values,
            colorscale=color_sequence,
            colorbar=dict(title="Correlation"),
        )
    ))
    
    # Customize the layout
    fig.update_layout(
        title=f"<b>Top {n_top_features} Features Correlated with {target_col.capitalize()}</b>",
        xaxis_title="<b>Correlation Coefficient</b>",
        yaxis_title="<b>Feature</b>",
        height=800,
        width=1200,
        template=template_theme,
    )
    
    # Add vertical line at x=0 for reference
    fig.add_shape(
        type="line",
        x0=0, y0=-0.5,
        x1=0, y1=len(feature_names) - 0.5,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Display the plot
    fig.show()
    
plot_target_correlation(preprocessed_train_df)

# --- Cell Separator ---

def compare_area_to_patient_mean(df: pd.DataFrame, area_col: str = "tbp_lv_areaMM2", patient_col: str = "patient_id") -> pd.DataFrame:
    """
    Create a new column that quantifies how much bigger a particular area is compared to the mean for the patient.

    Args:
        df (pd.DataFrame): The input dataframe.
        area_col (str, optional): The name of the column containing the area measurements. Defaults to "tbp_lv_areaMM2".
        patient_col (str, optional): The name of the column containing patient IDs. Defaults to "patient_id".

    Returns:
        pd.DataFrame: The input dataframe with a new column added.
    """
    # Copy
    _df = df.copy()
    
    # Calculate the mean area for each patient
    patient_mean_area = _df.groupby(patient_col)[area_col].transform('mean')
    
    # Calculate the ratio of each area to the patient's mean area
    _df[f'{area_col}_ratio_to_patient_mean'] = _df[area_col] / patient_mean_area
    
    # Calculate the percentage difference from the patient's mean area
    _df[f'{area_col}_pct_diff_from_patient_mean'] = (_df[area_col] - patient_mean_area) / patient_mean_area * 100
    
    return _df

# Possible new features could be created...
compare_area_to_patient_mean(train_df)[["tbp_lv_areaMM2_pct_diff_from_patient_mean", "target"]].corr()

# --- Cell Separator ---

def plot_img_from_id(
    isic_id: str,
    hdf5_file_path: str = "/kaggle/input/isic-2024-challenge/train-image.hdf5",
    figsize: tuple = (6, 6)
):
    """Plot an image from the HDF5 dataset file by specifying an ISIC ID.
    
    Args:
        isic_id (str): The ID of the image to plot.
        hdf5_file_path (str): The path to the HDF5 file.
        figsize (tuple): The size of the figure (width, height) in inches.
    
    Returns:
        None; Plots the image
    
    Example Usage:
        plot_img_from_id('ISIC_0000000')
    """
    # Load the image using the existing function
    img = load_img_from_hdf5(isic_id, hdf5_file_path)
    
    # Create a new figure with the specified size
    plt.figure(figsize=figsize)
    
    # Set the title to the ISIC ID
    plt.title(f"ISIC ID: {isic_id}", fontweight="bold")
    
    # Display the image
    plt.imshow(img)
    
    # Remove axis and update plot layout
    plt.axis('off')
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
plot_img_from_id(train_df.sample(1).isic_id.values[0])

# --- Cell Separator ---

def add_colored_border(img: np.ndarray, color: tuple[int] | str, border_width: int = 3) -> np.ndarray:
    """Add a colored border to an image.
    
    Args:
        img (np.ndarray): Input image as a numpy array.
        color (tuple[int] | str): Border color in RGB format OR a HEX string.
        border_width (int, optional): Width of the border in pixels.
    
    Returns:
        np.ndarray: Image with added border.
    """
    
    # Ensure color is RGB
    if isinstance(color, str) and color.startswith("#"):
        color = ImageColor.getcolor(color, "RGB")
        
    # Draw border
    bordered_img = cv2.copyMakeBorder(
        src=img, 
        top=border_width, 
        bottom=border_width, 
        left=border_width, 
        right=border_width, 
        borderType=cv2.BORDER_CONSTANT, 
        value=color
    )
    return bordered_img


def plot_patient_images(
    patient_isic_ids: list[str],
    labels: list[int] | None = None,
    patient_id: str | None = None,
    hdf5_file_path: str = "/kaggle/input/isic-2024-challenge/train-image.hdf5",
    max_images: int = 60,
    images_per_row: int = 10,
    fig_width: int = 20,
):
    """Plot multiple images for a patient in a tiled layout with colored borders
    
    Args:
        patient_isic_ids (list[str]): 
            List of ISIC IDs for the patient's images.
        labels (list[int], optional): 
            The list of labels which will be used to color code malignant images.
                - Malignant images will be outlined with red (COLORS[1])
                - Benign images will be outlined with green (COLORS[0])
        patient_id (str, optional): 
            The patient id the isic_ids belong to.
        hdf5_file_path (str, optional): 
            The path to the HDF5 file.
        max_images (int, optional): 
            Maximum number of images to display.
        images_per_row (int, optional): 
            Number of images to display in each row.
        fig_width (int, optional): 
            The size of the figure width
    
    Returns:
        None; 
            plots the tiled images
    """
    # Limit the number of images to plot
    total_num_images = len(patient_isic_ids)
    patient_isic_ids = patient_isic_ids[:max_images]
    num_of_images_to_plot = len(patient_isic_ids)
    
    # Calculate the number of rows needed and figsize
    num_rows = math.ceil(num_of_images_to_plot / images_per_row)
    figsize = (fig_width, int(2.666*num_rows))
    
    # Create the figure
    fig = plt.figure(figsize=figsize)
    plt.suptitle(f"IMAGES FOR PATIENT: {patient_id}  (showing {num_of_images_to_plot} out of {total_num_images} images)", fontsize=16, fontweight="bold")
    
    # Process images in batches
    for i, isic_id in enumerate(patient_isic_ids):
        # Calculate the subplot position
        position = i + 1
        # Create a new subplot
        plt.subplot(num_rows, images_per_row, position)
        plt.title(f"{isic_id}{' - '+LBL_MAP_I2S[labels[i]] if labels is not None else ''}", fontsize=8 if labels is None else 7)
        
        # Load the image
        img = load_img_from_hdf5(isic_id, hdf5_file_path)
        
        # Add colored border
        if labels is not None:
            img = add_colored_border(img, COLORS[labels[i]])
        
        # Display the image
        plt.imshow(img)
        plt.axis('off')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    

DEMO_PATIENT_DF = train_df[train_df.patient_id==train_df[train_df.target==1]["patient_id"].sample(1).values[0]].sort_values(by=["target", "isic_id"], ascending=False).reset_index(drop=True)

display_hr(True, True)
clr_print(f"# OF MALIGNANT TILES:  <code>{DEMO_PATIENT_DF.target.sum()}</code>")
display_hr(True, True)
display(DEMO_PATIENT_DF)
display_hr(True, True)

plot_patient_images(
    patient_isic_ids=DEMO_PATIENT_DF.isic_id.to_list(),
    labels=DEMO_PATIENT_DF.target.to_list(),
    patient_id=DEMO_PATIENT_DF.patient_id[0]
)

# --- Cell Separator ---

def plot_image_batch(
    isic_ids: list[str],
    labels: list[int] | None = None,
    batch_description: str | None = None,
    hdf5_file_path: str = "/kaggle/input/isic-2024-challenge/train-image.hdf5",
    max_images: int = 24,
    images_per_row: int = 8,
    fig_width: int = 20,
):
    """Plot multiple images as a batch.
    
    Args:
        isic_ids (list[str]): 
            List of ISIC IDs for the batch.
        labels (list[int], optional): 
            The list of labels which will be used to color code malignant images.
                - Malignant images will be outlined with red (COLORS[1])
                - Benign images will be outlined with green (COLORS[0])
        batch_description (str, optional):
            The description for what we are trying to visualize within the batch
        hdf5_file_path (str, optional): 
            The path to the HDF5 file.
        max_images (int, optional): 
            Maximum number of images to display.
        images_per_row (int, optional): 
            Number of images to display in each row.
        fig_width (int, optional): 
            The size of the figure width
    
    Returns:
        None; 
            plots the tiled images
    """
    # Limit the number of images to plot
    total_num_images = len(isic_ids)
    isic_ids = isic_ids[:max_images]
    num_of_images_to_plot = len(isic_ids)
    
    # Calculate the number of rows needed and figsize
    num_rows = math.ceil(num_of_images_to_plot / images_per_row)
    figsize = (fig_width, int(3*num_rows))
    
    # Create the figure
    fig = plt.figure(figsize=figsize)
    plt.suptitle(f"IMAGE BATCH - {batch_description or '<UNK>'}", fontsize=14, fontweight="bold")
    
    # Process images in batches
    for i, isic_id in enumerate(isic_ids):
        # Calculate the subplot position
        position = i + 1
        
        # Create a new subplot
        plt.subplot(num_rows, images_per_row, position)
        plt.title(f"{isic_id}{' - '+LBL_MAP_I2S[labels[i]] if labels is not None else ''}", fontsize=8 if labels is None else 7)
        
        # Load the image
        img = load_img_from_hdf5(isic_id, hdf5_file_path)
        
        # Add colored border
        if labels is not None:
            img = add_colored_border(img, COLORS[labels[i]])
        
        # Display the image
        plt.imshow(img)
        plt.axis('off')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    
    
def get_balanced_df(df: pd.DataFrame, balance_col: str, shuffle: bool = True) -> pd.DataFrame:
    """Balances the DataFrame by sampling an equal number of rows from each category.
    
    Args:
        df (pd.DataFrame): The input DataFrame to balance.
        balance_col (str): The column name to balance by.
        shuffle (bool): Whether to shuffle the resulting dataframe. Defaults to True.
    
    Returns:
        pd.DataFrame: A balanced DataFrame with an equal number of rows from each category.
    """
    # Create safe edit copy
    _df = df.copy()
    
    # Get value counts and determine the size of the smallest category
    min_col_count = _df[balance_col].value_counts().min()
    
    # Sample from each category and concatenate all balanced samples
    balanced_df = pd.concat([
        _df[_df[balance_col] == category].sample(n=min_col_count, replace=False) 
        for category in df[balance_col].unique()
    ], ignore_index=True)
    
    
    # Shuffle the combined DataFrame if required
    if shuffle:
        balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    
    return balanced_df


def process_batches(
    df: pd.DataFrame, 
    label_split: str, 
    feature_col: str
) -> dict[str, dict[str, list[Any]]]:
    """Processes batches from the training DataFrame based on label split criteria.

    Args:
        df (pd.DataFrame): 
            The input DataFrame.
        label_split (str): 
            The criteria for splitting labels ("malignant", "benign", "half", or "random").
        feature_col (str): 
            The column name to group by for batch processing.

    Returns:
        dict[str, dict[str, list[Any]]]: 
            A dictionary with feature strings as keys and dictionaries of ISIC IDs and labels as values.
    """
    batches = {}
    for feature_str, _df in train_df.groupby(feature_col):
        if label_split == "malignant":
            _df = _df[_df["target"] == 1]
        elif label_split == "benign":
            _df = _df[_df["target"] == 0]
        elif label_split == "half":
            _df = get_balanced_df(_df, "target")
        batches[feature_str] = {"isic_ids": _df["isic_id"].to_list(), "labels": _df["target"].tolist()}
    return batches


def plot_batches(
    feature_col: str, 
    label_split: str,
    df: pd.DataFrame | None = None,
    batches: dict[str, dict[str, list[Any]]] | None = None, 
) -> None:
    """Displays and plots image batches with descriptions.

    Args:
        feature_col (str): 
            The column name to describe the batches.
        label_split (str): 
            The label split style used.
        df (pd.DataFrame, optional): 
            The input DataFrame.
        batches (dict[str, dict[str, list[Any]]], optional): 
            The dictionary containing batch data to display and plot.
            
    Raises:
        ValueError:
            If the input data is not correctly passed.
    
    Returns:
        None; Plots...
    """
    
    if df is None and batches is None:
        raise ValueError("\n... One of `df` or `batches` must be provided as an input ...\n")
    elif batches is None:
        batches = process_batches(df, label_split, feature_col)
    
    display_hr(True, True)
    clr_print(METADATA_COL2DESC[feature_col])
    for feature_str, feature_batch in batches.items():
        display_hr(True, True)
        plot_image_batch(
            **feature_batch, 
            batch_description=f"{METADATA_COL2NAME[feature_col]} - '{feature_str}' - SPLIT STYLE={label_split}"
        )
        display_hr(True, True)
        

LABEL_SPLIT = "half"  # malignant, benign, half, random
DEMO_FEAT_COL = "anatom_site_general"

DEMO_BATCHES = process_batches(train_df, LABEL_SPLIT, DEMO_FEAT_COL)
plot_batches(DEMO_FEAT_COL, LABEL_SPLIT, batches=DEMO_BATCHES)

plot_batches("tbp_tile_type", LABEL_SPLIT, df=train_df)

# --- Cell Separator ---

# # train_df[train_df.target==1]["tbp_lv_areaMM2"].describe()
# tbp_lv_areaMM2_thresh = 10.0

# random_ss_df = train_df[["isic_id", "target"]].copy()
# random_ss_df["target"] = [random.random() for i in range(len(random_ss_df))]
# train_ss_df = train_df[["isic_id", "target"]].copy()
# train_ss_df["target"] = train_df["tbp_lv_areaMM2"]>tbp_lv_areaMM2_thresh

# clr_print("AREA THRESH PREDICTION SCORE ON TRAINING DATA:")
# print(score(train_df[["isic_id", "target"]].copy(), train_ss_df, row_id_column_name="isic_id"))

# clr_print("<br>RANDOM PREDICTION SCORE ON TRAINING DATA:")
# print(score(train_df[["isic_id", "target"]].copy(), random_ss_df, row_id_column_name="isic_id"))

# ss_df["target"] = test_df["tbp_lv_areaMM2"]>tbp_lv_areaMM2_thresh
# ss_df.to_csv("submission.csv", index=False)
# ss_df

# --- Cell Separator ---

def create_additional_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create additional features based on domain knowledge and existing features.

    Args:
        df (pd.DataFrame): Input dataframe containing original and engineered features.

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple containing:
            - The dataframe with additional features added
            - A list of new numerical column names
    """
    additional_num_cols = []

    # 1. Color Variance Ratio
    #    - This feature compares the color variance within the lesion to the color variance of the surrounding skin.
    df["color_variance_ratio"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_stdLExt"]
    

    # 2. Border Color Interaction
    #    - This interaction term combines border irregularity with color variation.
    df["border_color_interaction"] = df["tbp_lv_norm_border"] * df["tbp_lv_norm_color"]

    # 3. Size Color Contrast Ratio
    #    - This ratio might help identify large lesions with low contrast, or small lesions with high contrast.
    df["size_color_contrast_ratio"] = df["clin_size_long_diam_mm"] / df["tbp_lv_deltaLBnorm"]

    # 4. Age Normalized Nevi Confidence
    #    - This feature adjusts the nevus confidence score by age.
    df["age_normalized_nevi_confidence"] = df["tbp_lv_nevi_confidence"] / df["age_approx"]

    # 5. Color Asymmetry Index
    #    - This combines color asymmetry with border asymmetry.
    df["color_asymmetry_index"] = df["tbp_lv_radial_color_std_max"] * df["tbp_lv_symm_2axis"]

    # 6. 3D Volume Approximation
    #    - This attempts to approximate the volume of the lesion in 3D space.
    df["3d_volume_approximation"] = df["tbp_lv_areaMM2"] * np.sqrt(df["tbp_lv_x"]**2 + df["tbp_lv_y"]**2 + df["tbp_lv_z"]**2)

    # 7. Color Range
    #     - This feature captures the total range of color difference across all color dimensions.
    df["color_range"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs() + (df["tbp_lv_A"] - df["tbp_lv_Aext"]).abs() + (df["tbp_lv_B"] - df["tbp_lv_Bext"]).abs()

    # 8. Shape Color Consistency
    #    - This interaction term might help identify lesions that are both elongated and have inconsistent coloration.
    df["shape_color_consistency"] = df["tbp_lv_eccentricity"] * df["tbp_lv_color_std_mean"]

    # 9. Border Length Ratio
    #    - This ratio compares the actual perimeter to the perimeter of a perfect circle with the same area.
    df["border_length_ratio"] = df["tbp_lv_perimeterMM"] / (2 * np.pi * np.sqrt(df["tbp_lv_areaMM2"] / np.pi))

    # 10. Age Size Symmetry Index
    #    - This composite feature combines age, size, and asymmetry.
    df["age_size_symmetry_index"] = df["age_approx"] * df["clin_size_long_diam_mm"] * df["tbp_lv_symm_2axis"]
    
    additional_num_cols += [
        "color_variance_ratio", "border_color_interaction", "size_color_contrast_ratio", 
        "age_normalized_nevi_confidence", "color_asymmetry_index", "3d_volume_approximation", 
        "color_range", "shape_color_consistency", "border_length_ratio", "age_size_symmetry_index"
    ]
    return df, additional_num_cols


def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Perform comprehensive feature engineering on the input dataframe for skin cancer detection.
    
    Code originally from here --> https://www.kaggle.com/code/vyacheslavbolotin/ensemble-lgbm-cat-with-new-features
    I have made a function out of the above code and done my best to explain things in more detail.

    This function creates new features based on existing ones, potentially improving
    the model's ability to detect skin cancer. It includes various geometric, color,
    and composite features that may be indicative of malignancy.

    Args:
        df (pd.DataFrame): Input dataframe containing original features.

    Returns:
        Tuple[pd.DataFrame, list[str], list[str]]: A tuple containing:
            - The dataframe with new features added
            - A list of new numerical column names
            - A list of new categorical column names
    """
    
    ### [Original feature engineering code from notebook] ###
    
    # Geometric features
    #   - This ratio can help identify irregular growth patterns typical in melanomas
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    #   - A measure of compactness; melanomas often have more irregular shapes
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    #   - Another measure of shape irregularity; higher values may indicate more complex borders
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]

    # Color-based features
    #   - Contrast between lesion and surrounding skin can be indicative of malignancy
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    #   - Overall color difference in 3D color space; larger differences may suggest malignancy
    df["lesion_color_difference"] = np.sqrt(df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
    #   - Measure of color consistency; benign lesions often have more uniform color
    df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]

    # Composite features
    #   - Combines border irregularity and asymmetry, both indicators of potential malignancy
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    #   - 3D position might correlate with certain high-risk body areas
    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2) 
    #   - Combines color and brightness differences, potentially highlighting atypical lesions
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    #   - More specific anatomical location might correlate with cancer risk
    df["combined_anatomical_site"] = df["anatom_site_general"] + "_" + df["tbp_lv_location"]
    #   - Interaction between symmetry and border regularity; asymmetric lesions with irregular borders are more suspicious
    df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    #   - Measure of color consistency within the lesion compared to surrounding skin
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
    #   - Larger lesions in older patients might be more concerning
    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    #   - Interaction between hue and color variation might highlight atypical pigmentation
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    #   -  Composite score combining border, color, and shape irregularities
    df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
    #   - Overall measure of shape complexity
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    #   - Comprehensive measure of color contrast
    df["color_contrast_index"] = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLBnorm"]
    #   - Log transformation can help handle skewed size distributions
    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    #   - Size relative to patient age; rapid growth might be more concerning in younger patients
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    #   - Average hue might indicate overall pigmentation level
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    #   - Standard deviation of color contrast across different color dimensions
    df["std_dev_contrast"] = np.sqrt((df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
    #   - Composite index combining color variation, shape, and symmetry
    df["color_shape_composite_index"] = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df["tbp_lv_symm_2axis"]) / 3
    #   - Orientation of the lesion in 3D space might correlate with certain types of growths
    df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    #   - Average color difference across different color dimensions
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    #   - Interaction between symmetry and perimeter; large asymmetric lesions might be more concerning
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    #    - Comprehensive index combining multiple aspects of lesion characteristics
    df["comprehensive_lesion_index"] = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df["tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4
    
    new_num_cols = [
        "lesion_size_ratio", "lesion_shape_index", "hue_contrast",
        "luminance_contrast", "lesion_color_difference", "border_complexity",
        "color_uniformity", "3d_position_distance", "perimeter_to_area_ratio",
        "lesion_visibility_score", "symmetry_border_consistency", "color_consistency",
        "size_age_interaction", "hue_color_std_interaction", "lesion_severity_index", 
        "shape_complexity_index", "color_contrast_index", "log_lesion_area",
        "normalized_lesion_size", "mean_hue_difference", "std_dev_contrast",
        "color_shape_composite_index", "3d_lesion_orientation", "overall_color_difference",
        "symmetry_perimeter_interaction", "comprehensive_lesion_index",
    ]
    new_cat_cols = ["combined_anatomical_site"]
    
    # Call our new function for additional features
    df, additional_num_cols = create_additional_features(df)

    # Update new_num_cols to include additional features
    
    new_num_cols.extend(additional_num_cols)

    return df, new_num_cols, new_cat_cols


def get_feature_columns(new_num_cols: list[str], new_cat_cols: list[str]) -> tuple[list[str], list[str]]:
    """Combine base feature columns with newly engineered features.

    Args:
        new_num_cols (list[str]): List of new numerical column names.
        new_cat_cols (list[str]): List of new categorical column names.

    Returns:
        Tuple[list[str], list[str]]: A tuple containing:
            - List of all numerical column names
            - List of all categorical column names
    """
    base_num_cols = [
        'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 
        'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 
        'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 
        'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB',
        'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
        'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
        'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
        'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
        'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
    ]
    base_cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple", "anatom_site_general"]
    
    num_cols = base_num_cols + new_num_cols
    cat_cols = base_cat_cols + new_cat_cols
    
    return num_cols, cat_cols


def handle_missing_values(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """Handle missing and infinite values in numerical columns using robust imputation.

    This function replaces infinity values with NaN, then uses median imputation
    for missing values. It also includes a fallback to mean imputation if median
    fails due to all-NaN slices.

    Args:
        df (pd.DataFrame): Input dataframe.
        num_cols (list[str]): List of numerical column names.

    Returns:
        pd.DataFrame: Dataframe with missing and infinite values handled.
    """
    df = df.copy()  # Create a copy to avoid modifying the original dataframe

    for col in num_cols:
        # Replace infinity values with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Check if the column has any non-NaN values
        if df[col].notna().any():
            # Use median imputation
            imputer = SimpleImputer(strategy='median')
            try:
                df[col] = imputer.fit_transform(df[[col]])
            except ValueError:
                # If median imputation fails, fall back to mean imputation
                print(f"Warning: Median imputation failed for column {col}. Using mean imputation instead.")
                imputer = SimpleImputer(strategy='mean')
                df[col] = imputer.fit_transform(df[[col]])
        else:
            # If all values are NaN, fill with 0 or another appropriate value
            print(f"Warning: All values in column {col} are NaN. Filling with 0.")
            df[col] = 0

    return df


def encode_categorical_features(df: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """Encode categorical features using OrdinalEncoder.

    Args:
        df (pd.DataFrame): Input dataframe.
        cat_cols (list[str]): List of categorical column names.

    Returns:
        tuple[pd.DataFrame, OrdinalEncoder]: A tuple containing:
            - Dataframe with encoded categorical features
            - Fitted OrdinalEncoder object
    """
    category_encoder = OrdinalEncoder(
        categories='auto',
        dtype=int,
        handle_unknown='use_encoded_value',
        unknown_value=-2,
        encoded_missing_value=-1,
    )
    
    df[cat_cols] = category_encoder.fit_transform(df[cat_cols])
    return df, category_encoder


def create_group_kfolds(
    df: pd.DataFrame,
    n_splits: int = 5,
    target_col: str = "target",
    group_col: str = "patient_id",
    random_state: int | None = None
) -> pd.DataFrame:
    """Create fold assignments for GroupKFold cross-validation.

    This function adds a 'fold' column to the input dataframe, assigning each row
    to a specific fold while ensuring that all data from the same group (e.g., patient)
    stays in the same fold.

    Args:
        df (pd.DataFrame): 
            Input dataframe.
        n_splits (int, optional): 
            Number of folds.
        target_col (str, optional): 
            Name of the target column.
        group_col (str, optional):
            Name of the column to use for grouping.
        random_state (int, optional): 
            Random state for reproducibility. 
            If None, the splits will not be shuffled.

    Returns:
        pd.DataFrame: The input dataframe with an additional 'fold' column.

    Note:
        If random_state is provided, it will use KFold with shuffle=True instead of GroupKFold,
        as GroupKFold does not support shuffling.
    """
    df = df.copy()  # Create a copy to avoid modifying the original dataframe
    df["fold"] = -1  # Initialize fold column with -1

    if random_state is not None:
        # Use KFold with shuffling if random_state is provided
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_method = kf.split(df)
    else:
        # Use GroupKFold without shuffling
        gkf = GroupKFold(n_splits=n_splits)
        split_method = gkf.split(df, df[target_col], groups=df[group_col])

    # Assign folds
    for fold, (_, val_idx) in enumerate(split_method):
        df.loc[val_idx, "fold"] = fold

    return df


# Get updated dataframes and num/cat cols
train_df, new_num_cols, new_cat_cols = feature_engineering(train_df.copy())
test_df, _, _ = feature_engineering(test_df.copy())

# update the feature columns
num_cols, cat_cols = get_feature_columns(new_num_cols, new_cat_cols)

# Handle missing values (ordinal)
train_df = handle_missing_values(train_df, num_cols)
test_df = handle_missing_values(test_df, num_cols)

# Encode categorical features
train_df, category_encoder = encode_categorical_features(train_df, cat_cols)
test_df[cat_cols] = category_encoder.transform(test_df[cat_cols])

# Combine all columns
train_cols = num_cols + cat_cols

# Add fold identification to train dataframe
train_df = create_group_kfolds(train_df, n_splits=5)
train_df

# --- Cell Separator ---

def generate_lgb_params(random_seed: int | None = None) -> dict[str, Any]:
    """Generate LightGBM parameters using a structured random search approach.

    This function generates a set of LightGBM parameters by randomly selecting values
    from predefined ranges for each hyperparameter. It provides a balance between
    exploration of the parameter space and control over the ranges.

    Args:
        random_seed (int, optional): 
            Seed for random number generator. If None, uses system time.

    Returns:
        dict[str, Any]: A dictionary of LightGBM parameters.
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Define parameter ranges (instead of specific values)
    param_ranges = {
        'n_estimators': (1400, 2400),
        'learning_rate': (0.001, 0.003),
        'num_leaves': (16, 40),
        'min_data_in_leaf': (16, 60),
        'pos_bagging_fraction': (0.74, 0.78),
        'neg_bagging_fraction': (0.04, 0.08),
        'feature_fraction': (0.5, 0.78),
        'lambda_l1': (0.1, 0.4),
        'lambda_l2': (0.7, 3.0)
    }

    # Generate random values for each parameter
    params = {
        'n_estimators': int(random.uniform(*param_ranges['n_estimators'])),
        'learning_rate': random.uniform(*param_ranges['learning_rate']),
        'num_leaves': int(random.uniform(*param_ranges['num_leaves'])),
        'min_data_in_leaf': int(random.uniform(*param_ranges['min_data_in_leaf'])),
        'pos_bagging_fraction': random.uniform(*param_ranges['pos_bagging_fraction']),
        'neg_bagging_fraction': random.uniform(*param_ranges['neg_bagging_fraction']),
        'feature_fraction': random.uniform(*param_ranges['feature_fraction']),
        'lambda_l1': random.uniform(*param_ranges['lambda_l1']),
        'lambda_l2': random.uniform(*param_ranges['lambda_l2'])
    }

    # Add fixed parameters
    fixed_params = {
        'objective': 'binary',
        'random_state': 42,
        'bagging_freq': 1,
        'verbosity': -1,
        # 'is_unbalance': True
    }
    params.update(fixed_params)

    return params


def print_lgb_params(params: dict[str, Any]) -> None:
    """Print the generated LightGBM parameters in a readable format.

    Args:
        params (dict[str, Any]): 
            Dictionary of LightGBM parameters.
    """
    clr_print("\nGenerated LightGBM Parameters:")
    for key, value in params.items():
        print(f"\t{repr(key):<22} --> {value}")
        
lgb_params = generate_lgb_params(random_seed=4)
print_lgb_params(lgb_params)

# --- Cell Separator ---

# We will try to use this loss directly next time...
def focal_loss(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    gamma: float = 2.0, 
    alpha: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Focal Loss for LightGBM.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted probabilities.
        gamma (float): Focusing parameter.
        alpha (float): Balancing parameter.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            Gradient and Hessian.
    """
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    
    focal_weight = alpha_t * np.power(1 - pt, gamma)
    
    gradient = focal_weight * (y_pred - y_true)
    hessian = focal_weight * (1 - pt) * pt
    
    return gradient, hessian


def perform_lightgbm_cv(
    df: pd.DataFrame,
    train_cols: list[str],
    target_col: str,
    fold_col: str,
    lgb_params: dict[str, Any],
    n_folds: int = 5
) -> tuple[list[float], list[lgb.LGBMRegressor]]:
    """Perform cross-validation using LightGBM and compute scores.

    Args:
        df (pd.DataFrame): 
            The full training dataframe.
        train_cols (list[str]): 
            List of column names to use for training.
        target_col (str): 
            Name of the target column.
        fold_col (str): 
            Name of the column containing fold assignments.
        lgb_params (dict[str, Any]): 
            Parameters for LightGBM model.
        n_folds (int, optional): 
            Number of folds for cross-validation.

    Returns:
        tuple[list[float], list[lgb.LGBMRegressor]]: 
            A tuple containing:
                - List of scores for each fold
                - List of trained LightGBM models
    """
    # Initialize
    scores, models, _df = [], [], df.copy()    
    for fold in range(n_folds):
        # Split data into train and validation sets
        _train_df = _df[_df[fold_col] != fold].reset_index(drop=True)
        _val_df = _df[_df[fold_col] == fold].reset_index(drop=True)

        # Initialize and train the model
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(_train_df[train_cols], _train_df[target_col])

        # Make predictions
        preds = model.predict(_val_df[train_cols])

        # Prepare DataFrames for scoring
        true_df = _val_df[['isic_id', target_col]].copy()
        pred_df = pd.DataFrame({'isic_id': _val_df['isic_id'], 'prediction': preds})

        # Compute score with 'score' function
        _score = score(true_df, pred_df, "isic_id")
        clr_print(f"Fold: {fold} - Score: {_score:.5f}")

        scores.append(_score)
        models.append(model)

    print(f"\nMean Score: {sum(scores) / len(scores):.5f}")
    print(f"Std Dev of Score: {pd.Series(scores).std():.5f}")
    return scores, models


def perform_lightgbm_cv_with_partial_smote(
    df: pd.DataFrame,
    train_cols: list[str],
    target_col: str,
    fold_col: str,
    lgb_params: dict[str, Any],
    n_folds: int = 5,
    smote_ratio: float = 0.025,
    random_state: int = 42
) -> tuple[list[float], list[lgb.LGBMRegressor]]:
    """Perform cross-validation using LightGBM with partial SMOTE and class weights.

    Args:
        df (pd.DataFrame): The full training dataframe.
        train_cols (list[str]): List of column names to use for training.
        target_col (str): Name of the target column.
        fold_col (str): Name of the column containing fold assignments.
        lgb_params (dict[str, Any]): Parameters for LightGBM model.
        n_folds (int, optional): Number of folds for cross-validation.
        smote_ratio (float, optional): Desired ratio of minority to majority class after SMOTE.
        random_state (int, optional): Random state for reproducibility.

    Returns:
        tuple[list[float], list[lgb.LGBMRegressor]]: A tuple containing:
            - List of scores for each fold
            - List of trained LightGBM models
    """
    scores, models = [], []
    _df = df.copy()

    # Calculate class weights
    class_weights = {0: 1, 1: len(df[df[target_col] == 0]) / len(df[df[target_col] == 1])}
    lgb_params['class_weight'] = class_weights

    # Initialize SMOTE with the desired ratio
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    
    for fold in range(n_folds):
        train_df = _df[_df[fold_col] != fold].reset_index(drop=True)
        val_df = _df[_df[fold_col] == fold].reset_index(drop=True)

        X_train, y_train = train_df[train_cols], train_df[target_col]

        # Apply partial SMOTE
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        X_resampled_df = pd.DataFrame(X_resampled, columns=train_cols)

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_resampled_df, y_resampled)

        X_val = val_df[train_cols]
        preds = model.predict(X_val)

        true_df = val_df[['isic_id', target_col]].copy()
        pred_df = pd.DataFrame({'isic_id': val_df['isic_id'], 'prediction': preds})

        _score = score(true_df, pred_df, "isic_id")
        print(f"Fold: {fold} - Score: {_score:.5f}")

        scores.append(_score)
        models.append(model)

    print(f"\nMean Score: {np.mean(scores):.5f}")
    print(f"Std Dev of Score: {np.std(scores):.5f}")

    return scores, models


## ... Mean of ~0.15 on run w/ original parameters... ###
scores, models = perform_lightgbm_cv(
    df=train_df,
    train_cols=train_cols,
    target_col="target",
    fold_col="fold",
    lgb_params=lgb_params,
)

# ### ... Mean of  on run w/ same parameters as below... ###
# scores, models = perform_lightgbm_cv_with_partial_smote(
#     df=train_df,
#     train_cols=train_cols,
#     target_col="target",
#     fold_col="fold",
#     lgb_params=lgb_params,
# )

# --- Cell Separator ---

def plot_feature_importance(
    models: list[lgb.LGBMRegressor], 
    top_n: int = 30,
    template_theme: str = "plotly_white",
) -> None:
    """Create an interactive bar chart of feature importances with text labels.

    This function calculates the mean feature importance across multiple 
    LightGBM models and plots the top N most important features with text labels.

    Args:
        models (list[lgb.LGBMRegressor]): 
            List of trained LightGBM models.
        top_n (int, optional): 
            Number of top features to display. 
        template_theme (str, optional):
            The plotly theme to use for plotly chart. .

    Returns:
        None; 
            Displays the Plotly chart.
    """
    # Calculate mean feature importance
    importances = np.mean([model.feature_importances_ for model in models], axis=0)
    feature_names = models[0].feature_name_

    # Create DataFrame and sort by importance
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=True).tail(top_n)
    
    #     for i, _c in enumerate(df_imp["feature"].values):
    #         if _c in ["color_variance_ratio", "border_color_interaction", "size_color_contrast_ratio", 
    #                   "age_normalized_nevi_confidence", "color_asymmetry_index", "3d_volume_approximation", 
    #                   "color_range", "shape_color_consistency", "border_length_ratio", "age_size_symmetry_index"]:
    #             print(i, _c)
    
    # Create Plotly bar chart
    fig = go.Figure(go.Bar(
        y=df_imp["feature"],
        x=df_imp["importance"],
        orientation='h',
        marker=dict(
            color=df_imp["importance"],
            colorscale='Magma',
            colorbar=dict(title="Importance")
        ),
        text=df_imp["importance"].apply(lambda x: f"{x:.1f}"),  # Format importance values
        textposition='outside',  # Position text outside of bars
        textfont=dict(size=10),  # Adjust text size as needed
    ))

    # Update layout for better readability
    fig.update_layout(
        title={
            'text': f'<b>Top {top_n} Feature Importances</b>',
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title="<b>Importance</b>",
        yaxis_title="<b>Features</b>",
        height=1500,
        width=1200,
        yaxis={'categoryorder':'total ascending'},
        template=template_theme,
        margin=dict(l=200, r=200),  # Increase left and right margins for text labels
        xaxis=dict(range=[0, df_imp["importance"].max() * 1.1])  # Extend x-axis range for text labels
    )

    # Show the plot
    fig.show()

    
plot_feature_importance(models, top_n=20)

# --- Cell Separator ---

def predict_with_ensemble(
    models: list[lgb.LGBMRegressor],
    df: pd.DataFrame,
    train_cols: list[str],
    aggregation_method: str = 'mean'
) -> pd.DataFrame:
    """
    Make predictions on data using an ensemble of LightGBM models.

    Args:
        models (list[lgb.LGBMRegressor]): 
            List of trained LightGBM models.
        df (pd.DataFrame): 
            Dataframe to predict on.
        train_cols (list[str]): 
            List of column names used for training.
        aggregation_method (str, optional): 
            Method to aggregate predictions ('mean' or 'median'). 
    
    Raises:
        ValueError:
            If the aggregation method provided is invalid.
    
    Returns:
        pd.DataFrame: 
            DataFrame with 'isic_id' and aggregated 'target' columns.
    """
    _df = df.copy()
    
    # Make predictions with each model
    all_predictions = [model.predict(_df[train_cols]) for model in models]
    
    # Aggregate predictions
    if aggregation_method == 'mean':
        final_predictions = np.mean(all_predictions, axis=0)
    elif aggregation_method == 'median':
        final_predictions = np.median(all_predictions, axis=0)
    else:
        raise ValueError("Invalid aggregation method. Choose 'mean' or 'median'.")
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'isic_id': _df['isic_id'],
        'target': final_predictions
    })
    
    return result_df

test_predictions = predict_with_ensemble(
    models=models,  
    df=test_df,
    train_cols=train_cols,
    aggregation_method='mean'
)
test_predictions.to_csv('submission.csv', index=False)
display(test_predictions)

# --- Cell Separator ---

