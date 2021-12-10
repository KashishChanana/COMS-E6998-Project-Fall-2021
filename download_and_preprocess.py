import urllib.request
import zipfile

def download_and_extract(download_link, path_to_zip_file, directory_to_extract_to):
    """
    Download zip file from link and extract contents to required folder.

    Parameters
    ----------
    video_path : str
        Path of the .mp4 file to be extracted.
    frame_path : str
        Path to extract the frames of the .mp4 file to.
    f : int
        Intervals at which the frames are to be extracted (default=10).

    """
    print("\nStarting download...")
    urllib.request.urlretrieve(download_link, path_to_zip_file)
    print("Download complete!")
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    print("\nExtraction done to: " + directory_to_extract_to)


videos_link = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
videos_path = "./Charades_v1_480.zip"
videos_dir = "./"

annotations_link = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip"
annotations_path = "./Charades.zip"
annotations_dir = "./"

print("\nDownloading Charades_v1_480.zip. This contains all the videos.")
download_and_extract(videos_link, videos_path, videos_dir)

print("\nDownloading Charades.zip. This contains all the annotations.")
download_and_extract(annotations_link, annotations_path, annotations_dir)



####################################################################################################

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2
import gzip

# check if val is nan
def is_nan(val):
    return val != val


def FrameCapture(video_path, frame_path, id, f=10):
    """
    Extract frames from .mp4 file with certain frequency.

    Parameters
    ----------
    video_path : str
        Path of the .mp4 file to be extracted.
    frame_path : str
        Path to extract the frames of the .mp4 file to.
    f : int
        Intervals at which the frames are to be extracted (default=10).

    """
    frame_path = os.path.join(frame_path, id) + ".npy.gz"
    video_path = os.path.join(video_path, id) + ".mp4"

    vidObj = cv2.VideoCapture(video_path)

    count = 0
    image_arr = []

    while True:
        success, image = vidObj.read()
        if success==False:
            break
        count += 1
        if count%f>0:
            continue
        image_arr.append(cv2.resize(image, dsize=(256, 256)))

    f = gzip.GzipFile(frame_path, "w")
    np.save(f, np.array(image_arr))
    f.close()


def one_hot_df(df, actions_list):
    """
    One hot encode the actions in the dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        The pandas DataFrame which contains the actions to be one hot encoded.
    actions_list : list
        List of the actions to present in the dataset.
    
    Returns
    -------
    pandas DataFrame
        pandas DataFrame containing the one hot encoded action(s) of each video.

    """
    new_df = []
    classes_one_hot_vec = np.zeros(len(actions_list))

    for i in range(len(df)):
        actions = df.iloc[i]["actions"]
        id = df.iloc[i]["id"]
        if is_nan(actions):
            continue
        classes_actions = actions.split(";")
        temp_classes_one_hot_vec = classes_one_hot_vec.copy()
        for c in classes_actions:
            temp_classes_one_hot_vec[actions_list.index(c.split()[0])] = 1
        tup = [id]+list(temp_classes_one_hot_vec)
        new_df.append(tup)
    
    new_df = pd.DataFrame(new_df, columns=['id']+actions_list)
    
    return new_df


videos_path = "./Charades_v1_480"
annotations_path = "./Charades"

with open(os.path.join(annotations_path, "Charades_v1_classes.txt")) as f:
    lines = f.readlines()
    classes_df = [l.strip('\n').split(maxsplit=1) for l in lines]

classes_df = pd.DataFrame(classes_df)
classes_df.to_csv("./Charades_v1_classes.csv", index=False)
classes_list = list(classes_df[0])

df_train = pd.read_csv(os.path.join(annotations_path, "Charades_v1_train.csv"))
df_train = one_hot_df(df_train, classes_list)
new_train_df_path = "./Charades_v1_train_one_hot.csv"
df_train.to_csv(new_train_df_path, index=False)

df_test = pd.read_csv(os.path.join(annotations_path, "Charades_v1_test.csv"))
df_test = one_hot_df(df_test, classes_list)
new_test_df_path = "./Charades_v1_test_one_hot.csv"
df_test.to_csv(new_test_df_path, index=False)

print("\nOne Hot Encoded .csv of Charades_v1_train.csv saved in: " + new_train_df_path)
print("\nOne Hot Encoded .csv of Charades_v1_test.csv saved in: " + new_test_df_path)

frames_path = "./Frames_Charades_v1_480"
os.makedirs(frames_path, exist_ok=True)
print("\nFrames of .mp4 file will be extracted to: " + frames_path)

print("\nExtracting frames from .mp4 files in train set...")
for idx in tqdm(range(len(df_train))):
    id = df_train.iloc[idx]["id"]
    FrameCapture(videos_path, frames_path, id)
print("Frame extraction of train set complete!")

print("\nExtracting frames from .mp4 files in test set...")
for idx in tqdm(range(len(df_test))):
    id = df_test.iloc[idx]["id"]
    FrameCapture(videos_path, frames_path, id)
print("Frame extraction of test set complete!")