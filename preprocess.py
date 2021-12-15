import cv2      # for capturing videos
import numpy as np      # efficient array operations
import pandas as pd
import gzip     # efficient compression of numpy arrays     
from tqdm import tqdm
import os

def SaveVideoFrames(video_path, frames_array, f=10, max_frames=10, resize=(224, 224)):
    """
    Extract frames from video file with certain frequency.

    Parameters
    ----------
    video_path : str
        Path of the video file to be extracted.
    frame_array_path : str
        Path to store the NumPy array of frames of the video file to.
    f : int
        Intervals at which the frames are to be extracted (default=10).
    max_frames : int
        The maximum no. of frames to be extracted.
    resize : tuple of pair of int
        The size to which frames should be resized
    """

    count = 0
    frames = []
    zero_frame = np.zeros(shape=(resize[0], resize[1], 3))
    try:
        vidObj = cv2.VideoCapture(video_path)
        while len(frames) < max_frames:
            success, frame = vidObj.read()
            if success == False:
                while len(frames) < max_frames:
                    frames.append(zero_frame)
                break
            count += 1
            if count%f > 0:
                continue
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
        vidObj.release()
        f = gzip.GzipFile(frames_array + ".npy.gz", "w")
        np.save(f, np.array(frames))
        f.close()
    except:
        print("Extraction of " + video_path + " failed.")

videos_path = "./UCF-101"
annotations_path = "./ucfTrainTestlist"

frames_path = "./preprocessed"
os.makedirs(frames_path, exist_ok=True)
print("\nFrames of video file will be extracted to: " + frames_path)

with open(os.path.join(annotations_path, "trainlist01.txt")) as f:
    lines = f.readlines()
    train_df = [l.strip('\n').split(maxsplit=1) for l in lines]
train_df = pd.DataFrame(train_df, columns=["path", "label"])

print("\nExtracting frames from video files in train set...\n")
for idx in tqdm(range(len(train_df))):
    video = os.path.join(videos_path, train_df.iloc[idx]["path"])
    frames_array = os.path.join(frames_path, video.split('/')[-1].split('.avi')[0])
    SaveVideoFrames(video, frames_array, f=3, max_frames=20, resize=(224, 224))
print("\nFrame extraction of train set complete!")

train_df['path'] = train_df['path'].apply(lambda x: os.path.join(frames_path, x.split('/')[-1].split('.avi')[0]+'.npy.gz'))
train_df_path = "./train.csv"
train_df.to_csv(train_df_path, index=False)
print("\nTraining DataFrame saved as: " + train_df_path)