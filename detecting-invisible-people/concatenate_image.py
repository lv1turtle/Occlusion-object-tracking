import os

import cv2
import numpy as np

def gather_sequence_info(sequence_dir):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
    }
    return seq_info

if __name__ == "__main__":
    sequences = os.listdir("/home/adriv/detect-invisible/data/")
    for seq in sequences:
        if 'data' not in seq:
            continue
        sequence_dir =  "/home/adriv/detect-invisible/data/" + seq
        seq_info = gather_sequence_info(sequence_dir)
        frame_idx = seq_info["min_frame_idx"]
        last_idx = seq_info["max_frame_idx"]
        img = cv2.imread(
                os.path.join(
                    sequence_dir,
                    'img1',
                    '{:06d}.jpg'.format(frame_idx)),
                0
            )

        frame_idx += 1
        while frame_idx <= last_idx:
            current_frame = cv2.imread(
                os.path.join(
                    sequence_dir,
                    'img1',
                    '{:06d}.jpg'.format(frame_idx)),
                0
            )
            
            img = np.concatenate((img,current_frame))
            frame_idx += 1
            print(img.shape)

        img = img.reshape((-1, 1080, 1920))
        print(img.shape)
        output_file = "/home/adriv/detect-invisible/data/" + seq + "/img_set"
        np.save(output_file,arr=img)
