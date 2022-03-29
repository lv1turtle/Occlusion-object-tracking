import os

import argparse
import cv2
import numpy as np

import time



def make_dataset(frame_idx, image, objects, output_file, count):

    for object in objects:
        object = [int (i) for i in object[0:8]]
        bbox = object[2:6]
        object_class = object[7]

        '''
            image : frame의 image
            frame_idx: frame number
            bbox: bounding box의 정보 - Left, Top, Width, Height
            object_class: object의 class
            id : Each pedestrian trajectory is identified by a unique ID
            filename : pedestrian_id(class)/type of cam(C1)/tracklet(id#)/frame# for tracklet(frame#) , ex)0001/C1/T0001/F001
            file_size : 128*256
        '''
        # print("frame_idx", frame_idx)
        # print("bbox", bbox)
        # print("class", object_class)
        
        # if bbox[2] < 64 or bbox[3] < 128:

        f = 0
        if bbox[2] < 128 and bbox[3] < 256:

            #---- 확대 -----
            if bbox[0] < 0 :
                bbox[0] = 0
            
            if bbox[1] < 0 :
                bbox[1] = 0

            if bbox[0] > 1920 :
                continue
            
            if bbox[1] > 1080 :
                continue

            bottom = bbox[1] + bbox[3]
            right = bbox[0] + bbox[2]

            if bottom > 1080 : 
                bbox[1] = bbox[1] - (bottom - 1080)

            if right > 1920 : 
                bbox[0] = bbox[0] - (right - 1920)

            roi = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
            roi = cv2.resize(roi,(128,256),interpolation=cv2.INTER_AREA)
            
            path = output_file +'/{0:0>4}'.format(object[1]+count) + '/{0:0>4}C1T{1:0>4}{2:0>3}.jpg'.format(object[1]+count,object[1]+count,f)
            while os.path.exists(path):
                f = f + 1
                path = output_file +'/{0:0>4}'.format(object[1]+count) + '/{0:0>4}C1T{1:0>4}{2:0>3}.jpg'.format(object[1]+count,object[1]+count,f)

            cv2.imwrite(path,roi)
            # ----------------------------------------------------------------- end


            # bounding box의 중심 좌표
            y = bbox[1] + bbox[3] // 2
            x = bbox[0] + bbox[2] // 2

            # if bbox[2] < 64 :
            #     bbox[2] = 64
            # elif bbox[3] < 128 :
            #     bbox[3] = 128
            
            bbox[2] = 128
            bbox[3] = 256

            bbox[1] = y - bbox[3] // 2
            bbox[0] = x - bbox[2] // 2

        if bbox[0] < 0 :
            bbox[0] = 0
        
        if bbox[1] < 0 :
            bbox[1] = 0

        if bbox[0] > 1920 :
            continue
        
        if bbox[1] > 1080 :
            continue

        bottom = bbox[1] + bbox[3]
        right = bbox[0] + bbox[2]

        if bottom > 1080 : 
            bbox[1] = bbox[1] - (bottom - 1080)

        if right > 1920 : 
            bbox[0] = bbox[0] - (right - 1920)

        roi = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        roi = cv2.resize(roi,(128,256),interpolation=cv2.INTER_AREA)

        path = output_file +'/{0:0>4}'.format(object[1]+count) + '/{0:0>4}C1T{1:0>4}{2:0>3}.jpg'.format(object[1]+count,object[1]+count,f)
        while os.path.exists(path):
            f = f + 1
            path = output_file +'/{0:0>4}'.format(object[1]+count) + '/{0:0>4}C1T{1:0>4}{2:0>3}.jpg'.format(object[1]+count,object[1]+count,f)
        
        cv2.imwrite(path,roi)


def run(sequence_dir, output_file, display):
    """

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.

    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.

    display : bool
        If True, show visualization of intermediate tracking results.

    """
    print("Processing: ", sequence_dir)
    sequences = os.listdir(args.sequence_dir)
    count = 0
    print("sequences", sequences)
    # load image file names
    for seq in sequences:
        if 'data' not in seq:
            continue
        seq_dir = sequence_dir +"/" + seq
        image_dir = os.path.join(seq_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}    

        # load objects info from "gt.txt" [frame#, ID, L, T, W, H, CS,Class, Vi]
        groundtruth_file = os.path.join(seq_dir, "gt/gt.txt")
        groundtruth = None
        if os.path.exists(groundtruth_file):
            groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

        groundtruth_indices = groundtruth[:,0].astype(np.int)

        # 최소, 최대 frame 정보
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())

        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            #print("frame_idx: ", frame_idx)
            mask = groundtruth_indices == frame_idx
            objects = groundtruth[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            print(image_filenames[frame_idx])
            # image: 해당 frame의 image
            image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
            #print(image_filenames[frame_idx])
            #print(count)
            make_dataset(frame_idx, image, objects, output_file, int(count))

        count = count + max(groundtruth[:, 1])



def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def clear(path):
    folders = os.listdir(path)

    for seq in folders:

        folder = os.path.join(path, seq)

        files = os.listdir(folder)
        if len(files) == 0:
            os.rmdir(folder)



def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default="../../data")
    parser.add_argument(
        "--output_dir", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",default="./DatasetForFRCNN/bbox_train03")
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=False, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for i in range(1,800) :
      path = os.path.join(args.output_dir, '{0:0>4}'.format(i))
      if os.path.exists(path):
          continue
      os.mkdir(path)
    run(args.sequence_dir, args.output_dir, args.display)

    clear(args.output_dir)
    # run(args.sequence_dir)