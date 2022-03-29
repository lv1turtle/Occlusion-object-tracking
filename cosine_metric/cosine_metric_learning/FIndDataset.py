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


def run(sequence_dir, output_file):
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

    # load image file names
    for seq in sequences:
        seq_dir = sequence_dir +"/" + seq
        images = os.listdir(seq_dir)
        image_path = seq_dir + "/" + images[0]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        path = output_file + "/"+images[0]
        # print(path)
        cv2.imwrite(path,image)

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default="DatasetForFRCNN/bbox_train")
    parser.add_argument(
        "--output_dir", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",default="./output/find")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.sequence_dir, args.output_dir)
