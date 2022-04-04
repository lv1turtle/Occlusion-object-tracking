# Occlusion-object-tracking
We are developing it with reference to [[Detecting Invisible People](https://github.com/tarashakhurana/detecting-invisible-people)] & [[MegaDepth](https://github.com/zhengqili/MegaDepth)] & [[Deep Sort](https://github.com/nwojke/deep_sort)]

The code skeleton is based on "https://github.com/tarashakhurana/detecting-invisible-people"

#### Dependencies:
* create a conda environment (name : deepsort)
```bash
  conda env create -f detecting-invisible-people/environment.yml
```

## Preprocessing
* The code expects the directory structure of your dataset in the MOT17 data format

```
MOT17/
-- train/
---- seq_01/
------ img1/                           /* necessary */
------ img1Depth/           /* Can generate by using MegaDepth */
------ gt/gt.txt                       /* necessary */
------ det/det.txt                     /* necessary */
------ seqinfo.ini
...
-- test/
---- seq_02/  
------ img1/                          /* necessary */
------ img1Depth/           /* Can generate by using MegaDepth */
------ det/det.txt                    /* necessary */
------ seqinfo.ini
...
resources/
-- detections/
---- seq_01.npy     /* Can generate by using ./tools/generate_detection.py */
-- networks/
---- mars_###.pb        /* Can generate by using cosine_metric_learning */
```

* If you want to use custom datasets, see below for references.

  * #### First, make the directory structure
  
    seqinfo.ini
    ```
      [Sequence]
      name=MOT17-02-FRCNN   /* Name of dataset directory */
      imDir=img1            /* Name of Imageset directory, It's better to fix 'img1' */
      frameRate=30
      seqLength=600         /* Number of frame */
      imWidth=1920
      imHeight=1080
      imExt=.jpg
    ```
    
  * #### Second, prepare imagesets.

    Form of Imageset
    ```bash
      image name : 6 digit frame number starting with 1
      e.g.) 000001.jpg ~ 000600.jpg
    ```
  * #### Third, make the gt.txt & det.txt with imageset

    Part of the gt.txt in MOT17
    ```
      599,51,910,408,26,129,0,9,0.046154
      600,51,910,408,26,129,0,9,0.046154
      1,52,730,509,37,60,0,4,0.92105
      2,52,730,509,37,60,0,4,0.94737
    ```
    The gt.txt format (Each line must contain 9 values)
    ![Untitled](https://user-images.githubusercontent.com/32154881/160889755-3b3655e7-da6f-4037-8975-6023794af0a4.png)
    ![Untitled](https://user-images.githubusercontent.com/32154881/160890340-2dbb26db-c797-4609-8109-939a7186b412.png)
    
    Part of the det.txt in MOT17
    ```
      436,-1,696.2,429.5,72.8,285.6,0.996
      436,-1,528.8,466.7,24.2,71.6,0.306
      294,-1,752.6,445,65.1,198,1
      294,-1,1517.6,430.2,241.1,461.2,1
    ```
    
    The det.txt format (Each line must contain 7 values)
    ```
      frame id, default(-1), x,  y,  width, height, confidence score
    ```
  * #### Fourth, generate the img1.npy from resources/detections
    Using ./tools/generate_detection.py
    ```
      python tools/generate_detections.py \
        --model=resources/networks/mars-###.pb \
        --mot_dir=./MOT17/train \
        --output_dir=./resources/detections/MOT17_train
    ```
    
  * #### Fifth, generate the img1Depth from img1
  
    Using megadepth,
    Fix the lines 134 in MegaDepth/demo_images_new.py
    ```bash
        images = sorted(glob.glob( " path of img1/*.jpg " ))
    ```
    
    Generate image_depth sets
    ```bash
        python MegaDepth/demo_images_new.py
    ```
    
## Cosine Metric Learning    

pretrained model in [here](https://drive.google.com/drive/folders/13HtkxD6ggcrGJLWaUcqgXl2UO6-p4PK0)

#### Generating a dataset for cosine_metric_learning :
* prepare the gt.txt & image sets


