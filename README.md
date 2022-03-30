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
------ img1/                 /* necessary */
------ img1Depth/   /* Can generate by using MegaDepth */
------ gt/gt.txt             /* necessary */
------ det/det.txt           /* necessary */
------ seqinfo.ini
...
-- test/
---- seq_02/
------ img1/                 /* necessary */
------ img1Depth/   /* Can generate by using MegaDepth */
------ det/det.txt           /* necessary */
------ seqinfo.ini
```

* If you want to use custom datasets, see below for references.

  * First, make the directory structure
  
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
    
  * Second, prepare imagesets.

    Form of Imageset
    ```bash
      image name : 6 digit frame number starting with 1
      e.g.) 000001.jpg ~ 000600.jpg
    ```
  * Third, make the gt.txt with imageset

    Part of the gt.txt in MOT17
    ```
      599,51,910,408,26,129,0,9,0.046154
      600,51,910,408,26,129,0,9,0.046154
      1,52,730,509,37,60,0,4,0.92105
      2,52,730,509,37,60,0,4,0.94737
    ```
    ![Untitled](https://user-images.githubusercontent.com/32154881/160889755-3b3655e7-da6f-4037-8975-6023794af0a4.png)
    ![Untitled](https://user-images.githubusercontent.com/32154881/160890340-2dbb26db-c797-4609-8109-939a7186b412.png)

#### Generating a dataset for cosine_metric_learning :
* You need a gt.txt file

* Download MOT17 datasets

* If you want custom datasets, 
