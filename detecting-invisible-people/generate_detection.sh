# python tools/generate_detections.py \
# 	--model=/home/adriv/detect-invisible/detecting-invisible-people/resources/networks/mars_carla1.pb \
# 	--mot_dir=/home/adriv/detect-invisible/data/ \
# 	--output_dir=/home/adriv/detect-invisible/detecting-invisible-people/resources/detections/MOT17_train/

CUDA_VISIBLES=3 python tools/generate_detections.py \
	--model=/home/adriv/detect-invisible/detecting-invisible-people/resources/networks/mars_carla2.pb \
	--mot_dir=/home/adriv/detect-invisible/data/ \
	--output_dir=/home/adriv/detect-invisible/detecting-invisible-people/resources/detections/carla_train/