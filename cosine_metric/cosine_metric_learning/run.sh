CUDA_VISIBLE_DEVICES=6 python train_mars.py --batch_size=128  --dataset_dir=./DatasetForFRCNN --loss_mode=cosine-softmax --log_dir=./output/carla_train/ --run_id=cosine-softmax