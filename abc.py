print("hello")

python visualizeDataset.py --images="data/dataset1/images_prepped_train/" --annotations="data/dataset1/annotations_prepped_train/" --n_classes=10 


THEANO_FLAGS=device=cuda,floatX=float32  python  train.py --save_weights_path=weights_unet/ex1 --train_images="data/dataset1/images_prepped_train/" --train_annotations="data/dataset1/annotations_prepped_train/" --val_images="data/dataset1/images_prepped_test/" --val_annotations="data/dataset1/annotations_prepped_test/" --n_classes=10 --input_height=288 --input_width=384 --model_name="vgg_unet" 


THEANO_FLAGS=device=cuda,floatX=float32  python  predict.py --save_weights_path=weights/ex1 --epoch_number=0 --test_images="data/dataset1/images_prepped_test/" --output_path="data/predictions_unet/" --n_classes=10 --input_height=288 --input_width=384 --model_name="vgg_segnet" 


THEANO_FLAGS=device=gpu,floatX=float32  python  predict.py \
 --save_weights_path=weights/ex1 \
 --epoch_number=0 \
 --test_images="data/dataset1/images_prepped_test/" \
 --output_path="data/predictions/" \
 --n_classes=10 \
 --model_name="vgg_segnet" 

 python tools/process.py --input_dir photo/original --b_dir photo/ground --operation combine --output_dir photo/combined

 python tools/split.py --dir photo/combined

 CUDA_VISIBLE_DEVICE=6 python pix2pix.py --mode train --output_dir facades_train --max_epochs 200 --input_dir ./photo/combined/train --which_direction AtoB

python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 200 \
  --input_dir facades/train \
  --which_direction BtoA

 python pix2pix.py \
  --mode test \
  --output_dir facades_test \
  --input_dir ./photo/combined/val \
  --checkpoint facades_train


python build_voc2012_data.py --image_folder="pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages" --semantic_segmentation_folder="pascal_voc_seg/VOCdevkit/VOC2012/Label" --list_folder='pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation' --image_format="jpg" --output_dir='./tfrecord'  

python train.py --logtostderr --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=256 --train_crop_size=256 --train_batch_size=2 --training_number_of_steps=100 --fine_tune_batch_norm=False --tf_initial_checkpoint='datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt' --train_logdir='./checkpoint' --dataset_dir='datasets/tfrecord'

python train.py --logtostderr --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=256 --train_crop_size=256 --train_batch_size=2 --training_number_of_steps=100 --fine_tune_batch_norm=False --tf_initial_checkpoint='datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt' --train_logdir='./checkpoint' --dataset_dir='datasets/pascal_voc_seg/tfrecord'

python train.py \
  --logtostderr \
  --train_split="train" \  选择用于训练的数据集 
  --model_variant="xception_65" \  
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=256 \  # 当因内存不够而报错时，可适当调小该参数
  --train_crop_size=256 \
  --train_batch_size=2 \  # 因内存不够，设置为2
  --training_number_of_steps=100 \  # 尝试训练100步
  --fine_tune_batch_norm=False \  # 当batch_size大于12时，设置为True
  --tf_initial_checkpoint='./weights/deeplabv3_pascal_train_aug/model.ckpt' \ # 加载权重，权重下载链接见文章开头
  --train_logdir='./checkpoint' \ # 保存训练的中间结果的路径
  --dataset_dir='./datasets/tfrecord'  # 第二步生成的tfrecord的路径


python eval.py --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=513 --eval_crop_size=513 --checkpoint_dir='./checkpoint' --eval_logdir='./validation_output' --dataset_dir='datasets/pascal_voc_seg/tfrecord' 

python eval.py --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=513 --eval_crop_size=513 --checkpoint_dir='./checkpoint' --eval_logdir='./validation_output' --dataset_dir='datasets/tfrecord' 


python train.py --logtostderr --training_number_of_steps=100 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=1 --dataset="pascal_voc_seg" --tf_initial_checkpoint='datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt' --train_logdir='./checkpoint' --dataset_dir='datasets/tfrecord'

python eval.py --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=513 --eval_crop_size=513 --dataset="pascal_voc_seg" --checkpoint_dir='./checkpoint' --eval_logdir='./validation_output' --dataset_dir='datasets/tfrecord'

python build_voc2012_data.py \
  --image_folder="./VOC2012/JPEGImages" \  # 保存images的路径
  --semantic_segmentation_folder="./VOC2012/SegmentationClass" \ #保存labels的路径，为单通道
  --list_folder='./VOC2012/ImageSets/Segmentation' \ # 保存train\val.txt文件的路径
  --image_format="jpg（image格式）" \
  --output_dir='./tfrecord'  # 生成tfrecord格式的数据所要保存的位
image = tf.image.decode_image(file_contents)#读取的格式，是bgr格式

python vis.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=513 --vis_crop_size=513 --checkpoint_dir='./checkpoint' --vis_logdir='./vis_output' --dataset_dir='datasets/tfrecord' 

python build_voc2012_data.py --image_folder='/Users/Krystal/Downloads/models/my_data/images' --semantic_segmentation_folder='/Users/Krystal/Downloads/models/my_data/label' --list_folder='/Users/Krystal/Downloads/models/my_data/index' --output_dir='/Users/Krystal/Downloads/models/my_data/tfrecord'

python train.py --logtostderr --training_number_of_steps=100 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=1 --dataset="my_data" --tf_initial_checkpoint='/Users/Krystal/Downloads/models/research/deeplab/datasets/my_data/init_models/deeplabv3_pascal_train_aug/model.ckpt' --train_logdir='/Users/Krystal/Downloads/models/research/deeplab/checkpoint' --dataset_dir='/Users/Krystal/Downloads/models/research/deeplab/datasets/my_data/tfrecord'



python vis.py --logtostderr --vis_split="val" --model_variant="xception_41" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=288 --vis_crop_size=384 --dataset="screw"  --checkpoint_dir='/Users/Krystal/Downloads/models/research/deeplab/datasets/screw_seg/exp/train_on_train_set/train' --vis_logdir='/Users/Krystal/Downloads/models/research/deeplab/datasets/screw_seg/exp/train_on_train_set/vis' --dataset_dir='/Users/Krystal/Downloads/models/research/deeplab/datasets/screw_seg/tfrecord'

python eval.py --logtostderr --eval_split="val" --model_variant="xception_41" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=288 --eval_crop_size=384 --dataset="screw" --checkpoint_dir='/Users/Krystal/Downloads/models/research/deeplab/datasets/screw_seg/exp/train_on_train_set/train' --eval_logdir='/Users/Krystal/Downloads/models/research/deeplab/datasets/screw_seg/exp/train_on_train_set/eval' --dataset_dir='/Users/Krystal/Downloads/models/research/deeplab/datasets/screw_seg/tfrecord'

   --eval_crop_size=288 --eval_crop_size=384 
 --vis_crop_size=288 --vis_crop_size=384

python train.py --logtostderr --training_number_of_steps=200 --split_name='train' --model_variant="xception_65"   --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=1 --dataset_name='my_data' 
--tf_initial_checkpoint="/Users/Krystal/Downloads/models/research/deeplab/datasets/my_data/init_models/deeplabv3_pascal_train_aug/model.ckpt" 
--train_logdir="/Users/Krystal/Downloads/models/research/deeplab/datasets/my_data/checkpoint" --dataset_dir="/Users/Krystal/Downloads/models/research/deeplab/datasets/my_data/tfrecord" --fine_tune_batch_norm=false

python eval.py --logtostderr --eval_split="val" --model_variant="xception_65"  --atrous_rates=6 --atrous_rates=12 --atro
us_rates=18 --output_stride=16 --decoder_output_stride=4  --dataset_name="my_data" --checkpoint_dir="J:/python3.6/deeplab3/my_data/checkpoint" --eval_logdir="J:/python3.6/deeplab3/my_data/eval"  --dataset_dir="J:/python3.6/deeplab3/my_data/tfrecord"

python vis.py --logtostderr --vis_split="val" --model_variant="xception_65"  --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4  --dataset_name="my_data" --checkpoint_dir="/Users/Krystal/Downloads/models/research/deeplab/datasets/my_data/checkpoint" --vis_logdir="/Users/Krystal/Downloads/models/research/deeplab/datasets/my_data/vis" --dataset_dir="/Users/Krystal/Downloads/models/research/deeplab/datasets/my_data/tfrecord"

python build_voc2012_data.py --image_folder='/Users/Krystal/Downloads/models/pascal_voc_seg/images' --semantic_segmentation_folder='/Users/Krystal/Downloads/models/pascal_voc_seg/label' --list_folder='/Users/Krystal/Downloads/models/pascal_voc_seg/index' --output_dir='/Users/Krystal/Downloads/models/pascal_voc_seg/tfrecord'

python train.py --logtostderr --training_number_of_steps=200 --split_name='train' --model_variant="xception_65"   --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=288 --train_crop_size=384 --train_batch_size=2 --dataset_name='my_data' --tf_initial_checkpoint="/Users/Krystal/Downloads/models/research/deeplab/weights/deeplabv3_pascal_train_aug/model.ckpt" --train_logdir="/Users/Krystal/Downloads/models/my_data/checkpoint" --dataset_dir="/Users/Krystal/Downloads/models/my_data/tfrecord" --fine_tune_batch_norm=false

python vis.py --logtostderr --vis_split="val" --model_variant="xception_65"  --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4  --dataset_name="my_data" --checkpoint_dir="/Users/Krystal/Downloads/models/my_data/checkpoint" --vis_logdir="/Users/Krystal/Downloads/models/my_data/vis" --dataset_dir="/Users/Krystal/Downloads/models/my_data/tfrecord"

CUDA_VISIBLE_DEVICES=6,7 python eval.py --logtostderr --eval_split="val" --model_variant="xception_65"  --atrous_rates=6 --atrous_rates=12 --atrous_rates=18  --output_stride=16 --decoder_output_stride=4  --dataset_name="pascal_voc_seg" --eval_crop_size=385 --eval_crop_size=385 --checkpoint_dir="/mnt/models/pascal_voc_seg/checkpoint" --eval_logdir="/mnt/models/pascal_voc_seg/eval" --dataset_dir="/mnt/models/pascal_voc_seg/tfrecord"

CUDA_VISIBLE_DEVICES=6，7 python train.py --logtostderr  --training_number_of_steps=10000 --split_name='train' --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=385 --train_crop_size=385 --train_batch_size=8 --dataset_name='pascal_voc_seg' --tf_initial_checkpoint="/mnt/models/research/deeplab/weights/deeplabv3_pascal_train_aug/model.ckpt" --train_logdir="/mnt/models/pascal_voc_seg/checkpoint" --dataset_dir="/mnt/models/pascal_voc_seg/tfrecord" --fine_tune_batch_norm=False
python train.py --logtostderr  --training_number_of_steps=10000 --split_name='train' --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=385 --train_crop_size=385 --train_batch_size=8 --dataset_name='pascal_voc_seg' --tf_initial_checkpoint="/mnt/models/research/deeplab/weights/deeplabv3_pascal_train_aug/model.ckpt" --train_logdir="/mnt/models/pascal_voc_seg/checkpoint" --dataset_dir="/mnt/models/pascal_voc_seg/tfrecord" --fine_tune_batch_norm=False

CUDA_VISIBLE_DEVICES=6，7 python vis.py --logtostderr --vis_split="val" --model_variant="xception_65"  --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --dataset_name="pascal_voc_seg" --checkpoint_dir="/mnt/models/pascal_voc_seg/checkpoint" --vis_logdir="/mnt/models/pascal_voc_seg/vis" --dataset_dir="/mnt/models/pascal_voc_seg/tfrecord"

python build_voc2012_data.py --image_folder='/mnt/models/pascal_voc_seg/images' --semantic_segmentation_folder='/mnt/models/pascal_voc_seg/label' --list_folder='/mnt/models/pascal_voc_seg/index' --output_dir='/mnt/models/pascal_voc_seg/tfrecord'

--num_clones=2

CUDA_VISIBLE_DEVICES=5 


NV_GPU=2 nvidia-docker run --name wangyg-xyy_VGG -it -v /data/wangyagang/xiyiyuan/xyy2:/mnt nvdl.githost.io:4678/dgx/theano:17.12-stage bash

cd /mnt/models/pascal_voc_seg/checkpoint/
cd /mnt/models/research/deeplab


models-master:

CUDA_VISIBLE_DEVICES=6,7 python build_voc2012_data.py --image_folder='/mnt/models-master/pascal_voc_seg/images/' --semantic_segmentation_folder='/mnt/models-master/pascal_voc_seg/label/' --list_folder='/mnt/models-master/pascal_voc_seg/index/' --output_dir='/mnt/models-master/pascal_voc_seg/tfrecord/'
/mnt/models-master/research/deeplab/datasets

CUDA_VISIBLE_DEVICES=6,7 python train.py --logtostderr  --training_number_of_steps=10000 --split_name='train' --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=385 --train_crop_size=385 --train_batch_size=8 --dataset_name='pascal_voc_seg' --tf_initial_checkpoint="/mnt/models-master/research/deeplab/weights/deeplabv3_pascal_train_aug/model.ckpt" --train_logdir="/mnt/models-master/pascal_voc_seg/checkpoint" --dataset_dir="/mnt/models-master/pascal_voc_seg/tfrecord" --fine_tune_batch_norm=False

CUDA_VISIBLE_DEVICES=6,7 python eval.py --logtostderr --eval_split="val" --model_variant="xception_65"  --atrous_rates=6 --atrous_rates=12 --atrous_rates=18  --output_stride=16 --decoder_output_stride=4  --dataset_name="pascal_voc_seg" --eval_crop_size=385 --eval_crop_size=385 --checkpoint_dir="/mnt/models-master/pascal_voc_seg/checkpoint" --eval_logdir="/mnt/models-master/pascal_voc_seg/eval" --dataset_dir="/mnt/models-master/pascal_voc_seg/tfrecord"


cityscapes:

CUDA_VISIBLE_DEVICES=4,5,6,7 python build_voc2012_data.py --image_folder='/mnt/models/cityscapes/images' --semantic_segmentation_folder='/mnt/models/cityscapes/label' --list_folder='/mnt/models/cityscapes/index' --output_dir='/mnt/models/cityscapes/tfrecord'

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --logtostderr  --training_number_of_steps=10000 --split_name='train' --model_variant="xception_71" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --train_crop_size=385 --train_crop_size=385 --train_batch_size=4 --dataset_name='cityscapes' --dense_prediction_cell_json="/mnt/models/research/deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" --tf_initial_checkpoint="/mnt/models/research/deeplab/weights/trainval_fine/model.ckpt" --train_logdir="/mnt/models/cityscapes/checkpoint" --dataset_dir="/mnt/models/cityscapes/tfrecord" --fine_tune_batch_norm=False

CUDA_VISIBLE_DEVICES=4,5,6,7 python eval.py --logtostderr --eval_split="train" --model_variant="xception_71"  --atrous_rates=12 --atrous_rates=24 --atrous_rates=36  --output_stride=8 --decoder_output_stride=4  --dataset_name="cityscapes" --eval_crop_size=385 --eval_crop_size=385 --dense_prediction_cell_json="/mnt/models/research/deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" --checkpoint_dir="/mnt/models/cityscapes/checkpoint" --eval_logdir="/mnt/models/cityscapes/eval" --dataset_dir="/mnt/models/cityscapes/tfrecord"

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --logtostderr  --training_number_of_steps=10000 --split_name='train' --model_variant="xception_71" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --train_crop_size=385 --train_crop_size=385 --train_batch_size=4 --dataset_name='cityscapes' --dense_prediction_cell_json="/mnt/models/research/deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" --tf_initial_checkpoint="/mnt/models/research/deeplab/weights/trainval_fine/model.ckpt" --train_logdir="/mnt/models/cityscapes/checkpoint" --dataset_dir="/mnt/models/cityscapes/tfrecord" --fine_tune_batch_norm=False

mac:

python build_voc2012_data.py --image_folder='/Users/Krystal/Downloads/models/cityscapes/images' --semantic_segmentation_folder='/Users/Krystal/Downloads/models/cityscapes/label' --list_folder='/Users/Krystal/Downloads/models/cityscapes/index' --output_dir='/Users/Krystal/Downloads/models/cityscapes/tfrecord'

python train.py --logtostderr  --training_number_of_steps=10000 --split_name='train' --model_variant="xception_71" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --train_crop_size=385 --train_crop_size=385 --train_batch_size=4 --dataset_name='cityscapes' --dense_prediction_cell_json="/Users/Krystal/Downloads/models/research/deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" --tf_initial_checkpoint="/Users/Krystal/Downloads/models/research/deeplab/weights/trainval_fine/model.ckpt" --train_logdir="/Users/Krystal/Downloads/models/cityscapes/checkpoint" --dataset_dir="/Users/Krystal/Downloads/models/cityscapes/tfrecord" --fine_tune_batch_norm=False

python eval.py --logtostderr --eval_split="train" --model_variant="xception_71"  --atrous_rates=12 --atrous_rates=24 --atrous_rates=36  --output_stride=8 --decoder_output_stride=4  --dataset_name="cityscapes" --eval_crop_size=385 --eval_crop_size=385 --dense_prediction_cell_json="/Users/Krystal/Downloads/models/research/deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" --checkpoint_dir="/Users/Krystal/Downloads/models/cityscapes/checkpoint" --eval_logdir="/Users/Krystal/Downloads/models/cityscapes/eval" --dataset_dir="/Users/Krystal/Downloads/models/cityscapes/tfrecord"

python vis.py --logtostderr --vis_split="val" --model_variant="xception_71"  --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --dataset_name="cityscapes" --checkpoint_dir="/Users/Krystal/Downloads/models/cityscapes/checkpoint" --vis_logdir="/Users/Krystal/Downloads/models/cityscapes/vis" --dataset_dir="/Users/Krystal/Downloads/models/cityscapes/tfrecord"






