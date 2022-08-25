datapath=datasets/mvtec
datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))


#############
# Detection #
#############

# Baseline:
python bin/train_patchcore.py --gpu 0 --seed 0 --save_segmentation_images --save_patchcore_model --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath

# Ensemble:
python bin/train_patchcore.py --gpu 0 --seed 3 --save_segmentation_images --save_patchcore_model --log_group IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1_S3 --log_project MVTecAD_Results results patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath

# -----------

# Baseline:
python bin/train_patchcore.py --gpu 0 --seed 22 --save_segmentation_images --save_patchcore_model --log_group IM320_WR50_L2-3_P001_D1024-1024_PS-3_AN-1_S22 --log_project MVTecAD_Results results patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath

# Ensemble:
python bin/train_patchcore.py --gpu 0 --seed 40 --save_segmentation_images --save_patchcore_model --log_group IM320_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1_S40 --log_project MVTecAD_Results results patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath


################
# Segmentation #
################

# Baseline:
python bin/train_patchcore.py --gpu 0 --seed 42 --save_segmentation_images --save_patchcore_model --log_group IM320_WR50_L2-3_P001_D1024-1024_PS-5_AN-3_S42 --log_project MVTecAD_Results results patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 1024 --anomaly_scorer_num_nn 3 --patchsize 5 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath

# Ensemble:
python bin/train_patchcore.py --gpu 0 --seed 88 --save_segmentation_images --save_patchcore_model --log_group IM320_Ensemble_L2-3_P001_D1024-384_PS-5_AN-5_S88 --log_project MVTecAD_Results results patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu --pretrain_embed_dimension 1024 --target_embed_dimension 384 --anomaly_scorer_num_nn 5 --patchsize 5 sampler -p 0.01 approx_greedy_coreset dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath