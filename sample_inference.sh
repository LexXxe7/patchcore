datapath=datasets/mvtec
loadpath=results/MVTecAD_Results
modelfolder=IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0
savefolder=evaluated_results'/'$modelfolder

datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))


python bin/infer_patchcore.py --gpu 0 --seed 0 --save_segmentation_images $savefolder patch_core_loader "${model_flags[@]}" --faiss_on_gpu dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath