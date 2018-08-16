CUDA_VISIBLE_DEVICES=0 python3 ./exps/basic_main.py \
	--train_lists ./cache_data/lists/G16k/300w.train.DET \
	--eval_ilists ./cache_data/lists/G16k/300w.test.common.DET \
	--num_pts 90 \
	--model_config ./configs/Detector.config \
	--opt_config ./configs/SGD.config \
	--save_path ./snapshots/300W-CPM-DET \
	--pre_crop_expand 0.3 --sigma 4 --batch_size 16 \
	--crop_perturb_max 30 --rotate_max 20 \
	--scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 --scale_eval 1 \
	--heatmap_type gaussian
