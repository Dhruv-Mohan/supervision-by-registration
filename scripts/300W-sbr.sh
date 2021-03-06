CUDA_VISIBLE_DEVICES=0 python3 ./exps/basic_main.py \
	--train_lists ./cache_data/lists/G16k/300w.train.DET\
	                ./cache_data/lists/300VW/300VW.train.lst \
	--eval_ilists   ./cache_data/lists/300VW/300VW.test-1.lst \
	                ./cache_data/lists/300VW/300VW.test-2.lst \
	                ./cache_data/lists/300VW/300VW.test-3.lst \
	--num_pts 90 \
	--model_config ./configs/Detector.config \
	--opt_config ./configs/SGD.config \
	--save_path ./snapshots/300W-CPM-DET \
	--pre_crop_expand 0.2 --sigma 4 --batch_size 8 \
	--crop_perturb_max 30 --rotate_max 20 \
	--scale_prob 1.0 --scale_min 0.9 --scale_max 1.1 --scale_eval 1 \
	--heatmap_type gaussian
