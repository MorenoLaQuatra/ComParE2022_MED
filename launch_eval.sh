# --------------------
# num_labels = 3
# --------------------

# Dev A
python eval_checkpoint.py \
        --ground_truth_csv_path eval_data/labels/dev/a/gt.csv \
        --metadata_csv_path eval_data/labels/dev/a/meta.csv \
        --pred_dir _temp/ \
        --checkpoint_dir checkpoints_3/best_checkpoint/ \
        --feature_extractor_model_name microsoft/wavlm-base \
        --num_labels 3 \
        --max_window_duration 0.06 \
        --pred_file_prefix nl3_A_ \
        --plot_filename results_20ms/nl3_roc_A.png \
        --dev_folder ../../data/dev/a/ \
        --use_cuda > results_20ms/nl3_A_res.txt

# Dev B
python eval_checkpoint.py \
        --ground_truth_csv_path eval_data/labels/dev/b/gt.csv \
        --metadata_csv_path eval_data/labels/dev/b/meta.csv \
        --pred_dir _temp/ \
        --checkpoint_dir checkpoints_3/best_checkpoint/ \
        --feature_extractor_model_name microsoft/wavlm-base \
        --num_labels 3 \
        --max_window_duration 0.06 \
        --pred_file_prefix nl3_B_ \
        --plot_filename results_20ms/nl3_roc_B.png \
        --dev_folder ../../data/dev/b/ \
        --use_cuda > results_20ms/nl3_B_res.txt


# --------------------
# num_labels = 2
# --------------------

# Dev A
python eval_checkpoint.py \
        --ground_truth_csv_path eval_data/labels/dev/a/gt.csv \
        --metadata_csv_path eval_data/labels/dev/a/meta.csv \
        --pred_dir _temp/ \
        --checkpoint_dir checkpoints_2/best_checkpoint/ \
        --feature_extractor_model_name microsoft/wavlm-base \
        --num_labels 2 \
        --max_window_duration 0.06 \
        --pred_file_prefix nl2_A_ \
        --plot_filename results_20ms/nl2_roc_A.png \
        --dev_folder ../../data/dev/a/ \
        --use_cuda > results_20ms/nl2_A_res.txt

# Dev B
python eval_checkpoint.py \
        --ground_truth_csv_path eval_data/labels/dev/b/gt.csv \
        --metadata_csv_path eval_data/labels/dev/b/meta.csv \
        --pred_dir _temp/ \
        --checkpoint_dir checkpoints_2/best_checkpoint/ \
        --feature_extractor_model_name microsoft/wavlm-base \
        --num_labels 2 \
        --max_window_duration 0.06 \
        --pred_file_prefix nl2_B_ \
        --plot_filename results_20ms/nl2_roc_B.png \
        --dev_folder ../../data/dev/b/ \
        --use_cuda > results_20ms/nl2_B_res.txt


# --------------------
# num_labels = 4
# --------------------

# Dev A
python eval_checkpoint.py \
        --ground_truth_csv_path eval_data/labels/dev/a/gt.csv \
        --metadata_csv_path eval_data/labels/dev/a/meta.csv \
        --pred_dir _temp/ \
        --checkpoint_dir checkpoints_4/best_checkpoint/ \
        --feature_extractor_model_name microsoft/wavlm-base \
        --num_labels 4 \
        --max_window_duration 0.06 \
        --pred_file_prefix nl4_A_ \
        --plot_filename results_20ms/nl4_roc_A.png \
        --dev_folder ../../data/dev/a/ \
        --use_cuda  > results_20ms/nl4_A_res.txt

# Dev B
python eval_checkpoint.py \
        --ground_truth_csv_path eval_data/labels/dev/b/gt.csv \
        --metadata_csv_path eval_data/labels/dev/b/meta.csv \
        --pred_dir _temp/ \
        --checkpoint_dir checkpoints_4/best_checkpoint/ \
        --feature_extractor_model_name microsoft/wavlm-base \
        --num_labels 4 \
        --max_window_duration 0.06 \
        --pred_file_prefix nl4_B_ \
        --plot_filename results_20ms/nl4_roc_B.png \
        --dev_folder ../../data/dev/b/ \
        --use_cuda > results_20ms/nl4_B_res.txt
