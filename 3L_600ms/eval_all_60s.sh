# DEVELOPMENT SET -  A


# 4L_60s

python src/predict_with_csv_old.py \
        --pred_dir data/predictions/dev/a/4L_60s/ \
        --checkpoint_dir /data1/mlaquatra/compare/mosquitos/ComParE-2022/mosquito_subchallenge/checkpoints_4/best_checkpoint/ \
        --test_folder /data1/mlaquatra/compare/mosquitos/data/dev/a/ \
        --max_seconds 60 \
        --num_labels 4 > outputs/dev_a_4L_60s.txt

python src/eval.py \
        --predictions_dir data/predictions/dev/a/4L_60s/ \
        --ground_truth_dir data/labels/dev/a/ >> outputs/dev_a_4L_60s.txt

python src/eval_DER.py --ground_truth_csv_path data/labels/dev/a/gt.csv \
        --prediction_csv_path data/predictions/dev/a/4L_60s/baseline_.csv >> outputs/dev_a_4L_60s.txt


# 3L_60s

python src/predict_with_csv_old.py \
        --pred_dir data/predictions/dev/a/3L_60s/ \
        --checkpoint_dir /data1/mlaquatra/compare/mosquitos/ComParE-2022/mosquito_subchallenge/checkpoints_3/best_checkpoint/ \
        --test_folder /data1/mlaquatra/compare/mosquitos/data/dev/a/ \
        --max_seconds 60 \
        --num_labels 3 > outputs/dev_a_3L_60s.txt

python src/eval.py \
        --predictions_dir data/predictions/dev/a/3L_60s/ \
        --ground_truth_dir data/labels/dev/a/ >> outputs/dev_a_3L_60s.txt

python src/eval_DER.py --ground_truth_csv_path data/labels/dev/a/gt.csv \
        --prediction_csv_path data/predictions/dev/a/3L_60s/baseline_.csv >> outputs/dev_a_3L_60s.txt



# 2L_60s

python src/predict_with_csv_old.py \
        --pred_dir data/predictions/dev/a/2L_60s/ \
        --checkpoint_dir /data1/mlaquatra/compare/mosquitos/ComParE-2022/mosquito_subchallenge/checkpoints_2/best_checkpoint/ \
        --test_folder /data1/mlaquatra/compare/mosquitos/data/dev/a/ \
        --max_seconds 60 \
        --num_labels 2 > outputs/dev_a_2L_60s.txt

python src/eval.py \
        --predictions_dir data/predictions/dev/a/2L_60s/ \
        --ground_truth_dir data/labels/dev/a/ >> outputs/dev_a_2L_60s.txt

python src/eval_DER.py --ground_truth_csv_path data/labels/dev/a/gt.csv \
        --prediction_csv_path data/predictions/dev/a/2L_60s/baseline_.csv >> outputs/dev_a_2L_60s.txt



# DEVELOPMENT SET -  B


# 4L_60s

python src/predict_with_csv_old.py \
        --pred_dir data/predictions/dev/b/4L_60s/ \
        --checkpoint_dir /data1/mlaquatra/compare/mosquitos/ComParE-2022/mosquito_subchallenge/checkpoints_4/best_checkpoint/ \
        --test_folder /data1/mlaquatra/compare/mosquitos/data/dev/b/ \
        --max_seconds 60 \
        --num_labels 4 > outputs/dev_b_4L_60s.txt

python src/eval.py \
        --predictions_dir data/predictions/dev/b/4L_60s/ \
        --ground_truth_dir data/labels/dev/b/ >> outputs/dev_b_4L_60s.txt

python src/eval_DER.py --ground_truth_csv_path data/labels/dev/b/gt.csv \
        --prediction_csv_path data/predictions/dev/b/4L_60s/baseline_.csv >> outputs/dev_b_4L_60s.txt


# 3L_60s

python src/predict_with_csv_old.py \
        --pred_dir data/predictions/dev/b/3L_60s/ \
        --checkpoint_dir /data1/mlaquatra/compare/mosquitos/ComParE-2022/mosquito_subchallenge/checkpoints_3/best_checkpoint/ \
        --test_folder /data1/mlaquatra/compare/mosquitos/data/dev/b/ \
        --max_seconds 60 \
        --num_labels 3 > outputs/dev_b_3L_60s.txt

python src/eval.py \
        --predictions_dir data/predictions/dev/b/3L_60s/ \
        --ground_truth_dir data/labels/dev/b/ >> outputs/dev_b_3L_60s.txt

python src/eval_DER.py --ground_truth_csv_path data/labels/dev/b/gt.csv \
        --prediction_csv_path data/predictions/dev/b/3L_60s/baseline_.csv >> outputs/dev_b_3L_60s.txt



# 2L_60s

python src/predict_with_csv_old.py \
        --pred_dir data/predictions/dev/b/2L_60s/ \
        --checkpoint_dir /data1/mlaquatra/compare/mosquitos/ComParE-2022/mosquito_subchallenge/checkpoints_2/best_checkpoint/ \
        --test_folder /data1/mlaquatra/compare/mosquitos/data/dev/b/ \
        --max_seconds 60 \
        --num_labels 2 > outputs/dev_b_2L_60s.txt

python src/eval.py \
        --predictions_dir data/predictions/dev/b/2L_60s/ \
        --ground_truth_dir data/labels/dev/b/ >> outputs/dev_b_2L_60s.txt

python src/eval_DER.py --ground_truth_csv_path data/labels/dev/b/gt.csv \
        --prediction_csv_path data/predictions/dev/b/2L_60s/baseline_.csv >> outputs/dev_b_2L_60s.txt