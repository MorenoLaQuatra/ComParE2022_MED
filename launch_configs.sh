python train.py --train_csv_path df/train.csv \
    --dev_csv_path df/dev.csv \
    --root_train_folder /data1/mlaquatra/compare/mosquitos/data/train/ \
    --root_dev_folder /data1/mlaquatra/compare/mosquitos/data/dev/a/ \
    --external_noise_folder /data1/mlaquatra/compare/mosquitos/noise_samples/ \
    --model_name_or_path microsoft/wavlm-base \
    --num_labels 2 \
    --num_train_epochs 15 \
    --output_dir checkpoints_2/ \
    --log_dir logs/

python train.py --train_csv_path df/train.csv \
    --dev_csv_path df/dev.csv \
    --root_train_folder /data1/mlaquatra/compare/mosquitos/data/train/ \
    --root_dev_folder /data1/mlaquatra/compare/mosquitos/data/dev/a/ \
    --external_noise_folder /data1/mlaquatra/compare/mosquitos/noise_samples/ \
    --model_name_or_path microsoft/wavlm-base \
    --num_labels 4 \
    --output_dir checkpoints_4/ \
    --num_train_epochs 15 \
    --log_dir logs/