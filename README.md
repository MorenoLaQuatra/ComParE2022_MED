This repository contains the code for the paper *How much attention should we pay to mosquitoes?* submitted for the ComParE 2022 Mosquito Sub-Challenge.

Additional information will be provided upon paper acceptance.

### Model training:

```console
python train.py --train_csv_path df/train.csv \
                --dev_csv_path df/dev.csv \
                --root_train_folder /data/train/ \
                --root_dev_folder data/dev/ab/ \
                --external_noise_folder data/noise_samples/ \
                --model_name_or_path microsoft/wavlm-base-plus-sd \
                --output_dir checkpoints/ \
                --log_dir logs/ 
```

### Checkpoint evaluation:

```console
python eval_checkpoint.py \
        --ground_truth_csv_path eval_data/labels/dev/a/gt.csv \
        --metadata_csv_path eval_data/labels/dev/a/meta.csv \
        --pred_dir _temp/ \
        --checkpoint_dir checkpoints_3/best_checkpoint/ \
        --feature_extractor_model_name microsoft/wavlm-base \
        --num_labels 3 \
        --pred_file_prefix nl3_ \
        --plot_filename nl3_roc.png \
        --dev_folder data/dev/a/
```