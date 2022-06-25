### Fine-tune the model:

```console
python train.py --train_csv_path df/train.csv \
                --dev_csv_path df/dev.csv \
                --root_dataset_folder path/to/root/folder \
                --external_noise_folder path/to/noise/folder \
                --model_name_or_path microsoft/wavlm-base-plus-sd \
                --output_dir checkpoints/ \
                --log_dir logs/ 
```