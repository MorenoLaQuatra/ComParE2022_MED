# Mosquito Event Detection Challange

This repository contains the code for the paper *How much attention should we pay to mosquitoes?* submitted for the [ComParE 2022](http://www.compare.openaudio.eu/2022-2/) Mosquito Sub-Challenge.

Mosquitoes are a major global health problem. They are responsible for the transmission of diseases and can have a large impact on local economies. Monitoring mosquitoes is therefore helpful in preventing the outbreak of mosquito-borne diseases. In this paper, we propose a novel data-driven approach that leverages Transformer-based models for the identification of mosquitoes in audio recordings. The task aims at detecting the time intervals corresponding to the acoustic mosquito events in an audio signal. We formulate the problem as a sequence tagging task and train a Transformer-based model using a real-world dataset collecting mosquito recordings. By leveraging the sequential nature of mosquito recordings, we formulate the training objective so that the input recordings do not require fine-grained annotations. We show that our approach is able to outperform baseline methods using standard evaluation metrics, albeit suffering from unexpectedly high false negatives detection rates. In view of the achieved results, we propose future directions for the design of more effective mosquito detection models. More details can be found in the [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3551594).

## Model training:

```shell
python train.py --train_csv_path df/train.csv \
                --dev_csv_path df/dev.csv \
                --root_train_folder /data/train/ \
                --root_dev_folder data/dev/ab/ \
                --external_noise_folder data/noise_samples/ \
                --model_name_or_path microsoft/wavlm-base-plus-sd \
                --output_dir checkpoints/ \
                --log_dir logs/ 
```

## Checkpoint evaluation:

```shell
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

## Citation

```
@inproceedings{10.1145/3503161.3551594,
  author = {La Quatra, Moreno and Vaiani, Lorenzo and Koudounas, Alkis and Cagliero, Luca and Garza, Paolo and Baralis, Elena},
  title = {How Much Attention Should We Pay to Mosquitoes?},
  year = {2022},
  isbn = {9781450392037},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3503161.3551594},
  doi = {10.1145/3503161.3551594},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
  pages = {7135–7139},
  numpages = {5},
  keywords = {mosquito detection, audio sequence modelling, audio event detection, transformer models},
  location = {Lisboa, Portugal},
  series = {MM '22}
}
```

