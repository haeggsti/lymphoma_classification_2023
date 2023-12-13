# Lymphoma Classification 2023
This is the repository for the modified code and trained model LARS-max/avg presented in the paper

**I. Häggström et al., _Deep learning for [18F]fluorodeoxyglucose-PET-CT classification in patients with lymphoma: a dual-centre retrospective analysis_, the Lancet Digital Health (2023)**

# Framework
The codes are run using PyTorch.

# Description of files
* `train.py`: script to train the model (here set to 2d ResNet34). Give appropriate arguments to the function (use `--help`).
* `predict.py`: script to uses the trained model for inference on the test set. Give appropriate arguments to the function (use `--help`).
* `dataset.py`: script containing all functions for loading and handling images. Called by `train.py` and `predict.py`. Note that the images are assumed to be stored in a binary float32 format. This can of course be altered to fit ones own data.
* `utils.py`: script containing the available models (here trimmed down to only ResNet34). Called by `train.py` and `predict.py`.
* `find_best_model.py`: script to create csv-file with ranking of the best performing models (used to choose top-10 ensemble). Give appropriate arguments to the function (use `--help`).
* `data.csv`: file with a dataframe containing all information about the image filenames and sizes, the binary target (0 or 1), as well as ensemble data splits. The file should contain (at least) the columnns below. E.g., there should be one column per each of N bootstrap data split, named _split0_..._splitN_ with the allocated split _train_, _val_, or _test_. Below, each scan has one coronal and one sagittal image (which of course should have same targets and same data split allocations). 
```
   df =
        scan_id   filename          target   matrix_size_1     matrix_size_2     split0     split1 ...   splitN
        0         image_0_cor.bin   0        250               250               train      train        val
        0         image_0_sag.bin   0        180               250               train      train        val
        1         image_1_cor.bin   1        234               210               train      train        train
        1         image_1_sag.bin   1        140               210               train      train        train
        2         image_2_cor.bin   1        245               199               train      val          train
        2         image_2_sag.bin   1        120               199               train      val          train
        3         image_3_cor.bin   0        189               249               test       test         test
        3         image_3_sag.bin   0        150               249               test       test         test
        ...       ...               ...      ...               ...               ...        ...          ...
        M         image_M_cor.bin   0        201               236               val        train        train
        M         image_M_sag.bin   0        120               236               val        train        train
```
* `convergence_split*_run*.csv`: file containing run of the epochs (epoch, loss, auc,...). Will be created upon running `train.py`.
* `checkpoint_split*_run*.pth`: file containing the checkpoint data (state_dict etc) of your model. Will be created upon running `train.py`, and overwritten with the latest epoch every iteration.
* `pred_split*_run*.csv`: file containing inference model predictions when running `predict.py`.

# How to run
1. Create dataframe according to template above `data.csv` (you can name the file however you want).
2. Update the path to your image files and dataframe filename at the top in `dataset.py`.
3. Start model training by running `>> python train.py <your arguments>`. This will save convergence files `convergence_split*_run*.csv` and model checkpoint `checkpoint_split*_run*.pth` in the output folder you specified. Run training on all your N data splits (to create ensemble).
4. After finishing training, run `>> python find_best_model.py <your arguments>` to evaluate ranking of your N trained models. This will create file `best_run.csv` in the same folder as the convergence + checkpoint files.
5. Run inference on test set by `>> python predict.py <your arguments>`. You decide which of your N models to run and the function uses the file `best_run.csv` from the previous step. This will save prediction file `pred_split*_run*.csv` in your specified output folder.
6. Analyze the predictions by grouping predictions on `scan_id` and aggregating by mean or max (LARS-avg or LARS-max), and averaging the final result for the top-10 models.

# Trained models
Downloadable checkpoints of the top-10 trained 2d LARS models are found here: [link](https://drive.google.com/drive/folders/1ObjxwcrKxtS3VubS8oOfCxGOPvHXmIaA?usp=sharing).

# Reference
Cite this work using:

# License for use
Creative Commons Non-Commercial license as seen in the file [LICENSE-CC-BY-NC-4.0.md](LICENSE-CC-BY-NC-4.0.md).\
Read more at https://creativecommons.org/licenses/by-nc/4.0/.
