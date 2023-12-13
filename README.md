# lymphoma_classification_2023
This is the repository for the modified code and trained model LARS-max/avg presented in the paper

**I. Häggström et al., _Deep learning for [18F]fluorodeoxyglucose-PET-CT classification in patients with lymphoma: a dual-centre retrospective analysis_, the Lancet Digital Health (2023)**

# Description of files
* `train.py`: script to train the model (here set to 2d ResNet34). Give appropriate arguments to the function (use `--help`).
* `predict.py`: script to uses the trained model for inference on the testset. Give appropriate arguments to the function (use `--help`).
* `dataset.py`: script containing all functions for loading and handling images. Called by `train.py` and `predict.py`. Note that the images are assumed to be stored in a binary float32 format. This can of course be altered to fit ones own data.
* `utils.py`: script containing the available models (here trimmed down to only ResNet34). Called by `train.py` and `predict.py`.
* `find_best_model.py`: script to create csv-file with the best performing models (used to choose top-10 ensemble). Give appropriate arguments to the function (use `--help`).
* `data.csv`: file with a dataframe containig all information about the image filenames and sizes, the binary target (0 or 1), as well as ensemble data splits. The file should contain (at least) the columnns below. E.g., there should be one column per each of N bootstrap data split, named _split0_..._splitN_ with the allocated split _train_, _val_, or _test_.
```
   df = filename      target   matrix_size_1     matrix_size_2     split0     split1 ...   splitN
        image_1.bin   0        250               250               train      train        val
        image_2.bin   1        234               210               train      train        train
        image_3.bin   1        245               199               train      val          train
        image_4.bin   0        189               249               test       test         test
        ...           ...      ...               ...               ...        ...          ...
        image_M.bin   0        201               236               val        train        train
```
* `convergence_split*_run*.csv`: file containting run of the epochs (epoch, loss, auc,...). Will be created upon running `train.py`.
* `checkpoint_split*_run*.pth`: file containting the checkpoint data (state_dict etc) of your model. Will be created upon running `train.py`, and overwritten with the latest epoch every iteration.
* `pred_split*_run*.csv`: file containting inference model predictions when running `predict.py`.

# How to run
1. Create dataframe according to template above `data.csv` (you can name the file however you want).
2. Update the path to your image files and dataframe filename at the top in `dataset.py`.
3. Start model training by runnning `>> python train.py <your arguments>`. This will save convergence files `convergence_split*_run*.csv` and model checkpoint `checkpoint_split*_run*.pth` in the output folder you specified. Run training on all your N data splits (to create ensemble).
4. After finishing training, run `>> python find_best_model.py <your arguments>` to evaluate which of your N trained models is best. This will create file `best_run.csv` in the same folder as the convergence + checkpoint files.
5. Run inference on test set by `>> python predict.py <your arguments>`. You decide which of your N models to run and the function uses the file `best_run.csv` from the previous step. This will save prediction file `pred_split*_run*.csv` in your specified output folder.
6. Analyze the predictions on e.g. the top-10 models.

# Reference
Cite this work using:

# License for use
Creative Commons Non-Commercial license as seen in the file [LICENSE-CC-BY-NC-4.0.md](LICENSE-CC-BY-NC-4.0.md).\
Read more at https://creativecommons.org/licenses/by-nc/4.0/.
