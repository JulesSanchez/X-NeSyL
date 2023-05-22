# architectural_style_classification
# Usage
Provide path to the data inside config.py
It is expected to follow the branching present originally in the dataset, otherwise perform adequat change to config.py and dataloader.py

# Folder organisation
* Root
    * Data
        * Dataset-IGRB1092_14cls
            * original disposition of the dataset
            * csv for training are created here (provided in this project, feel free to move them at the right place)
        * pascalPartDataset
            * original disposition of the dataset
            * txt for training are created here (provided in this project, feel free to move them at the right place)
    * MonuMAI-AutomaticStyleClassification (my branch)
    * pytorch-retinanet (my branch)
    * architectural_style_classification (this project)

If the folder organisation if different, change the various import path + the config.py file.

# How to run

First run 'python build-csv.py'
Then 'train.py' default parameter should work

# Models

Models can be found at : https://cloud.minesparis.psl.eu/index.php/s/TBKECKgGZoMp1aQ
