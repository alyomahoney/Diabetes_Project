# Diabetes_Project
This project completes part of the final assessment in HarvardX's Professional Certificate in Data Science. It explores data on patients from Sylhet Diabetic Hospital. The main focus is to construct a model to predict whether a patient is at risk of being diabetic.

## Graded Files

### `main_code.R`
This is the main R scrips which loads and visualises the data and constructs the models. this script requires that the file `diabetes_data_upload.csv` is in the working (project) directory. The data is included in this repo and relative file paths (in line with the repo) are used to load the data.

### `final_report.rmd`
This is the final report in .Rmd form. The files which are loaded are created in `main_code.R`. Because the script takes a while to run, all of the R objects and images required to run this report are included in the folder `rmd_files`. This folder must be present in the working (project) directory. Again, relative file paths in line with this repo are used to load the files. This report is also available through [this RPubs link](https://rpubs.com/alyomahoney/diabetes).

### `final_report.pdf`
This is the final report in PDF format.


## Extra Files

### `rmd_files`
This folder contains all necessary files to run `final_report.Rmd`.

### `diabetes_data_upload.csv`
This is the original data set downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.).

### `save_files.R`
This R script keeps track of what R objects are saved under what file names. Feel free to ignore.