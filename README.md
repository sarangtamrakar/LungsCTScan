# Lungs CTScan

This is a flask app that allows users to upload CT scan images in the form of .nii files and check the predicted covid 19 masks. The model used to predict the masks has been trained on the [COVID-19 CT scans dataset](https://www.kaggle.com/andrewmvd/covid19-ct-scans). A unet architecture based model is used to predict the masks and the masks are then visualized and shown to the user.


## Structure
```
    .
    ├── src                  <- folder for ML models & pipelines
    ├── static               <- folder for css, js files and the processed images
    │   ├── Records  
    │   │   ├── Masked       <- processed segmentation masks
    │   │   └── Original     <- original CT scans                         
    ├── templates            <- templates for the HTML 
    └── README.md
```



## Preparation

### 1. Fork / Clone this repository

```bash
$ git clone https://github.com/sarangtamrakar/LungsCTScan/
$ cd LungsCTScan
```

### 2. Install Requirements
```bash
$ pip install -r requirements.txt
``` 

### 3. Run main.py
```bash
$ python main.py
``` 