# object_classification_pytorch

This is my learning about Object Classification with custom dataset.

# 1. Download dataset 
I download dataset from:  https://www.kaggle.com/c/dogs-vs-cats/data and store into /database folder.

My folder like below:

    ____Objetc_Classification                     # includes .py files

      |______database                         # includes images and csv
      
                |_______images                # includes image files
                
                |_______training              # includes csv file 
                
      |______model                            # includes trained model
      
 
# 2. Pre-processing dataset
I divide the dataset into training dataset (80%) and testing dataset (20%).

Run preprocessing.py to create .csv files. Note: All paths in this file is default, so if you
create folder like me, you do not care about it. In contrast, please change these paths.

After running this file, you will get 3 csv file in training folder:

    * data.csv            # store full dataset with id: name of the image, label is 1 - cat; 0 - dog
    
    * train_labels.csv    # store training dataset
    
    * test_labels.csv     # store testing dataset

# 3. Train
Run file train.py to train the model with custom dataset.
