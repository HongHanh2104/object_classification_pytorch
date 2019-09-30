import os 
import pandas as pd
import numpy as np

image_path = "database/images"
training_path = "database/training/"

def preprocessing():
    file_list = []
    for files in os.listdir(image_path):
        filename, file_extension = os.path.splitext(files)
        #name = filename[4:]
        if "cat" in filename:
            label = 1
            value = (filename, label)
            file_list.append(value)
        elif "dog" in filename:
            label = 0
            value = (filename, label)
            file_list.append(value)
    return file_list

def create_full_csv(data):
    column_name = ['id', 'label']
    dataframe = pd.DataFrame(data, columns = column_name)
    dataframe.to_csv(training_path + "data.csv", index = None)
    print('Successfully converted to csv.')

def split_test_train_file(csv_file):
    df = pd.read_csv(csv_file)
    rows = len(df)
    gb = df.groupby('id')
    
    grouped_list = [gb.get_group(x) for x in gb.groups]
    n = len(grouped_list)
    train_index = np.random.choice(n, size = 20000, replace = False)
    test_index = np.setdiff1d(list(range(n)), train_index)
    print(len(train_index), len(test_index))
    
    train = pd.concat([grouped_list[i] for i in train_index])
    test = pd.concat([grouped_list[i] for i in test_index])
    train.to_csv('database/training/train_labels.csv', index = None)
    test.to_csv('database/training/test_labels.csv', index = None)
    

dataframe = preprocessing()
create_full_csv(dataframe)
split_test_train_file('database/training/data.csv')
