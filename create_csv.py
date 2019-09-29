import os
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    # returns the list of files with their full path
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            '''
            tmp = member[4][0].text
            if tmp == 'hand':
                if os.path.exists(xml_file):
                    os.remove(xml_file)
            '''
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(str(member[4][0].text)),
                     int(str(member[4][1].text)),
                     int(str(member[4][2].text)),
                     int(str(member[4][3].text))
                     )
            xml_list.append(value)
            
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns = column_name)
    return xml_df

def split_test_train_file(csv_file):
    full_labels = pd.read_csv(csv_file)
    gb = full_labels.groupby('filename')
    grouped_list = [gb.get_group(x) for x in gb.groups]
    n = len(grouped_list)
    train_index = np.random.choice(n, size = 3980, replace = False)
    test_index = np.setdiff1d(list(range(n)), train_index)
    #print(len(train_index), len(test_index))
    
    train = pd.concat([grouped_list[i] for i in train_index])
    test = pd.concat([grouped_list[i] for i in test_index])
    train.to_csv('database/training/train_labels.csv', index = None)
    test.to_csv('database/training/test_labels.csv', index = None)
    

def main():
    '''
    # Creare csv file
    xml_path = os.path.join(os.getcwd(), 'database/annotations')
    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv('database/training/labels.csv', index = None)
    print('Successfully converted xml to csv.')
    '''
    # Then split into train and test file
    split_test_train_file("database/training/labels.csv")

main()