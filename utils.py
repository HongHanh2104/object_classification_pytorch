import numpy as np
from random import uniform
import cv2
from torch.utils.data.dataloader import default_collate

class Compose(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        for function in self.transform:
            data = function(data)
        return data

class Resize(object):
    def __init__(self, image_size):
        super().__init__() 
        self.image_size = image_size
    
    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        width_ratio = float(self.image_size)/ width
        height_ratio = float(self.image_size)/ height
        new_obj = []
        for lb in label:
            resized_xmin = lb[0] * width_ratio
            resized_ymin = lb[1] * height_ratio
            resized_xmax = lb[2] * width_ratio
            resized_ymax = lb[3] * height_ratio
            resize_width = resized_xmax - resized_xmin
            resize_height = resized_ymax - resized_ymin
            new_obj.append([resized_xmin, resized_ymin, resize_width, resize_height, lb[4]])
        return image, new_obj

class Crop(object):
    def __init__(self, max_crop=0.1):
        super().__init__()
        self.max_crop = max_crop

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]
        xmin = width
        ymin = height
        xmax = 0
        ymax = 0
        for lb in label:
            xmin = min(xmin, lb[0])
            ymin = min(ymin, lb[1])
            xmax = max(xmax, lb[2])
            ymax = max(ymax, lb[2])
        cropped_left = uniform(0, self.max_crop)
        cropped_right = uniform(0, self.max_crop)
        cropped_top = uniform(0, self.max_crop)
        cropped_bottom = uniform(0, self.max_crop)
        new_xmin = int(min(cropped_left * width, xmin))
        new_ymin = int(min(cropped_top * height, ymin))
        new_xmax = int(max(width - 1 - cropped_right * width, xmax))
        new_ymax = int(max(height - 1 - cropped_bottom * height, ymax))

        image = image[new_ymin:new_ymax, new_xmin:new_xmax, :]
        label = [[lb[0] - new_xmin, lb[1] - new_ymin, lb[2] - new_xmin, lb[3] - new_ymin, lb[4]] for lb in label]

        return image, label


class VerticalFlip(object):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, data):
        image, label = data
        if uniform(0, 1) >= self.prob:
            image = cv2.flip(image, 1)
            width = image.shape[1]
            label = [[width - lb[2], lb[1], width - lb[0], lb[3], lb[4]] for lb in label]
        return image, label

def custom_collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = list(items[1])
    return items



'''
def parse_annotation(ann_dir, img_dir, labels = []):
    all_imgs = []
    seen_labels = {}
    
    ann_files = os.listdir(ann_dir)
    for ann in tqdm(sorted(ann_files)):
        img = {'object': []}

        tree = et.parse(os.path.join(ann_dir, ann))
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return all_imgs, seen_labels 

def parse_annotation_csv(csv_file, image_path=""):
    # This is a generic parser that uses CSV files
    # File_path,xmin,ymin,xmax,ymax,class

    print("parsing {} csv file can took a while, wait please.".format(csv_file))
    all_imgs = []
    seen_labels = {}

    all_imgs_indices = {}
    count_indice = 0
    with open(csv_file, "r") as annotations:
        annotations = annotations.read().split("\n")
        for i, line in enumerate(tqdm(annotations)):
            if line == "":
                continue
            try:
                fname, width, height, obj_name, xmin, ymin, xmax, ymax = line.strip().split(",")
                fname = os.path.join(image_path, fname)
                
                image = cv2.imread(fname)
                height, width, channel = image.shape

                img = dict()
                img['object'] = []
                img['filename'] = fname
                img['width'] = width
                img['height'] = height

                if obj_name == "":  # if the object has no name, this means that this image is a background image
                    all_imgs_indices[fname] = count_indice
                    all_imgs.append(img)
                    count_indice += 1
                    continue

                obj = dict()
                obj['xmin'] = int(xmin)
                obj['xmax'] = int(xmax)
                obj['ymin'] = int(ymin)
                obj['ymax'] = int(ymax)
                obj['name'] = obj_name
                
                
                if len(labels) > 0 and obj_name not in labels:
                    continue
                else:
                    img['object'].append(obj)
                

                if fname not in all_imgs_indices:
                    all_imgs_indices[fname] = count_indice
                    all_imgs.append(img)
                    count_indice += 1
                else:
                    all_imgs[all_imgs_indices[fname]]['object'].append(obj)

                if obj_name not in seen_labels:
                    seen_labels[obj_name] = 1
                else:
                    seen_labels[obj_name] += 1

            except:
                print("Exception occured at line {} from {}".format(i+1, csv_file))
                raise
    return all_imgs, seen_labels
'''
