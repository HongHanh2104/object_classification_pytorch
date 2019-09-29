
import os

data_path = '/media/honghanh/STUDY/DOCUMENT/MY_SWEET/MY_DISCOVERY/Database/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
id_list_path = os.path.join(data_path, "ImagesSets/Main/{}.txt".format("train"))
print(id_list_path)
#print("/media/honghanh/STUDY/DOCUMENT/MY_SWEET/MY_DISCOVERY/Database/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/train.txt")
ids = [id.strip() for id in open(id_list_path)]#"/media/honghanh/STUDY/DOCUMENT/MY_SWEET/MY_DISCOVERY/Database/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/train.txt")]
#print(ids) 