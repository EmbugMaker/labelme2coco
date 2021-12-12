from torch.utils.data import random_split
import os
import shutil
from labelme2coco_util import get_files
from labelme2coco_util import to_coco
from labelme2coco_util import tqdm


class labelme2coco:
    def __init__(self, coco_dir, labelme_dir, classes, val_size_ratio=0.3):
        __ratio = val_size_ratio
        self.labelme_dir = labelme_dir
        self.classes = classes
        self.coco_dir = coco_dir
        self.sub_dir = ('annotations', 'train2017', 'val2017')
        self.coco_path = {self.sub_dir[0]: os.path.join(coco_dir, self.sub_dir[0]),
                          self.sub_dir[1]: os.path.join(coco_dir, self.sub_dir[1]),
                          self.sub_dir[2]: os.path.join(coco_dir, self.sub_dir[2])}
        for value in self.coco_path.values():
            if not os.path.exists(value):
                os.makedirs(value)

        self.suffixes = ['.json']
        self.json_files = get_files(self.labelme_dir, [], self.suffixes)

        self.all_size = len(self.json_files)
        self.val_size = int(self.all_size * __ratio)
        self.train_size = self.all_size - self.val_size
        print('The training set size:{}\nThe validation set size:{}\n'.format(self.train_size, self.val_size))

    def convert2coco(self):
        print('Creating COCO dataset' + "." * 6)
        __train_json_paths, __val_json_paths = random_split(dataset=self.json_files,
                                                            lengths=[self.train_size, self.val_size])
        for json_path in tqdm(__train_json_paths):
            path = os.path.splitext(json_path)[0]
            name = json_path.split("/")[-1].split(".")[0]
            img_path = path + '.jpg'
            new_img_path = self.coco_path['train2017'] + '/' + name + '.jpg'
            shutil.copy(img_path, new_img_path)

        for json_path in tqdm(__val_json_paths):
            path = os.path.splitext(json_path)[0]
            name = json_path.split("/")[-1].split(".")[0]
            img_path = path + '.jpg'
            new_img_path = self.coco_path['val2017'] + '/' + name + '.jpg'
            shutil.copy(img_path, new_img_path)

        train_json = to_coco(__train_json_paths,
                             self.coco_path['annotations'] + '/instances_train2017.json',
                             self.classes)
        train_json.save_json()
        val_json = to_coco(__val_json_paths,
                           self.coco_path['annotations'] + '/instances_val2017.json',
                           self.classes)
        val_json.save_json()
        print("\nDone!")


if __name__ == '__main__':
    # object_classes
    object_classes = {"background": 0, 'dreg': 1, 'damp': 2, 'dot': 3, 'whole': 4, 'scratch': 5}
    # Specify the path of the COCO file you want to generate
    coco_path = './coco/2017'
    # The path of the image marked by labelme and the generated json file
    labelme_json_path = './enhance_imgs_and_marks'
    convert = labelme2coco(coco_dir=coco_path, labelme_dir=labelme_json_path, classes=object_classes)
    convert.convert2coco()
