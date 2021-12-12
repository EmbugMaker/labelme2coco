# labelme2coco
# This is a small tool for converting files marked by labelme into COCO datasets.
# Config as following samples:
  1) coco_path = "your_own_coco_path"
  2) labelme_path = "the_path_of_the_image_marked_by_labelme_and_the_generated_json_file"
  3) object_classes = {"background": 0, "your_classe_1": 1, "your_classe_2": 2, ...}
# Required environment:
  1) shutil
  2) torch >= 1.0
  3) json
  4) tqdm
  5) numpy
  
