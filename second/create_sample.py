import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import glob
import codecs
import json
import random
import numpy as np
import cv2


class CreateClassifySample():

    def __init__(self):
        pass

    def process_sample(self, input_dir, output_dir, flag, probability=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if "train_val" == flag.strip():
            self.process_class_train_val(input_dir, output_dir, probability)
        elif "train" == flag.strip():
            self.process_train(input_dir, output_dir, flag)
        elif "val" == flag.strip():
            self.process_train(input_dir, output_dir, flag)

    def process_train_val(self, input_dir, output_dir, probability):
        save_train_path = os.path.join(output_dir, "train.txt")
        save_val_path = os.path.join(output_dir, "val.txt")
        save_train_file = open(save_train_path, "w")
        save_val_file = open(save_val_path, "w")

        image_list = list(self.getDirFiles(input_dir, "*.bin"))
        annotation_dir = os.path.join(input_dir, "../Annotations")
        random.shuffle(image_list)
        for image_index, image_path in enumerate(image_list):
            path, fileNameAndPost = os.path.split(image_path)
            fileName, post = os.path.splitext(fileNameAndPost)
            # numpy_points = np.fromfile(str(image_path),
            #                            dtype=np.float32, count=-1).reshape([-1, 4])
            annotation_file = fileName + ".json"
            annotation_path = os.path.join(annotation_dir, annotation_file)
            if not os.path.exists(annotation_path):
                print(image_path)
                continue
            if image_index % probability == 0:
                write_content = "%s\n" % fileNameAndPost
                save_val_file.write(write_content)
            else:
                write_content = "%s\n" % fileNameAndPost
                save_train_file.write(write_content)
        save_train_file.close()
        save_val_file.close()

    def process_class_train_val(self, input_dir, output_dir, probability):
        car_save_train_path = os.path.join(output_dir, "Car_train.txt")
        cyclist_save_train_path = os.path.join(output_dir, "Cyclist_train.txt")
        pedestrian_save_train_path = os.path.join(output_dir, "Pedestrian_train.txt")
        van_save_train_path = os.path.join(output_dir, "Van_train.txt")
        save_val_path = os.path.join(output_dir, "val.txt")

        car_save_train_file = open(car_save_train_path, "w")
        cyclist_save_train_file = open(cyclist_save_train_path, "w")
        pedestrian_save_train_file = open(pedestrian_save_train_path, "w")
        van_save_train_file = open(van_save_train_path, "w")
        save_val_file = open(save_val_path, "w")

        image_list = list(self.getDirFiles(input_dir, "*.bin"))
        annotation_dir = os.path.join(input_dir, "../Annotations")
        random.shuffle(image_list)
        car_class_index = 1
        cyclist_class_index = 1
        pedestrian_class_index = 1
        van_class_index = 1
        for image_index, image_path in enumerate(image_list):
            path, fileNameAndPost = os.path.split(image_path)
            fileName, post = os.path.splitext(fileNameAndPost)
            # numpy_points = np.fromfile(str(image_path),
            #                            dtype=np.float32, count=-1).reshape([-1, 4])
            annotation_file = fileName + ".json"
            annotation_path = os.path.join(annotation_dir, annotation_file)
            if not os.path.exists(annotation_path):
                print(image_path)
                continue
            _, gt_names = self.read_annotations_data(annotation_path)
            if (image_index + 1) % probability == 0:
                write_content = "%s\n" % fileNameAndPost
                save_val_file.write(write_content)
            else:
                if "Car" in gt_names:
                    car_class_index += 1
                    write_content = "%s\n" % fileNameAndPost
                    car_save_train_file.write(write_content)
                if "Cyclist" in gt_names:
                    cyclist_class_index += 1
                    write_content = "%s\n" % fileNameAndPost
                    cyclist_save_train_file.write(write_content)
                if "Pedestrian" in gt_names:
                    pedestrian_class_index += 1
                    write_content = "%s\n" % fileNameAndPost
                    pedestrian_save_train_file.write(write_content)
                if "Van" in gt_names:
                    van_class_index += 1
                    write_content = "%s\n" % fileNameAndPost
                    van_save_train_file.write(write_content)
        car_save_train_file.close()
        cyclist_save_train_file.close()
        pedestrian_save_train_file.close()
        van_save_train_file.close()
        save_val_file.close()

    def process_train(self, input_dir, output_dir, flag):
        data_class = self.get_data_class(input_dir)

        save_train_path = os.path.join(output_dir, "%s.txt" % flag)
        save_train_file = open(save_train_path, "w")

        for class_index, class_name in enumerate(data_class):
            data_class_dir = os.path.join(input_dir, class_name)
            image_list = list(self.getDirFiles(data_class_dir,
                                                           "*.*"))
            random.shuffle(image_list)
            for image_index, image_path in enumerate(image_list):
                print(image_path)
                self.write_data(image_path, class_name, class_index, save_train_file)

        save_train_file.close()
        self.write_data_class(data_class, output_dir)

    def write_data(self, image_path, class_name, class_index, save_file):
        path, fileNameAndPost = os.path.split(image_path)
        fileName, post = os.path.splitext(fileNameAndPost)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        if image is not None:
            write_content = "%s/%s %d\n" % (class_name, fileNameAndPost,
                                            class_index)
            save_file.write(write_content)

    def get_data_class(self, data_dir):
        result = []
        dir_names = os.listdir(data_dir)
        for name in dir_names:
            if not name.startswith("."):
                file_path = os.path.join(data_dir, name)
                if os.path.isdir(file_path):
                    result.append(name)
        return sorted(result)

    def write_data_class(self, data_class, output_dir):
        class_define = {}
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for index, className in enumerate(data_class):
            class_define[index] = className
        save_class_path = os.path.join(output_dir, "class.json")
        with codecs.open(save_class_path, 'w', encoding='utf-8') as f:
            json.dump(class_define, f, sort_keys=True, indent=4, ensure_ascii=False)

    def getDirFiles(self, dataDir, filePost="*.*"):
        if os.path.isdir(dataDir):
            imagePathPattern = os.path.join(dataDir, filePost)
            for filePath in glob.iglob(imagePathPattern):
                yield filePath
            return
        else:
            return None

    def read_annotations_data(self, annotation_path):
        my_file = open(annotation_path, encoding='utf-8')
        result = json.load(my_file)
        object_list = result['objects']['rect3DObject']
        box_names = []
        box_locs = []
        for box_value in object_list:
            if box_value['class'].strip() != 'DontCare':
                # print("yaw", box_value['yaw'])
                yaw = -box_value['yaw']  # inverse clockwise
                box = [box_value['centerX'],
                       box_value['centerY'],
                       box_value['centerZ'],
                       box_value['width'],
                       box_value['length'],
                       box_value['height'],
                       yaw]
                if (box[0] >= -1.5) and (box[0] <= 1.5) and \
                        (box[1] >= -2.5) and (box[1] <= 2.5):
                    continue
                if (box[0] >= -41) and (box[0] <= 41) and \
                        (box[1] >= -81) and (box[1] <= 41):
                    box_names.append(box_value['class'].strip())
                    box_locs.append(box)
        return box_locs, box_names


def main():
    print("start...")
    test = CreateClassifySample()
    test.process_sample("/home/lpj/github/data/my_point_cloud/ali_dataset/pcds",
                        "/home/lpj/github/data/my_point_cloud/ali_dataset/ImageSets",
                        "train_val",
                        8)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()

