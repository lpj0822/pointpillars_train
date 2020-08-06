import os


def getFileData(dataFilePath):
    with open(dataFilePath, 'r') as file:
        for line in file:
            if line.strip():
                yield line.strip()


def main():
    file_path = "/home/lpj/github/data/my_point_cloud/kitti/ImageSets/val.txt"
    with open('val.txt', 'w') as writer:
        for data in getFileData(file_path):
            write_data = data + ".bin\n"
            writer.write(write_data)


if __name__ == '__main__':
    main()

