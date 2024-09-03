# YOLOV1 - Tensorflow 2.0

In this post we will construct the famous YOLOv1 neural network. It stands for “You Only Look Once” and at the time it was first introduced, it performed much better than its competitors in terms of detection speed and accuracy. YOLOv1 was introduced in 2015 by Redmon et al. you can see the original paper [here](https://arxiv.org/abs/1506.02640). Bear in mind that, in the following post, we have taken that you have a solid knowledge about neural networks and a basic knowledge about machine vision.

## Data acquisition
First, to train our network, we need a large amount of images that have been annotated. We will use **Fiftyone** library to this end. You can see the original documentation [here](https://docs.voxel51.com/). We will use google’s open images v7. The following code will be used to acquire 1000 samples of images with “Person” annotations for the testing dataset. You can change **dsSplit** to “train” to the the data necessary for the training dataset as well.

    import fiftyone as fo
    import fiftyone.zoo as foz
    # Download the necessary data and dataset images for testing. We have chosen "open-images-v7-object-detection-DS"
    # as the name.
    dsName = "open-images-v7-object-detection-DS-test"
    dsClasses = ["Person"]
    dsSplit = "test"
    dsLblTypes = ["detections", "classifications"]
    nSamples = 1000
    if not fo.dataset_exists(dsName):
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split = dsSplit,
            label_types = dsLblTypes,
            classes = dsClasses,
            max_samples = nSamples,
            seed = 2,
            shuffle = True,
            dataset_name = dsName,
        )
    else:
        datasetTest = fo.load_dataset(dsName)
        print("Dataset already loaded. ")

Now, the downloaded dataset should be moved to desired directories. It is a common practice to have separate directories for images and their respective labels. Furthermore, each directory (image or label) should contain two sub-directories named train and validation. Because we already know what the desired network is (YOLOv1) and what its parameters are, we don’t need validation dataset. Do know that Fiftyone saved the labels in a JSON file which we will process next.

    # Exporting the downloaded datasets to the desired locations.
    # Train data
    datasetTrain.export(
        data_path = "./data/images/train",
        labels_path = "./data/labels/train/labels.json",
        dataset_type = fo.types.FiftyOneImageDetectionDataset,
        classes = dsClasses,
        include_confidence = False
    )
    
    # Test data
    datasetTest.export(
        data_path = "./data/images/test",
        labels_path = "./data/labels/test/labels.json",
        dataset_type = fo.types.FiftyOneImageDetectionDataset,
        classes = dsClasses,
        include_confidence = False
    )
As the final step for data preparation, we deserialize the JSON files in the label folder (train and test sub-directories) to text files. This way, each image’s annotations are located in a text file with image ID as its file name, which is of more convenience. The annotations are saved in the following format: I) Each object’s annotations are noted in a separate ling (e.g. If an image contains two persons, its annotation text file contains two lines of text). II) Annotations are noted by the bounding boxes (which will be explained later). The following information about each object and its bounding box is denoted in the annotation file: **{label} {centerX} {centerY} {width} {height}**. In the reminder of this article we will explain the bounding boxes and how they work very deeply, **don’t worry if you find this last paragraph a little vague!**

    import json
    
    # Deserialize the json file and convert it to text files to make it compatible with yolov8
    # Training dataset
    file = open("./data/labels/train/labels.json")
    js = json.load(file)
    for item in js["labels"]:
        txt = ""
        with open(f"./data/labels/train/{item}.txt", 'w') as txtFile:
            for subItem in js["labels"][item]: 
                width = subItem["bounding_box"][2]
                height = subItem["bounding_box"][3]
                centerX = subItem["bounding_box"][0] + width/2
                centerY = subItem["bounding_box"][1] + height/2
                label = subItem["label"]
                txt += f"{label} {centerX} {centerY} {width} {height}\n"
            txtFile.write(txt)
            txtFile.close()
        
    
    file = open("./data/labels/test/labels.json")
    js = json.load(file)
    for item in js["labels"]:
        txt = ""
        with open(f"./data/labels/test/{item}.txt", 'w') as txtFile:
            for subItem in js["labels"][item]: 
                width = subItem["bounding_box"][2]
                height = subItem["bounding_box"][3]
                centerX = subItem["bounding_box"][0] + width/2
                centerY = subItem["bounding_box"][1] + height/2
                label = subItem["label"]
                txt += f"{label} {centerX} {centerY} {width} {height}\n"
            txtFile.write(txt)
            txtFile.close()
A sample image from the dataset that I downloaded to my own local machine is included below. You can see four **“Persons”**s in this image. The bounding box of each object is also illustrated. The illustration is implemented via matplotlib which will be explained in the following sections.

![a](https://github.com/amirabbasja/YOLOv1-Tensorflow/blob/main/img/Screenshot-from-2024-07-02-10-55-48.png)
