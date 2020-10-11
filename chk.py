import numpy as np
import os
#import xml.etree.ElementTree as et
from bs4 import BeautifulSoup
import cv2
import sklearn

class Batch_Generator:

    def __init__(self, box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'],):
        self.labels = []
        self.filenames = []
        self.image_ids = []
        self.box_output_format = box_output_format

    def parse_xml(self, annot_dirs, img_dirs, image_sets, classes):
        
        
        
        self.images_dirs = img_dirs
        self.annotations_dirs = annot_dirs
        self.image_set_filenames = image_sets
        self.classes = classes

        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        

        with open(image_sets) as f:
                image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
                self.image_ids += image_ids

            # Loop over all images in this dataset.
        for image_id in image_ids :
            filename = '{}'.format(image_id) + '.jpg'
            self.filenames.append(os.path.join(img_dirs, filename))

            if not annot_dirs is None:
                    # Parse the XML file for this image.
                with open(os.path.join(annot_dirs, image_id + '.xml')) as f:
                    soup = BeautifulSoup(f, 'xml')

                 # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
                    #filename = soup.filename.text

                boxes = [] # We'll store all boxes for this image here
                objects = soup.find_all('object') # Get a list of all objects in this image

                    # Parse the data for each object
                for obj in objects:
                    class_name = obj.find('name').text
                    class_id = self.classes.index(class_name)
                        # Check if this class is supposed to be included in the dataset
                    xmin = int(obj.bndbox.xmin.text)
                    ymin = int(obj.bndbox.ymin.text)
                    xmax = int(obj.bndbox.xmax.text)
                    ymax = int(obj.bndbox.ymax.text)
                    item_dict = {
                                 'image_name': filename,
                                 'image_id': image_id,
                                 'class_name': class_name,
                                 'class_id': class_id,
                                 'xmin': xmin,
                                 'ymin': ymin,
                                 'xmax': xmax,
                                 'ymax': ymax}
                    box = []
                    for item in self.box_output_format:
                        box.append(item_dict[item])
                    boxes.append(box)

                self.labels.append(boxes)

        
    def total_samples(self):
        return len(self.filenames)



    def generate(self, batch_size = 32, ssd_box_encoder = None,  train = False):

        current = 0
        
        while True :
    
            batch_x, batch_y = [], []

            if (current + batch_size >= len(self.filenames)):
                current = 0

            self.filenames, self.labels, self.image_ids = sklearn.utils.shuffle(self.filenames, self.labels, self.image_ids)

            batch_y = np.array(self.labels[current : current + batch_size])


            batch_filenames = self.filenames[current : current + batch_size]

            for filename in batch_filenames :
                batch_x.append(cv2.resize(cv2.imread(filename), (300, 300)))


            
            
            batch_x = np.array(batch_x)
            elements = np.reshape(batch_x, [-1])
            total = np.sum(elements)
            
            mean = total/len(elements)

            var = np.sum(np.square(elements - mean))

            sd = np.sqrt(var)

            batch_x = (batch_x - mean) / sd

            current += batch_size

            ret = []
            ret.append(batch_x)
            if train:
                batch_y_true = ssd_box_encoder.encode_y(batch_y)
                ret.append(batch_y_true)

            yield ret
            