import numpy as np
import os
import xml.etree.ElementTree as et
import cv2
import sklearn

class Batch_Generator:

    def __init__(self):
        self.labels = []
        self.filenames = []
        self.image_ids = []

    def parse_xml(self, annot_dirs, img_dirs, image_sets, classes):
        
        
        
        with open(image_sets) as f:
            image_id = [line.strip() for line in f]
            self.image_ids += image_id

        for image_id in self.image_ids:
            filename = image_id + '.jpg'
            self.filenames.append(os.path.join(img_dirs, filename))

            
            tree = et.parse(os.path.join(annot_dirs, image_id + '.xml'))
            objects = tree.findall('object')

            boxes = []
            for obj in objects:
                box = []
                class_name = obj.findtext('name')

                if (not class_name in classes) : continue

                class_id = classes.index(class_name)
                xmin = float (obj.find('bndbox').findtext('xmin'))
                ymin = float (obj.find('bndbox').findtext('ymin'))
                xmax = float (obj.find('bndbox').findtext('xmax'))
                ymax = float (obj.find('bndbox').findtext('ymax'))

                items = {
                        'class_id' : class_id,
                        'xmin' : xmin,
                        'ymin' : ymin,
                        'xmax' : xmax,
                        'ymax' : ymax
                    }
                for item in items.keys():
                    box.append(items[item])

                boxes.append(box)

            self.labels.append(boxes)
        return self.labels

        
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
            
