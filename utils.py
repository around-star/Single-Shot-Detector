import numpy as np

def iou(boxes1, boxes2, coord = 'centroids'):
    if len(boxes1.shape) == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if len(boxes2.shape) == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if (coord == 'centroid'):
        boxes1 = convert_coordinates(boxes1, start_index = 0, conversion = 'centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index = 0, conversion = 'centroids2corners')

    intersection = np.maximum(0, np.minimum(boxes1[:,2], boxes2[:,2]) - np.maximum(boxes1[:,0], boxes2[:,0])) * np.maximum(0, np.minimum(boxes1[:,3], boxes2[:,3]) - np.maximum(boxes1[:,1], boxes2[:,1]))
    union = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1]) + (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1]) - intersection

    return intersection/union


def convert_coordinates(tensor, start_index, conversion):
    ind = start_index

    tensor1 = np.copy(tensor)

    if conversion == 'corners2centroids':
        tensor1[...,ind] = (tensor[...,ind] + tensor[...,ind+2])/2
        tensor1[...,ind+1] = (tensor[...,ind+1] + tensor[...,ind+3])/2
        tensor1[...,ind+2] = tensor[...,ind+2] - tensor[...,ind+0]
        tensor1[...,ind+3] = tensor[...,ind+3] - tensor[...,ind+1]

    elif conversion == 'centroids2corners':
        tensor1[...,ind] = tensor[...,ind] - tensor[...,ind+2] / 2.0 
        tensor1[...,ind+1] = tensor[...,ind+1] - tensor[...,ind+3] / 2.0
        tensor1[...,ind+2] = tensor[...,ind] + tensor[...,ind+2] / 2.0
        tensor1[...,ind+3] = tensor[...,ind+1] + tensor[...,ind+3] / 2.0

    return tensor1
    
def _greedy_nms(predictions, iou_threshold=0.45):
    
    boxes_left = np.copy(predictions)
    maxima = [] 
    while boxes_left.shape[0] > 0: 
        maximum_index = np.argmax(boxes_left[:,0]) 
        maximum_box = np.copy(boxes_left[maximum_index]) 
        maxima.append(maximum_box) 
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) 
        if boxes_left.shape[0] == 0: break 
        similarities = iou(boxes_left[:,1:], maximum_box[1:], coord='corner') 
        boxes_left = boxes_left[similarities <= iou_threshold] 
    return np.array(maxima)




class SSDBoxEncoder:
    
    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 scales = None,
                 min_scale=0.1,
                 max_scale=0.9,
                 aspect_ratios_per_layer=None,
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.3,
                 normalize_coords = False):
        
        predictor_sizes = np.array(predictor_sizes)
        if len(predictor_sizes.shape) == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

       
        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes 
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.aspect_ratios = aspect_ratios_per_layer
        

        self.scales = scales
       



        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.normalize_coords = normalize_coords

        # Compute the number of boxes per cell.

        self.n_boxes = []
        for aspect_ratios in aspect_ratios_per_layer:

            self.n_boxes.append(len(aspect_ratios))

        self.boxes_list = [] # This will contain the anchor boxes for each predicotr layer.
        self.steps_diag = []
        self.wh_list_diag = []
        self.offsets_diag = []
        self.centers_diag = []

        for i in range(len(self.predictor_sizes)):
            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                                                                                   aspect_ratios=self.aspect_ratios[i],
                                                                                   this_scale=self.scales[i])
            self.boxes_list.append(boxes)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)
        

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale):
        
        

        size = min(self.img_height, self.img_width)
        wh_list = []
        for ar in aspect_ratios:
 
            box_width = this_scale * size * np.sqrt(ar)
            box_height = this_scale * size / np.sqrt(ar)
            wh_list.append((box_width, box_height))

        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)


        
        step_height = self.img_height / feature_map_size[0]
        step_width = self.img_width / feature_map_size[1]
        
       
        offset_height = 0.5
        offset_width = 0.5

        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # necessary for np.tile()
        cy_grid = np.expand_dims(cy_grid, -1) # necessary for np.tile()

        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h



        if self.normalize_coords:
            boxes_tensor[:, :, :, 0] /= self.img_width
            boxes_tensor[:, :, :, 2] /= self.img_width
            boxes_tensor[:, :, :, 1] /= self.img_height
            boxes_tensor[:, :, :, 3] /= self.img_height

        
        return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)


    def generate_encode_template(self, batch_size):
        boxes_batch = []
        for boxes in self.boxes_list:
            
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

    
        y_encode_template = np.concatenate((classes_tensor, boxes_tensor), axis=2)

        return y_encode_template

    def encode_y(self, ground_truth_labels):

       
        y_encode_template = self.generate_encode_template(batch_size=len(ground_truth_labels))
        y_encoded = np.copy(y_encode_template) 


        class_vector = np.eye(self.n_classes) # An identity matrix that we'll use as one-hot class vectors

        for i in range(y_encode_template.shape[0]): 
            available_boxes = np.ones((y_encode_template.shape[1])) 
            negative_boxes = np.ones((y_encode_template.shape[1])) 
            for true_box in ground_truth_labels[i]: 
    
                if abs(true_box[3] - true_box[1] < 0.001) or abs(true_box[4] - true_box[2] < 0.001): continue 
                if self.normalize_coords:
                    true_box[1] /= self.img_width
                    true_box[3] /= self.img_width 
                    true_box[2] /= self.img_height
                    true_box[4] /= self.img_height 

                true_box = convert_coordinates(np.array(true_box, dtype = np.float32), start_index=1, conversion='corners2centroids')
                
                similarities = iou(y_encode_template[i,:,-4:], np.array(true_box[1:]))

                for indices in range(len(similarities)):
                    if similarities[indices] >= self.neg_iou_threshold : negative_boxes[indices] = 0
 
                similarities *= available_boxes
                available_and_thresh_met = np.copy(similarities)
                for indices in range(len(similarities)):
                    if available_and_thresh_met[indices] < self.pos_iou_threshold : available_and_thresh_met[indices] = 0 
 
                assign_indices = np.nonzero(available_and_thresh_met)[0] 
                if len(assign_indices) > 0: # If we have any matches
                    for indices in assign_indices:
                        y_encoded[i,indices,:] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis=0) 
                        available_boxes[indices] = 0 
                else: # If we don't have any matches
                    best_match_index = np.argmax(similarities) # Get the index of the best iou match out of all available boxes
                    
                    y_encoded[i,best_match_index,:] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis=0) # Write the ground truth box coordinates and class to the best match anchor box position
                    available_boxes[best_match_index] = 0 # Make the assigned anchor box unavailable for the next ground truth box
                    negative_boxes[best_match_index] = 0 # The assigned anchor box is no longer a negative box
            # Set the classes of all remaining available anchor boxes to class zero
            background_class_indices = np.nonzero(negative_boxes)[0]
            y_encoded[i,background_class_indices,0] = 1

            
        y_encoded[:,:,-4] = (y_encoded[:,:,-4] - y_encode_template[:,:,-4]) / y_encode_template[:,:,-2] # (cx(gt) - cx(anchor)) / w(anchor)
        y_encoded[:,:,-3] = (y_encoded[:,:,-3] - y_encode_template[:,:,-3]) / y_encode_template[:,:,-1]# (cy(gt) - cy(anchor)) / h(anchor)

        y_encoded[:,:,-2] = np.log(y_encoded[:,:,-2] / y_encode_template[:,:,-2]) # log (w(gt) / w(anchor))
        y_encoded[:,:,-1] = np.log(y_encoded[:,:,-1] / y_encode_template[:,:,-1]) # log (h(gt) / h(anchor))

        
        return y_encoded



    def decode_y(self, y_pred, confidence_thresh = 0.1, iou_threshold = 0.45, top_k = 200, normalize_coords = False, img_height = None, img_width = None):
        
        y_encode_template = self.generate_encode_template(batch_size = len(y_pred))

        y_pred_decoded = np.copy(y_pred)

        y_pred_decoded[:,:,-2] = np.exp(y_pred[:,:,-2]) * y_encode_template[:,:,-2] # width of gt
        y_pred_decoded[:,:,-1] = np.exp(y_pred[:,:,-1]) * y_encode_template[:,:,-1] # height of gt

        y_pred_decoded[:,:,-4] = (y_pred[:,:,-4] * y_encode_template[:,:,-2]) + y_encode_template[:,:,-4] # cx of gt
        y_pred_decoded[:,:,-3] = (y_pred[:,:,-3] * y_encode_template[:,:,-1]) + y_encode_template[:,:,-3] # cy of gt

        y_pred_decoded = convert_coordinates(y_pred_decoded, start_index = -4, conversion='centroids2corners')

        if normalize_coords :
            y_pred_decoded[:,:,-4] *= img_width
            y_pred_decoded[:,:,-2] *= img_width
            y_pred_decoded[:,:,-3] *= img_height
            y_pred_decoded[:,:,-1] *= img_height

        y_pred = []

        for batch_item in y_pred_decoded:
            pred = []
            
            for class_id in range(1, self.n_classes):
                single_class = batch_item[:,[class_id, -4, -3, -2, -1]]
            
                threshold_met = single_class[single_class[:,0] > confidence_thresh]
                
                if threshold_met.shape[0] > 0:
                    
                    maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold) 
                    maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                    maxima_output[:,0] = class_id # class ID to the first column
                    maxima_output[:,1:] = maxima # the maxima to the other columns
                    pred.append(maxima_output)

            
            if pred: # If there are any predictions left after confidence-thresholding
                
                pred = np.concatenate(pred, axis=0)
                if pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,
                    top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                    pred = pred[top_k_indices] # keep only those entries of `pred`
            y_pred.append(pred) 

        return y_pred
