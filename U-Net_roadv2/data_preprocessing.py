from utils.data_agu import *
from PIL import Image
import PIL
import imageio


class ImageFolder(data.Dataset):

    def __init__(self, trainlist, image_root, gt_root, resize_shape, do_preprocessing=True):
        self.ids = trainlist
        self.loader = default_loader
        self.image_root = image_root
        self.gt_root = gt_root
        self.resize_shape = resize_shape
        self.do_preprocessing = do_preprocessing

    def __getitem__(self, index):
        filename = self.ids[index]
        img, mask = self.loader(filename, self.image_root,
                                self.gt_root, self.resize_shape, self.do_preprocessing)

        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        mask = mask.unsqueeze(0)
        return img, mask

    def __len__(self):
        return len(self.ids)

def default_loader(filename, image_root, gt_root, resize_shape, do_preprocessing=True):
    
    img = cv2.imread(os.path.join(image_root, filename))
    mask = cv2.imread(os.path.join(gt_root, filename), cv2.IMREAD_GRAYSCALE)
    

    # The network need the size to be a multiple of 32, resize is introduced
    img = cv2.resize(img, resize_shape)
    mask = cv2.resize(mask, resize_shape)
    #n = len(os.listdir(image_root))
    #for i in range(n):
    if do_preprocessing:
        img= randomColor(img, 
                                    hue_shift_limit=(-20, 20),
                                    sat_shift_limit=(-10, 10),
                                    val_shift_limit=(-10, 10))
        
        

        img, mask = randomRotate(img, mask,
                                        shift_limit=(-0.1, 0.1),
                                        scale_limit=(-0.1, 0.1),
                                        aspect_limit=(-0.1, 0.1),
                                        rotate_limit=(-180, 180))
        

        img, mask = HFlip(img, mask)
        
        

        img, mask = VFlip(img, mask)
        

    
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1)/255.0 *3.2-1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1)/255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask
