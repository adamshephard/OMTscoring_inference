from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from tiatoolbox.tools.stainaugment import StainAugmentor


class StainAugment(object):
    def __init__(self, p=0.5) -> None:
        stain_matrix = np.array([[0.91633014, -0.20408072, -0.34451435],
                                [0.17669817, 0.92528011, 0.33561059]])
        self.stain_augmentor = StainAugmentor(
            "macenko",
            stain_matrix=stain_matrix,
            sigma1=0.2,
            sigma2=0.2,
            augment_background=True)
        self.p = p
    def __call__(self, img):
        """
        :param img: (PIL): Image 

        :return: ycbr color space image (PIL)
        """
        if np.random.random() > self.p:
            return img
        
        img = np.asarray(img, dtype='uint8')
        """Perturb the stain intensities of input images, i.e. stain augmentation."""
        
        # # ret = np.uint8(stain_augmentor.apply(np.uint8(img.copy())))
        ret = self.stain_augmentor.apply(img.copy())

        return Image.fromarray(ret)

    def __repr__(self):
        return self.__class__.__name__+'()'
