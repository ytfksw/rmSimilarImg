import numpy as np
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input, decode_predictions


class baseExtractor(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def feature_extraction(self, images):
        pass


class XceptionExtractor(baseExtractor):
    def __init__(self):
        
        self.model = Xception(weights='imagenet', include_top=False)
        self.batch_size = 32

    def feature_extraction(self, images):
        x = preprocess_input(images)
        out = []
        num_data = len(x)
        for start in tqdm(range(0, num_data, self.batch_size)):
            if start +self.batch_size < num_data:
                end = start + self.batch_size
            else:
                end = num_data

            preds = self.model.predict(x[np.arange(start, end, 1), :, :, :])
            preds = preds.reshape(preds.shape[0],
                                  preds.shape[1]*preds.shape[2]*preds.shape[3]
                                  )
            out.extend(preds)
        return out


class WHashExtractor(baseExtractor):
    def __init__(self):
        pass

    def feature_extraction(self, images):
        import imagehash
        from PIL import Image
        out = []
        for image in images:
            im = Image.fromarray(np.uint8(image))
            out.append(imagehash.whash(im).hash.astype(float).flatten())
        return out


def main():
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    img_one_array = image.img_to_array(img)
    images = np.expand_dims(img_one_array, axis=0).repeat(33, axis=0)
    extractor = XceptionExtractor()
    print(len(extractor.feature_extraction(images)))


if __name__ == "__main__":
    main()

