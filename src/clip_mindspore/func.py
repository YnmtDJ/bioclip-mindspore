from mindspore.common import dtype as mstype
import os
from mindspore.dataset import Dataset
import pandas as pd
from ..imageomics import naming_eval
from PIL import Image

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = mstype.bfloat16
    elif precision == 'fp16':
        cast_dtype = mstype.float16
    else:
        cast_dtype = mstype.float32
    return cast_dtype

def img_loader(filepath):
    img = Image.open(filepath)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    return img

class DatasetFromFile(Dataset):
    def __init__(self, filepath, label_filepath=None, transform=None, classes='asis'):
        super(DatasetFromFile, self).__init__()
        self.basefilepath = filepath
        if label_filepath is None:
            label_filepath = os.path.join(self.basefilepath, 'metadata.csv')
        else:
            label_filepath = os.path.join(self.basefilepath, label_filepath)

        self.data = pd.read_csv(label_filepath, index_col=0).fillna('')
        self.transform = transform
        self.data['class'] = naming_eval.to_classes(self.data, classes)
        self.classes = self.data['class'].unique()
        # create class_to_idx dict
        if 'class_idx' in self.data.columns:
            self.class_to_idx = dict([(x, y) for x, y in zip(self.data['class'], self.data['class_idx'])])
        else:
            self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
            self.data['class_idx'] = self.data['class'].apply(lambda x: self.class_to_idx[x])

        self.idx_to_class = dict([(v, k) for k, v in self.class_to_idx.items()])
        self.samples = self.data['filepath'].values.tolist()

        self.path_class_index = self.data.iloc

    def __next__(self):
        if self._index >= len(self.data):
            raise StopIteration
        else:
            item = self.path_class_index[self._index]
            filepath = os.path.join(self.basefilepath, item['filepath'].split('/')[-1])
            img = img_loader(filepath)
            if self.transform is not None:
                img = self.transform(img)
            label = item['class_idx']
            output = (img, label)
            self._index += 1
            return output

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self.data)


