## Image Captioning

Image Captioning on [Flick](https://github.com/li-xirong/flickr8kcn) data with attention and reinforcement learning.

Reference: [ruotianluo/self-critical.pytorch: Unofficial pytorch implementation for Self-critical Sequence Training for Image Captioning. and others. (github.com)](https://github.com/ruotianluo/self-critical.pytorch)


## Data preparing

* 6K images with id in `flickr7ktrain.txt`, 1K validation in `flickr7kval.txt` and 1K test in `flickr7ktest.txt`.
* Captions
    * Original English sentences: `flickr8kenc.caption.txt`
    * Chinese sentences: 
        * original: `flickr8kzhc.caption.txt`
        * Baidu translation: `flickr8kzhb.caption.txt`
        * Google translation: `flickr8kzhg.caption.txt`
        * human translation(for test set only): `flickr8kzhmtest.captions.txt`

#### Features

Use **1,024-dim GoogleNet pool5**.

Include three files:
* `shape.txt`: shape of (sample size, features) = (8091, 1024)
* `id.txt`: id of each images
* `feature.bin`: feature for each images

## Data preprocessing

#### (1) Convert caption to `json`

We need to convert our data to the required `json` from.

First let's see the structure of `json` file:

```{json}
{"images": [{"filepath": "val2014", "sentids": [770337, 771687, 772707, 776154, 781998], "filename": "COCO_val2014_000000391895.jpg", "imgid": 0, "split": "test", "sentences": [{"tokens": ["a", "man", "with", "a", "red", "helmet", "on", "a", "small", "moped", "on", "a", "dirt", "road"], "raw": "A man with a red helmet on a small moped on a dirt road. ", "imgid": 0, "sentid": 770337}, {"tokens": ["man", "riding", "a", "motor", "bike", "on", "a", "dirt", "road", "on", "the", "countryside"], "raw": "Man riding a motor bike on a dirt road on the countryside.", "imgid": 0, "sentid": 771687}, {"tokens": ["a", "man", "riding", "on", "the", "back", "of", "a", "motorcycle"], "raw": "A man riding on the back of a motorcycle.", "imgid": 0, "sentid": 772707}, {"tokens": ["a", "dirt", "path", "with", "a", "young", "person", "on", "a", "motor", "bike", "rests", "to", "the", "foreground", "of", "a", "verdant", "area", "with", "a", "bridge", "and", "a", "background", "of", "cloud", "wreathed", "mountains"], "raw": "A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ", "imgid": 0, "sentid": 776154}, {"tokens": ["a", "man", "in", "a", "red", "shirt", "and", "a", "red", "hat", "is", "on", "a", "motorcycle", "on", "a", "hill", "side"], "raw": "A man in a red shirt and a red hat is on a motorcycle on a hill side.", "imgid": 0, "sentid": 781998}], "cocoid": 391895}, ....
```

The tree structure is as follows:

* `images`: list of image captioning:
    * `filepath`: image folder under $IMAGE_ROOT
    * `sentids`: a list of sentence id 
    * `filename`: name of original image
    * `imgid`
    * `split`: train val or test
    * `sentences`: a list of sentences
        * `tokens`: a list of words
        * `raw`: a sentence
        * `imgid`: corresponding image id
        * `sentid`: id of this sentence
    * `cocoid`： In our case, it is `fkcnid` by `bigfile` order.
* `dataset`: name of dataset

For example, we can obtain a image by:
```{python}
for i,img in enumerate(imgs):
        # load the image
        I = skimage.io.imread(os.path.join(params['images_root'], img['filepath'], img['filename']))
```

#### (2) Run `prepro_labels.py`

For example
```
python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

#### (3) Download and process preextracted features

(The `scripts/prepro_feats.py` can extract features from resnet101 or etc. But now we just use downloaded features from Flick8kcn github repository.)

(The example feature file is in [google drive](https://drive.google.com/drive/folders/1eCdz62FAVCGogOuNhy87Nmlo5_I0sH2J): `cocotalk_att.tar` and `cocotalk_fc.tar`. They should be downloaded to `data/cocotalk_fc` folder. Each file is just a one dimensional numpy array with the name of `cocoid`.)

The job is done in notebook 2,where features are stored in `./data/feature/[id].npy` using functions from `bigfile.py`.


