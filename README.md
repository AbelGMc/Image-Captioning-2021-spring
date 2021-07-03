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

Use preprocessed flickr8kcn captions
```
python scripts/prepro_labels.py --input_json data/flickr8kcn_original.json --output_json data/f8ktalk.json --output_h5 data/f8ktalk
```
The image information and vocabulary are dumped into `data/f8ktalk.json` and discretized caption data are dumped into `data/f8ktalk_label.h5`.


#### (3) Train Resnet101 features
Download pretrained resnet models. The models can be downloaded from [here](https://drive.google.com/open?id=0B7fNdx_jAqhtbVYzOURMdDNHSGM), and should be placed in `data/imagenet_weights`.
```
python scripts/prepro_feats.py --input_json data/flickr8kcn_original.json --output_dir data/f8ktalk --images_root $IMAGE_ROOT
```

If you see error:  **RuntimeError: CUDA error: no kernel image is available for execution on the driver, when use pytorch on linux with RTX 3090**, you can try code:

```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

#### (4) Get the cache

```
python scripts/prepro_ngrams.py --input_json data/flickr8kcn_original.json --dict_json data/f8ktalk.json --output_pkl data/f8k-train --split train
```


### copy features
```
cp -r data/f8ktalk.json data/f8ktalk_label.h5 data/f8ktalk_att/ data/f8ktalk_fc/ XXXXXXXXPATHXXXXXXX
```

## Data training

As we are asked to use both RL and attention.

To use RL, the model must use `utils/rewards.py`, which include `init_scorer,get_self_critical_reward,get_scores,get_self_cider_scores`.

In the `train.py`, the model is build by `LossWrapper` in `captioning/loss_wrapper.py`, which is a subclass of `nn.Module`.

In the `LossWrapper`, `get_self_critical_reward` is only under two situations

* `struc_flag = True` and `opt.structure_loss_type = new_self_critical`. (need `opt.max_epochs > opt.structure_after`)
    * This occurs in `train.py` when `opt.structure_after != -1 and epoch >= opt.structure_after`
    * if `opt.structure_loss_weight > 0`,  `StructureLosses` is used, which is in `modules/losses.py`, if `opt.structure_loss_type = new_self_critical`, then RL is used.


* `struc_flag is False` and `sc_flag is True`. (need `opt.max_epochs>opt.self_critical_after` if `opt.structure_after == -1(default)`).
    * This occurs when 
    ```
    opt.self_critical_after != -1 and epoch >= opt.self_critical_after
    opt.structure_after == -1 or epoch < opt.structure_after
    ```
    * RL is directly used.



### Training preparing

As `self.document_frequency` is default to to have no assignment.

In `self-critical.pytorch/tools/train.py` we replace
```
#dp_model = torch.nn.DataParallel(model)
#dp_model.vocab = getattr(model, 'vocab', None)  # nasty
#dp_lw_model = torch.nn.DataParallel(lw_model)
```
To
```
dp_model = model 
dp_model.vocab = getattr(model, 'vocab', None)  # nasty
dp_lw_model = lw_model
```
The orginal code raise error for `DataParallel`, so we only use one CPU.


In `self-critical.pytorch/cider/pyciderevalcap/ciderD/ciderD_scorer.py` we replace
```
# df = np.log(max(1.0, self.document_frequency(ngram)
```
To
```
df = np.log(max(1.0, self.document_frequency.get(ngram,0)))
```
The original code is SB.

In `/captioning/utiles/eval_utils.py` function `eval_split`,add following before return:
```
print('average loss on validation: %.3f'%(loss_sum/loss_evals))
```



### Training


Now we Run: 

```
python tools/train.py --cfg configs/a2i2_sc.yml --id Att2in_sc  --save_checkpoint_every 1000
```

```
python tools/train.py --cfg configs/a2i2_sc.yml --id Att2in_sc  --max_epochs 2 --self_critical_after 1  --save_checkpoint_every 500
```


### Modification: Get cider score on test

In `captioning/utils/rewards.py` change `get_self_critical_reward()`
```
# return rewards

# to

return rewards,_
```

In `loss_wrapper`
```
# reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
# to
reward,_ = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)


#------------
# add in line 66
out['cider'] = _
```


In `train.py`:
```
##-----------------------------------------
if (iteration % opt.save_checkpoint_every == 0 and not opt.save_every_epoch) or \
    (epoch_done and opt.save_every_epoch):
    print("===============================================")
    print("Begin calculating cider score on TEST data set")
    
    import copy
    opt_test = copy.deepcopy(opt)
    opt_test.batch_size = 50
    opt_test.split = 'test'
    loader_test = DataLoader(opt_test)
    cider_sum = 0
    for _ in range(int(1000/opt_test.batch_size)):
        data_test = loader_test.get_batch(opt_test.split)
        print('Read data:', time.time() - start)
        torch.cuda.synchronize()
        start = time.time()
        tmp = [data_test['fc_feats'], data_test['att_feats'], data_test['labels'], data_test['masks'], data_test['att_masks']]
        tmp = [_ if _ is None else _.cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            greedy_res, _ = model(fc_feats, att_feats, att_masks,
                mode='sample',
                opt={'sample_method': opt_test.sc_sample_method,
                    'beam_size': opt_test.sc_beam_size})
            model.train()
            gen_result, sample_logprobs = model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt_test.train_sample_method,
                        'beam_size':opt_test.train_beam_size,
                        'sample_n': opt_test.train_sample_n},
                    mode='sample')
        gts = [data_test['gts'][_] for _ in torch.arange(0, len(data_test['gts'])).tolist()]
        reward, cider = get_self_critical_reward(greedy_res, gts, gen_result, opt_test)                    
        cider_sum += cider
    test_cider_score = cider_sum/int(1000/opt_test.batch_size)
    tb_summary_writer.add_scalar('test_cider_score', test_cider_score, iteration)
    print('Average cider score on test set: %.3f'%(test_cider_score))
    print("End calculating cider score on TEST data set")
    print("===============================================")
# model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data_test['gts'], torch.arange(0, len(data_test['gts'])), sc_flag, struc_flag, drop_worst_flag)
##--------------------------------------------------
```



### TO DO
1. Increase epoch （epoch = 100) (**finished**)
2. `prepro_labels.py` will map all words that occur <= 5 time to a special UNK token （modify <= 2) (**finished**)
3. `a2i2_nsc.yml(unstable)`
4. Features: Resnet101, GoogleNet (**delete**)
5. Evaluate score （**finished**）
6. Google translate

### Withour attention
```
python tools/train.py --cfg configs/fc_rl.yml --id fc_sc  --save_checkpoint_every 1000
```
