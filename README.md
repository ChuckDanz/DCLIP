# Distill CLIP (DCLIP): Enhancing Image-Text Retrieval via Cross-Modal Transformer Distillation
Repository and code for DCLIP

A lightweight distillation of CLIP that injects region-level cross-attention supervision to boost image–text retrieval by 15–35 pp while retaining ~94 % zero-shot accuracy, all on consumer GPUs.

## Key Features
- **Asymmetric distillation:** Fine-tune only the vision encoder; text encoder remains frozen.  
- **Meta-teacher:** A bidirectional cross-attention teacher that aligns YOLO-extracted regions to text spans.  
- **Efficient:** Trains in under 2 h/epoch on an RTX 2070 Super (8 GB).  
- **Small data:** Only ~67 K pairs needed for strong retrieval gains.

## Getting Started
```bash
git clone https://github.com/yourusername/DCLIP.git
cd DCLIP
conda create -n dclip python=3.9  # or virtualenv
pip install -r requirements.txt
```

## Steps to Train DCLIP
**Note** Leave ALL prjection module paths blank.

1. Install the various datasets you would like, I would recommend at least MSCOCO and Flickr30k. Under the `json_creation` open the `big_teacher_data.py` script. Adjust the values accordingly and run this command:
```bash
python big_teacher_data.py --output_dir "PATH TO OUTPUT DIRECTORY" \ 
--coco_images "path/to/MSCOCO/images/train2017" \
--coco_annotations "path/to/MSCOCO/annotations/captions_train2017.json" \ 
--flickr_images "path/to/flickr30k/images/flickr30k_images"  \
--flickr_annotations "path/to/flickr30k/annotations/results.csv" 
```
Note that you can also utilize Visual Genome and Conceptual Captions you just have to update the arguments accordingly.

This should create two datasets one for training and then one for validation.

2. For maximum efficiency we must cache both the CLIP embeddings and the YOLO bounding boxes.
Under the `training` folder open `train_pickle.py`

In the `main` function you will see a variable `json_file`, edit this variable to your training.json. Run `train_pickle.py`, then change that same path to the validation set and run `train_pickle.py` again. Now you should have two caches in a cache folder.

3. Now open `train_contrastive_teacher.py`, this is how we will train our meta teacher. In the script change ALL of the cache paths accordingly. Make sure all of the CLIP models are correct then run this command:
```bash
python train_contrastive_teacher.py --train_file "PATH/TO/TRAIN/JSON/teacher_100k_train.json" \ 
--epochs 5 \ 
--batch_size 32 \ 
--gradient_accumulation 1
```
For CLIP's ViT-B models train the teacher for 5 epochs. ViT-L requires less training time so train for 1 epoch.

4. Now open `CLIP_image_distillation.py`, now we are going to distill to a CLIP model. Go down to the functions `train_dataloader` and `val_dataloader` and change the cache paths. Then run this command: 
```bash 
python CLIP_image_distill_training.py --train_file "PATH\TO\TRAIN\JSON\teacher_100k_train.json" \
--val_file "PATH\TO\VAL\JSON\teacher_10k_val.json" \
--train_batch_size 32 \ 
--eval_batch_size 32 \ 
--learning_rate 1e-6 \
--phase1_epochs 15 \ 
--checkpoint_dir "./checkpoints"
```
Train the student for only 2 epochs to prevent 0 shot decay.

Great! Now you have sucessfully distilled a DCLIP model.

## Metrics Evaluation
1. Under `json_creation` folder utilize the `karpathy_downloader.py` script to download and make a json of the Karpathy split. Run the following command:
```bash
python karpathy_download.py --datasets both \
--coco_dir "path/to/coco" --flickr_dir \
"path/to/flickr" \
--output_dir "path/for/output"
```

2. Now under `eval_scripts` go to `flickr30k_eval.py`. Change the variable `DATASET_JSON` to the correct karpathy path that you want. Then change `checkpoint_path.py` to your new trained DCLIP model.

3. Run `flickr30k_eval.py`

**Zero Shot Evaluation**
1. On your web browser install the Image-Net 1k validation set. 

2. Open `test_zero_shot.py` change the `custom_model` path accordingly and change the default CLIP model to the one you want to evaluate against. Also change the zip path and the extract folder accordingly. 

3. Run `test_zero_shot.py`


