import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from CLIP_image_distillation import CLIPImageDistillation
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.multiprocessing as mp

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Use Tensor Cores more effectively
#torch.set_float32_matmul_precision('medium')

#REMOVE THE UNECCASARY TRAINING STAGES



# use transformer version on CLIP
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device) #openai/clip-vit-large-patch14 #openai/clip-vit-base-patch16
    clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model =CLIPImageDistillation(args, clip_model, clip_preprocess)
    
    # Define checkpoint callback
    def get_checkpoint_callback():
        return ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="epoch-{epoch:02d}-{train_loss:.2f}",
            save_top_k=10,
            monitor="train_loss",
            mode="min"
        )

    trainer = Trainer(
        max_epochs=args.phase1_epochs,
        accelerator="gpu",
        devices=1,
        precision=32,
        gradient_clip_val=0.5,
        accumulate_grad_batches=4, 
        callbacks=[get_checkpoint_callback()]
    )
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Image Distillation Training")
    parser = CLIPImageDistillation.add_model_specific_args(parser)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--phase1_epochs", type=int, default=10, help="Number of epochs for Phase 1 training.")
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    
    main(args)

    # python CLIP_image_distill_training.py --train_file "path/to/your/data.json" --val_file "path/to/your/val_data.json" --train_batch_size 128 --eval_batch_size 128 --learning_rate 1e-5 --phase1_epochs 20 --checkpoint_dir "./checkpoints"

    # python CLIP_image_distill_training.py --train_file "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/small_train_coco_dataset.json" --val_file "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/small_val_coco_dataset.json" --train_batch_size 128 --eval_batch_size 128 --learning_rate 1e-5 --phase1_epochs 20 --checkpoint_dir "./checkpoints"
    
    # python CLIP_image_distill_training.py --train_file "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/small_train_coco_dataset.json" --val_file "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/small_val_coco_dataset.json" --train_batch_size 64 --eval_batch_size 64 --learning_rate 1e-5 --phase1_epochs 20 --checkpoint_dir "./checkpoints"
    
    # python CLIP_image_distill_training.py --train_file "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/small_train_coco_dataset.json" --val_file "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/small_val_coco_dataset.json" --train_batch_size 32 --eval_batch_size 32 --learning_rate 1e-6 --phase1_epochs 20 --checkpoint_dir "./checkpoints"

    # python CLIP_image_distill_training.py --train_file "C:\Users\Daniel Csizmadia\Desktop\TokenizerCLIP\teacher_dataset\teacher_100k_train.json" --val_file "C:\Users\Daniel Csizmadia\Desktop\TokenizerCLIP\teacher_dataset\teacher_10k_val.json" --train_batch_size 64 --eval_batch_size 64 --learning_rate 1e-6 --phase1_epochs 15 --checkpoint_dir "./checkpoints"

    