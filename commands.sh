### A list of some useful commands

## Setup
# Normal setup
python setup.py bpe

# 5k vocab setup
python setup.py bpe --data_sub_dir=bpe5k --max_tokens=5000

## Roberta
python train.py roberta_pretrain -n roberta_pretrain --batch_size=12 --gradient_accumulation=1
--n_layers=12 --num_epochs=1000 --lr=0.04 --warmup_steps=10000 --power_decay=-0.5 --decay_forever=True --prenorm=true


## Electra
python train.py electra_pretrain -n electra_pretrain --batch_size=12 --gradient_accumulation=1
--n_layers=12 --num_epochs=1000 --lr=0.08 --warmup_steps=10000 --power_decay=-0.5 --decay_forever=True --prenorm=true

# With 5k vocab
python train.py electra_pretrain -n electra_pretrain --batch_size=16 --gradient_accumulation=1
--n_layers=12 --num_epochs=1000 --lr=0.08 --warmup_steps=10000 --power_decay=-0.5 --decay_forever=True --prenorm=true
--data_sub_dir=bpe5k

## Finetune
python train.py roberta_finetune -n electra_finetune --batch_size=12 --gradient_accumulation=1
--n_layers=12 --num_epochs=100 --lr=0.004 --warmup_steps=1300 --power_decay=-0.5 --decay_forever=True --prenorm=true
--na_class_weight=3.0 --eval_per_n_samples=12500 --load_path=save/train/electra_pretrain-01/best.pth.tar

# With 5k Vocab
python train.py roberta_finetune -n electra_finetune --batch_size=16 --gradient_accumulation=1
--n_layers=12 --num_epochs=100 --lr=0.004 --warmup_steps=1300 --power_decay=-0.5 --decay_forever=True --prenorm=true
--eval_per_n_samples=12500 --load_path=save/train/electra_pretrain-21/best.pth.tar
--data_sub_dir=bpe5k


## DIDAE
python train.py didae_pretrain -n didae_pretrain --batch_size=16 --gradient_accumulation=1 --n_layers=12 --num_epochs=1000
 --lr=0.08 --warmup_steps=10000 --power_decay=-0.5 --decay_forever=True --prenorm=true
--data_sub_dir=bpe5k

python train.py didae_pretrain -n didae_pretrain --batch_size=16 --gradient_accumulation=1 --n_layers=12 --num_epochs=1000
--lr=0.08 --warmup_steps=10000 --power_decay=-0.5 --decay_forever=True --prenorm=true  --mlm_sample_temperature=1.0 --data_sub_dir=bpe5k

!python train.py didae_pretrain -n didae_pretrain --batch_size=16 --gradient_accumulation=1 --n_layers=12 --num_epochs=1000
 --lr=0.08 --warmup_steps=10000 --power_decay=-0.5 --decay_forever=True --prenorm=true --mask_prob=0.35 --hint_prob=0.25 --sample_temperature=1.0
 --mlm_sample_temperature=1.0 --data_sub_dir=bpe5k

!python train.py didae_pretrain -n didae_pretrain --batch_size=16 --gradient_accumulation=1 --n_layers=12 --num_epochs=1000
 --lr=0.12 --warmup_steps=10000 --power_decay=-0.5 --decay_forever=True --prenorm=true --mask_prob=0.25 --hint_prob=0.10 --sample_temperature=0.5
  --mlm_sample_temperature=2.0 --data_sub_dir=bpe5k

## Finetune
!python train.py roberta_finetune -n didae_finetune --batch_size=16 --gradient_accumulation=1
 --n_layers=12 --num_epochs=100 --lr=0.004 --warmup_steps=1300 --power_decay=-0.5 --decay_forever=True --prenorm=true
  --eval_per_n_samples=12500 --data_sub_dir=bpe5k --load_path=save/train/didae_pretrain-11/checkpoint.pth.tar


## Augmentation
python train.py roberta_augment -n roberta_augment --batch_size=128 --gradient_accumulation=1 --n_layers=12 --num_epochs=1
 --lr=0.025 --power_decay=-0.5 --decay_forever=True --prenorm=true --sample_temperature=1.0 --data_sub_dir=bpe5k_aug --load_path=save/train/electra_pretrain-21/best.pth.tar

## Augmentation x4
python train.py roberta_augment -n roberta_augment --batch_size=128 --gradient_accumulation=1 --n_layers=12 --num_epochs=1 
 --lr=0.025 --power_decay=-0.5 --decay_forever=True --prenorm=true --sample_temperature=1.1 --augment_samples=4 --data_sub_dir=bpe5k_aug4
  --load_path=save/train/electra_pretrain-21/best.pth.tar

## Augment bpe
python setup.py bpe_aug --data_sub_dir=bpe5k_aug4

## Finetune augmentation
python train.py roberta_finetune -n electra_finetune_aug --batch_size=16 --gradient_accumulation=1 --n_layers=12 --num_epochs=100
--lr=0.004 --warmup_steps=1300 --power_decay=-0.5 --decay_forever=True --prenorm=true --na_class_weight=0.4 --eval_per_n_samples=12500 --data_sub_dir=bpe5k_aug
--load_path=save/train/electra_pretrain-21/best.pth.tar

## Finetune augmentation x4
python train.py roberta_finetune -n electra_finetune_aug --batch_size=16 --gradient_accumulation=1 --n_layers=12 --num_epochs=100
--lr=0.004 --warmup_steps=1300 --power_decay=-0.5 --decay_forever=True --prenorm=true --na_class_weight=0.2 --eval_per_n_samples=12500 --data_sub_dir=bpe5k_aug4
--load_path=save/train/electra_pretrain-21/best.pth.tar


## SCP
scp -i /content/.ssh/azure.pem -r jdshen@XXX:~/squad/save/train/electra_pretrain-02/ ./electra_pretrain-02/
scp -i /content/.ssh/azure.pem -r ./electra_pretrain-13/ jdshen@XXX:~/squad/save/train/electra_pretrain-13/

scp -i /content/.ssh/azure.pem -r ./didae_pretrain-11/ jdshen@XXX:~/squad/save/train/didae_pretrain-11/
scp -i /content/.ssh/azure.pem -r ./didae_finetune-01/ jdshen@XXX:~/squad/save/train/didae_finetune-11/
