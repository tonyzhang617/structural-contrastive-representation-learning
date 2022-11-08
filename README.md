# Official codebase for the paper "Structural Contrastive Representation Learning for Zero-shot Multi-label Text Classification" in Findings of EMNLP 2022

Code tested with `Python 3.7` on `Ubuntu 20.04`.

To run, please first install the packages in `requirements.txt` and follow the instructions in `dataset/README.md` to download datasets.

### Commands for Reproducing Results from Paper

```sh
# LF-Amazon-131K
python train.py --dataset dataset/LF-Amazon-131K --min-len 40 --max-len 80 --batch-size 384 --experiment a131k

# LF-Amazon-1M
python train.py --dataset dataset/LF-Amazon-1M --min-len 40 --max-len 80 --batch-size 384 --lr 5e-6 --min-lr 5e-7 --experiment a1m

# LF-WikiSeeAlso-320K
python train.py --dataset dataset/LF-WikiSeeAlso-320K --min-len 80 --max-len 160 --batch-size 256 --lr 5e-8 --min-lr 5e-9 --epochs 5 --experiment wsa320k

# LF-Wikipedia-500K
python train.py --dataset dataset/LF-Wikipedia-500K --min-len 80 --max-len 160 --batch-size 256 --lr 5e-8 --min-lr 5e-9 --epochs 5 --experiment w500k
```
