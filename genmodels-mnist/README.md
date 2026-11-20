# Flow matching vs diffusion (DDPM) sur MNIST

Projet personnel : implémentation from scratch (PyTorch) de deux modèles génératifs
et comparaison sur MNIST.

- **Flow matching** (trajectoires linéaires / rectified flow) : régression d'un champ de vitesse, génération par intégration d'une EDO.
- **DDPM** (Ho et al., 2020) : prédiction du bruit, échantillonnage ancestral en 1000 pas.
- Même U-Net pour les deux, afin de comparer proprement les méthodes.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation
```bash
python train.py --method flow --epochs 20
python train.py --method ddpm --epochs 20
python sample.py --method flow --steps 2 5 10 50
```
Test rapide du code sans télécharger MNIST : `python train.py --method flow --epochs 1 --fake`.

## Résultats
> À COMPLÉTER avec tes propres résultats (images, courbes, temps).

| Méthode | Loss finale | Pas d'échantillonnage | Temps de génération |
|---|---|---|---|
| Flow matching | | 50 | |
| DDPM | | 1000 | |

Effet du nombre de pas (flow matching) : `outputs/flow/gen_*steps.png`.

## Ce que j'ai appris
> À COMPLÉTER (3-5 phrases, avec tes mots).

## Références
- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
- Liu et al., *Flow Straight and Fast: Rectified Flow*, ICLR 2023.
- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
