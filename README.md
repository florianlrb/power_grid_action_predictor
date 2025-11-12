# POC - Power grid action predictior
Florian Larab - 2025

Ce projet met en place un workflow de bout en bout autour de Grid2Op pour prédire les conséquences d’actions sur le réseau (max rho post-action). Le but est de montrer une approche simple, structurée et testable : génération des données + cache, entraînement de modèles classiques, visualisation, notebook d’exploration et tests unitaires.

Certains commentaires sont en français, d'autres en anglais. Il faut être bilingual (at least) pour être un bon ingénieur.

> **Tech stack :** Python, Grid2Op, Polars, scikit‑learn, Matplotlib.  
> **OS cible :** Windows (fonctionne aussi sous Linux/macOS (normalement)). J'ai codé ceci avec un IDE VSCode, avec une invite de commande git bash.

---

## Points clés

- **Données** (style `main.py`) :
  - `extract_features(obs)` : vecteurise l’observation (gen/load/topo/rho).  
  - `create_realistic_observation(episode_count, env)` : collecte une liste d’états via “do‑nothing”.  
  - `create_training_data(list_obs, all_actions)` : cible = `max(rho)` **post‑action** (∞ si non‑convergence).
- **Cache** : Parquet en `data/cache/` avec clé sur `{case, episodes, actions}`. `--force` pour re‑générer.
- **ML** : modèles **simples** au choix (`dtree`, `rf`, `ridge`, `gbr`), **cross‑validation KFold**, métriques **RMSE moyen** et **Top‑1 accuracy** (l’action prédite la plus sûre est‑elle la vraie meilleure ?).
- **Visualisation** : histogrammes (cibles), scatter préd./vrai, plus des cellules notebook pour aller plus loin.
- **Notebook** : `notebooks/explore.ipynb` (chargement robuste + figures).
- On utilise argparse pour pouvoir paramétrer l'usage de ce package dans l'invite de commandes. Cela fonctionne sur ma config (windows/VSCode/invite git bash)

---

## Arborescence

```
.
├── requirements.txt
├── README.md
├── data/
│   └── cache/                # features_<key>.parquet, targets_<key>.parquet, actions_<key>.json
├── notebooks/
│   └── explore.ipynb
├── rte_challenge/
│   ├── __init__.py
│   ├── data.py               # extract_features / create_realistic_observation / create_training_data + test cache 
│   ├── model.py              # modèles, split, CV, évaluation, artefacts
│   └── viz.py                # figures simples
├── scripts/
│   ├── generate_data.py      # génère et met en cache
│   ├── train_model.py        # split, CV (optionnelle), entraînement, évaluation, artefacts
│   └── visualize.py          # figures à partir du cache et/ou des prédictions
└── tests/
    ├── conftest.py           # fixtures communes (root, python, cache rapide)
    ├── test_generate_data.py # par souci de temps,on teste avec les données en cache...
    ├── test_train_model.py   # entraînement, métriques, artefacts
    └── test_visualize.py     # génération des figures

```

---

## Installation

```bash
# 1) Environnement virtuel
python -m venv .venv
# Git Bash (Windows) :
source .venv/Scripts/activate
# PowerShell :
# .\.venv\Scripts\Activate.ps1

# 2) Dépendances
pip install -r requirements.txt
```


---

## Cas d'usage classique de ce code

### 1) Générer les données (cache)
```bash
python scripts/generate_data.py --episodes 2 --actions 100 --cache-dir data/cache
# Regénérer explicitement :
python scripts/generate_data.py --episodes 2 --actions 100 --cache-dir data/cache --force

```

Ceci crée dans `data/cache/` :  
- `features_<key>.parquet` (X : observations × features)  
- `targets_<key>.parquet` (Y : observations × actions)  
- `actions_<key>.json` (trace des actions candidates, `repr`, index)

> **Y[i, j] =** `max(rho)` **après action j** à l’observation i. **Petit = mieux**. `∞` si la simu n’a pas convergé -> action à éviter.

### 2) Entraîner (modèle + métriques + artefacts)
```bash
# Entraînement direct (DecisionTree par défaut)
python scripts/train_model.py --cache-dir data/cache --out-dir data/processed --test-size 0.2

# Cross-validation KFold et comparaison multi-modèles (choisit le meilleur et entraîne ensuite)
python scripts/train_model.py --cache-dir data/cache --out-dir data/processed   --model dtree,rf,ridge,gbr --cv 5 --test-size 0.2 --random-state 42
```

**Sorties** (dans `data/processed/`) :  
- `model.pkl`, `X_test.npy`, `Y_test.npy`, `Y_pred.npy`  
- Impression console des métriques : `rmse_mean`, `top1_acc`

>Essayer d'entraîner tous les modèles pour les comparer, avec la cv=4 ou 5 prend beaucop de temps. Par exemple, j'ai un ryzen7 3700X (overclock maison) et je n'avais rien au bout de 10 minutes avec tous les modèles et cv=5. 
Le modèle le plus rapide semble être celui par défaut, mais ce n'est pas forcément le meilleur.

### 3) Visualiser
```bash
python scripts/visualize.py --cache-dir data/cache --out-dir data/processed
```
- `targets_hist.png` : distribution de `rho` (finies)  
- `pred_vs_true.png` : scatter prédiction vs vérité (si artefacts présents)

### 4) Notebook
Ouvre `notebooks/explore.ipynb`.
La première cellule détecte la racine, charge le dernier cache et peut regénérer un mini-dataset si besoin.
Exemples de visualisations supplémentaires inclus : distributions, profils d’actions, PCA 2D, courbe top-k, etc.

---

## Détails de la partie ML

- **Modèles disponibles :**
  - `dtree` : `DecisionTreeRegressor`  
  - `rf` : `RandomForestRegressor`  
  - `ridge` : `Ridge` 
  - `gbr` : `GradientBoostingRegressor` 
  Tous sont gérés via un **`MultiOutputRegressor`** (une sortie par action).

- **Cibles infinies** (`∞`) : remplacées par `max(fini) + 1` avant apprentissage (pénalisation des actions dangereuses / non-convergentes).

- **Métriques :**
  - **RMSE moyen** (moyenne du RMSE sur les colonnes actions)
  - **Top‑1 accuracy** : est‑ce que l’action prédite la plus “sûre” (min ρ) est‑elle la meilleure vraie ?

- **Cross‑validation (KFold)** : `--cv k` lance k folds, affiche **moyenne ± écart‑type** pour RMSE et Top‑1, et, si plusieurs modèles sont passés via `--model`, **choisit le meilleur** (RMSE minimum) pour l’entraînement final.

---

## Détails des données

- **X (features)** : concaténation aplatie des principales observables de Grid2Op (puissance active et réactive générée et consommée, vecteur toplogique définissant le réseau, rho (facteur de charge))
  - `gen_p`, `gen_q`, `load_p`, `load_q`, `topo_vect`, `rho`
- **Y (targets)** : **observations × actions**
  - Chaque colonne correspond à une **action candidate** (actions aléatoires + **do‑nothing**).  
  - Colnames typiques (min du rho) : `rho_0`, `rho_1`, … .

**Important :** certaines versions de Polars inversent (colonnes vs lignes) si l’orientation n’est pas précisée. Le code **force `orient="row"`** lors de la création de Y, et le script d’entraînement **vérifie et transpose** si un ancien cache inversé est détecté. Normalement, j'ai forcé la bonne version de polars dans le requirements, mais on ne sait jamais : mieux vaut crééer un environnement local tel que je le décris plus haut, avec les requirements que je donne. J'ai été bloqué par cette erreur pendant quelques temps,donc j'ai inclus de quoi la parer dans les scripts et dans les tests.


