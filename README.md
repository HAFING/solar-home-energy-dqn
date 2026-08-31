# Optimisation de la gestion énergétique d’une maison solaire par DQN

Projet académique de Reinforcement Learning réalisé dans le cadre du Master en Intelligence Artificielle du Dakar Institute of Technology.

Le projet entraîne un agent Deep Q-Network (DQN), implémenté manuellement avec PyTorch, afin de piloter la batterie d’une maison équipée de panneaux solaires et connectée à un réseau électrique sujet à des coupures.

L’objectif est d’équilibrer l’autonomie énergétique, la continuité de service pendant les coupures et la préservation de la batterie.

## Membres du groupe

| Membre | Contribution principale |
|---|---|
| Metanha | Préparation, nettoyage et normalisation des données énergétiques |
| Takor | Modélisation de l’environnement énergétique et de la batterie |
| Ndiaye | Implémentation manuelle du DQN avec PyTorch |
| Davy | Intégration, méthodologie, entraînement, évaluation, expériences, visualisation et démonstration Gradio |

## Objectifs

À chaque heure, la maison possède :

- une production photovoltaïque ;
- une consommation électrique ;
- une batterie ;
- un réseau électrique disponible ou indisponible.

L’agent choisit une action parmi :

1. ne rien faire ;
2. charger la batterie ;
3. décharger la batterie.

Les objectifs sont :

- réduire l’importation depuis le réseau ;
- améliorer l’autonomie énergétique ;
- limiter la demande non satisfaite pendant les coupures ;
- limiter l’usure liée aux cycles de la batterie.

La revente d’électricité au réseau n’est pas prise en compte.

## Organisation du projet

```text
Données énergétiques
        ↓
Nettoyage et normalisation
        ↓
Simulation des coupures
        ↓
Environnement Gymnasium
        ↓
Agent DQN manuel avec PyTorch
        ↓
Entraînement et validation
        ↓
Évaluation sur données de test
        ↓
Expériences multi-graines et comparaison
        ↓
Démonstration interactive Gradio
```

## Données

Le jeu de données contient la production photovoltaïque et la consommation électrique horaires de l’année 2023.

- Fichier source : `data/Dataset.xlsx`
- Feuille utilisée : `2023 data`
- Source : <https://zenodo.org/records/15394961>

Colonnes utilisées :

```text
Date
PV generation (kW)
Consumption (kW)
```

Le script `src/prepare_data.py` :

- convertit les dates et valeurs numériques ;
- supprime les lignes invalides et doublons ;
- trie les observations chronologiquement ;
- remplace les valeurs énergétiques négatives par zéro ;
- ramène la consommation annuelle à 4 500 kWh ;
- ramène la production solaire à une installation de 5 kWc.

Le fichier final `data/processed_energy.csv` contient 8 760 observations horaires :

```text
timestamp
load_kwh
pv_kwh
```

## Découpage chronologique et normalisation

Les données sont séparées sans mélange aléatoire, afin d’éviter l’utilisation d’observations futures pendant l’entraînement.

| Ensemble | Durée | Observations |
|---|---:|---:|
| Entraînement | 255 jours | 6 120 |
| Validation | 55 jours | 1 320 |
| Test | 55 jours | 1 320 |

Les échelles de normalisation de la production solaire et de la consommation sont calculées uniquement sur l’ensemble d’entraînement. Les mêmes échelles sont ensuite appliquées à la validation et au test.

Cette précaution évite une fuite d’information depuis les données futures.

## Simulation des coupures

Le module `src/integration_data.py` adapte les données pour l’environnement :

```text
load_kwh → consumption
pv_kwh   → pv_production
```

Il ajoute la colonne `grid_available` :

- `1` : réseau disponible ;
- `0` : coupure du réseau.

Avec la graine `42`, la simulation produit 392 heures de coupure, avec des durées comprises entre une et quatre heures.

Les expériences utilisent plusieurs graines aléatoires afin de mesurer la stabilité des résultats.

## Environnement de Reinforcement Learning

L’environnement est défini dans `src/environment/energy_environment.py`.

La classe `EnergyEnvironment` respecte l’API de Gymnasium :

- `reset()` retourne l’observation initiale et les informations ;
- `step()` retourne l’observation suivante, la récompense, les indicateurs de fin d’épisode et les informations ;
- l’espace d’actions est discret avec trois actions ;
- l’espace d’observation contient six variables.

Gymnasium fournit une interface standard pour l’environnement ; l’algorithme DQN reste implémenté manuellement avec PyTorch.

### État

L’état observé par l’agent contient :

```text
[
    production_solaire_normalisée,
    consommation_normalisée,
    niveau_de_batterie,
    disponibilité_du_réseau,
    sinus_de_l_heure,
    cosinus_de_l_heure
]
```

Les composantes sinus et cosinus représentent le caractère cyclique des heures de la journée.

### Actions

| Action | Valeur | Signification |
|---|---:|---|
| `IDLE` | 0 | Ne rien faire |
| `CHARGE` | 1 | Charger la batterie avec le surplus solaire |
| `DISCHARGE` | 2 | Décharger la batterie pour couvrir un déficit |

### Batterie

| Paramètre | Valeur |
|---|---:|
| Capacité | 10 kWh |
| État de charge initial | 50 % |
| Puissance maximale | 2 kW |
| Rendement | 95 % |

L’environnement calcule également :

- le débit énergétique total de la batterie ;
- le coût de dégradation associé à ce débit ;
- les cycles équivalents.

```text
cycles_équivalents = débit_batterie / (2 × capacité_batterie)
```

### Récompense

La récompense encourage l’utilisation directe de l’énergie solaire et pénalise :

- l’importation depuis le réseau ;
- la demande non satisfaite ;
- le débit de la batterie, afin de représenter son usure.

```text
récompense =
    énergie_solaire_utilisée
    - importation_du_réseau
    - 5 × demande_non_satisfaite
    - 0,01 × débit_de_la_batterie
```

La demande non satisfaite reçoit la pénalité la plus élevée, car elle correspond à une consommation qui n’a pas pu être alimentée pendant une coupure.

## Agent DQN

Le DQN est implémenté manuellement avec PyTorch, sans Stable-Baselines3 ni RLlib.

Fichiers principaux :

```text
src/dqn.py
src/agent.py
src/replay_buffer.py
```

### Architecture du réseau

```text
Entrée : 6 variables
        ↓
Couche cachée : 64 neurones + ReLU
        ↓
Couche cachée : 64 neurones + ReLU
        ↓
Sortie : 3 valeurs Q
```

Chaque sortie représente la valeur Q estimée pour une action.

### Composants implémentés

- réseau de politique ;
- réseau cible ;
- replay buffer ;
- stratégie epsilon-greedy ;
- équation de Bellman ;
- perte Smooth L1 ;
- optimiseur Adam ;
- gradient clipping ;
- synchronisation périodique du réseau cible ;
- sauvegarde et chargement du modèle.

### Hyperparamètres principaux

| Paramètre | Valeur |
|---|---:|
| Taille de l’état | 6 |
| Nombre d’actions | 3 |
| Taille des couches cachées | 64 |
| Taille du batch | 64 |
| Capacité du replay buffer | 50 000 |
| Gamma | 0,99 |
| Learning rate | 0,001 |
| Epsilon initial | 1,00 |
| Epsilon minimum | 0,05 |
| Epsilon decay | 0,99995 |
| Mise à jour du réseau cible | 250 optimisations |
| Épisodes par expérience | 30 |
| Graines évaluées | 42, 123, 2026 |

## Métriques d’évaluation

L’autonomie énergétique mesure la part de la consommation qui ne dépend pas du réseau :

```text
autonomie = 100 × (1 - importation_du_réseau / consommation_totale)
```

La demande non satisfaite est volontairement suivie dans une métrique distincte, car elle mesure la continuité du service pendant les coupures :

```text
satisfaction_de_la_demande =
    100 × (1 - demande_non_satisfaite / consommation_totale)
```

Cette séparation permet de distinguer clairement :

- la dépendance au réseau ;
- la capacité à satisfaire la consommation lors des coupures.

## Installation

Prérequis :

- Python 3.11 recommandé ;
- Git ;
- pip.

```powershell
git clone https://github.com/HAFING/solar-home-energy-dqn.git
cd solar-home-energy-dqn

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Utilisation

### Préparer les données

```powershell
python -m src.prepare_data
python -m src.integration_data
```

### Exécuter une expérience rapide

```powershell
python run_experiments.py --episodes 1 --seeds 42
```

### Reproduire les expériences principales

```powershell
python run_experiments.py --episodes 30 --seeds 42 123 2026
```

Cette commande produit :

```text
results/experiments/manifest.csv
results/experiments/summary.csv
results/experiments/all_evaluation_results.csv
results/experiments/training_seed_*.csv
results/experiments/evaluation_seed_*.csv
```

Le modèle officiel `models/best_dqn.pt` correspond à la graine sélectionnée selon la meilleure récompense de validation.

### Évaluer le modèle officiel

```powershell
python evaluate.py
```

### Générer les graphiques

```powershell
python plot_results.py
```

Les graphiques sont enregistrés dans :

```text
results/figures/training_progress.png
results/figures/policy_comparison.png
```

### Lancer la démonstration Gradio

```powershell
python app.py
```

Ouvrez ensuite l’adresse locale affichée par PowerShell, généralement :

```text
http://127.0.0.1:7860
```

L’interface comporte quatre onglets :

- comparaison des politiques ;
- analyse détaillée de la politique sélectionnée ;
- graphiques du projet ;
- méthodologie.

Elle permet également de relancer l’évaluation officielle à partir du modèle enregistré.

### Exécuter les tests

```powershell
python -m pytest -q
```

Résultat de la vérification finale :

```text
59 passed
```

## Résultats expérimentaux

Trois expériences indépendantes ont été exécutées avec les graines `42`, `123` et `2026`, sur 30 épisodes chacune.

La sélection du modèle est faite à partir de la récompense de validation, et non à partir des données de test.

| Graine | Épisodes | Meilleure récompense de validation |
|---:|---:|---:|
| 42 | 30 | 64,159038 |
| 123 | 30 | 56,597855 |
| 2026 | 30 | 52,916793 |

La meilleure graine est `42`. Son modèle devient le modèle officiel évalué dans `models/best_dqn.pt`.

### Résultats moyens sur trois graines

| Politique | Récompense | Import réseau | Demande non satisfaite | Autonomie | Cycles équivalents |
|---|---:|---:|---:|---:|---:|
| Batterie inactive | -411,01 ± 9,61 | 496,45 ± 2,40 kWh | 21,52 ± 2,40 kWh | 30,04 ± 0,34 % | 0,00 |
| Règles simples | -309,27 ± 1,75 | 496,45 ± 2,40 kWh | 1,08 ± 0,12 kWh | 30,04 ± 0,34 % | 2,38 ± 0,25 |
| DQN | -324,00 ± 9,68 | 454,44 ± 3,19 kWh | 12,31 ± 2,56 kWh | 36,09 ± 0,45 % | 5,29 ± 0,04 |

### Résultats du modèle officiel, graine 42

| Politique | Récompense | Import réseau | Demande non satisfaite | Autonomie | Cycles équivalents |
|---|---:|---:|---:|---:|---:|
| Batterie inactive | -421,79 | 493,75 kWh | 24,22 kWh | 30,56 % | 0,00 |
| Règles simples | -307,30 | 493,75 kWh | 1,21 kWh | 30,56 % | 2,66 |
| DQN | -322,31 | 454,28 kWh | 12,00 kWh | 36,11 % | 5,33 |

![Progression de l’entraînement](results/figures/training_progress.png)

![Comparaison des politiques](results/figures/policy_comparison.png)

## Analyse

Comparativement à une batterie inactive, le DQN :

- réduit l’importation moyenne du réseau d’environ 42 kWh ;
- augmente l’autonomie moyenne de 6,05 points ;
- réduit la demande non satisfaite moyenne ;
- produit des résultats stables sur les trois graines.

La stratégie à règles simples reste néanmoins meilleure pour :

- la récompense globale ;
- la limitation de la demande non satisfaite ;
- la préservation de la batterie.

Le DQN obtient la meilleure autonomie, mais sollicite davantage la batterie : environ 5,29 cycles équivalents, contre 2,38 pour la politique à règles.

Ce résultat est cohérent avec l’absence de prévision des futures coupures : la politique à règles conserve plus souvent la batterie pour les coupures, tandis que le DQN la décharge plus volontiers pour réduire l’importation du réseau.

Le projet met ainsi en évidence un compromis concret entre autonomie, continuité de service et durée de vie de la batterie.

## Limites et améliorations possibles

Les principales limites sont :

- les coupures sont simulées, et non observées dans des données réelles ;
- l’agent ne connaît pas les futures coupures ;
- l’agent ne dispose pas de prévisions de production solaire ou de consommation ;
- seules trois graines et trente épisodes ont été évalués ;
- le DQN reste moins performant que les règles simples sur certains critères ;
- les coefficients de récompense peuvent encore être ajustés.

Améliorations envisageables :

- ajouter des prévisions de consommation et de production solaire ;
- estimer le risque de coupure future ;
- tester davantage de graines et d’épisodes ;
- effectuer une recherche d’hyperparamètres ;
- comparer DQN, Double DQN et Dueling DQN ;
- utiliser les capacités de parallélisation de RLlib pour des campagnes plus larges ;
- permettre à l’utilisateur de modifier les paramètres de simulation dans Gradio.

## Structure du dépôt

```text
solar-home-energy-dqn/
├── data/
│   ├── Dataset.xlsx
│   └── processed_energy.csv
├── models/
│   └── best_dqn.pt
├── results/
│   ├── experiments/
│   ├── figures/
│   ├── evaluation_results.csv
│   └── training_metrics.csv
├── src/
│   ├── environment/
│   │   └── energy_environment.py
│   ├── agent.py
│   ├── dqn.py
│   ├── integration_data.py
│   ├── prepare_data.py
│   └── replay_buffer.py
├── tests/
├── app.py
├── evaluate.py
├── plot_results.py
├── run_experiments.py
├── train.py
├── requirements.txt
└── README.md
```

## Collaboration et versionnement

Le projet a été construit avec des branches et Pull Requests distinctes afin de conserver la traçabilité des contributions.

```text
data/energy-pipeline
feature/energy-environment
feature/dqn-network
integration/training-evaluation
fix/rl-methodology
```

## Conclusion

Ce projet couvre l’ensemble de la chaîne d’un problème de Reinforcement Learning :

```text
Données
    ↓
Environnement
    ↓
États, actions et récompenses
    ↓
Agent DQN
    ↓
Entraînement et validation
    ↓
Évaluation sur données de test
    ↓
Analyse reproductible sur plusieurs graines
    ↓
Démonstration interactive
```

Le DQN réduit la dépendance au réseau et obtient la meilleure autonomie énergétique. La comparaison avec une politique à règles met également en évidence l’importance de la conception de la récompense, de la continuité de service et de la préservation de la batterie.