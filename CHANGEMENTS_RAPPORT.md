# Résumé des changements apportés au rapport

## Modifications architecturales

### 1. Architecture du modèle agent (Section 2)
**Changements:**
- **Blocs résiduels** : 3 → **4 blocs** (amélioration de la capacité représentationnelle)
- **Dropout** : 0.2 → **0.15** (amélioration du flux de gradient avec récompenses rares)
- **Tête de valeur** : architecture améliorée avec projection supplémentaire
  - Ancienne : `512 → 128 → 1`
  - Nouvelle : `512 → 256 → 64 → 1` avec activation Sigmoid
- **Dropout dans tête d'action** : ajout d'un dropout réduit de 50% dans la couche finale

### 2. Stratégies d'exploration et de régularisation

**Changements:**
- **Epsilon-greedy** : précision ajoutée sur la variabilité de ε (démarre à 0.5 puis décroit)
- **Bruit d'exploration** : σ = 0.15 (confirmé et justifié comme "réduit de 0.2")
- **Normalisation des récompenses** : 
  - Ajout de détails sur le clamping de l'écart-type [0.1, 0.5]
  - Ajout de clamping des valeurs normalisées à [-3, 3]
- **Pénalité de diversité** :
  - Formule mise à jour avec transformation logarithmique : `α · log(1 + 10 · freq_i)`
  - Coefficient α : 0.1 → **0.05** (réduit pour éviter sur-suppression)
  - Plafonnement maximum à 0.1
  - Ajout de la décroissance temporelle (facteur 0.99)

### 3. Méthode d'entraînement (Section 8)

**CHANGEMENT MAJEUR - Passage de RWR à Policy Gradient + InfoNCE:**

#### Ancienne approche (rapport original)
- **Reward-Weighted Regression** (RWR) simple avec pondération contrastive
- Catégorisation en positifs (R > 0.8), négatifs (R < 0.3), intermédiaires
- Poids : +1.0, -0.5, R - 0.5

#### Nouvelle approche (implémentation actuelle)
- **Policy Gradient (REINFORCE-style)** avec estimation d'avantage
- **InfoNCE contrastive loss** pour apprentissage contrastif approprié
- Séparation basée sur la médiane des récompenses du batch

**Formule de perte mise à jour:**
```
L_total = L_PG + 0.3·L_InfoNCE + 0.5·L_value + L_value-reg - α·H + β·Penalty_div
```

**Nouveaux composants détaillés:**
1. **Perte Policy Gradient (L_PG)** :
   - Utilise log-probabilité et avantage normalisé
   - REINFORCE avec baseline mobile
   
2. **Perte InfoNCE (L_InfoNCE)** :
   - Formule InfoNCE complète avec température τ = 0.1
   - Séparation positive/négative basée sur médiane
   
3. **Perte de valeur avec régularisation (L_value + L_value-reg)** :
   - MSE standard + terme de variance négative (-0.01 · Var)
   - Prévention de l'effondrement de la tête de valeur

4. **Baseline reward** :
   - Moyenne mobile exponentielle : `R_baseline ← 0.99·R_baseline + 0.01·R`
   - Utilisée pour calcul d'avantage

**Approximation de l'entropie mise à jour:**
```
H = std(prédiction) + 0.5 · mean(pdist(prédictions))
```

### 4. Replay Buffer

**Changements:**
- Capacité : 1000 → **5000** (augmentation pour meilleure stabilité)
- Nombre d'époques : 20 → **10** (réduit pour entraînement plus rapide)
- **Nouveau : Diversity-aware sampling**
  - Échantillonnage parmi 4× candidats
  - Sélection par maximisation de distance minimale
  - Meilleure couverture de l'espace des actions

### 5. Optimisation

**Changements confirmés:**
- Learning rate : **1e-4** (réduit de 5e-4)
- Batch size : **64**
- Gradient clipping : **max_norm=1.0**
- Scheduler cosinus : formule complète ajoutée avec η_max, η_min, T_max

### 6. Évaluation et checkpoints (Nouvelle section)

**Ajouts:**
- **Checkpoints automatiques** : tous les 100 phrases
  - Sauvegarde modèle + replay buffer
  - Compteur de phrases dans `sentence_counter.txt`
  
- **Évaluation régulière** : tous les 50 phrases
  - 10 premières phrases de validation
  - Sans exploration (ε = 0)
  - Mesure de performance réelle
  
- **Logging** : `training_progress.csv`
  - Numéro de phrase, CER, récompense, similarité cosinus

## Détails techniques manquants ajoutés

1. **Centroïdes** : précision sur le fichier `prompt_centroids.pt` et chargement automatique
2. **Écart-type du bruit local** : σ = 0.1 pour exploration autour des centroïdes
3. **Pondérations des composantes de perte** : coefficients explicites (1.0, 0.3, 0.5, etc.)
4. **Architecture de projection de valeur** : `512 → 256` avant la tête de valeur
5. **Formule du scheduler cosinus** : équation complète avec paramètres

## Justifications ajoutées

- Réduction du dropout : "pour améliorer le flux de gradient pendant les récompenses rares"
- Augmentation des blocs résiduels : "pour améliorer la capacité d'apprentissage de représentations"
- Réduction du learning rate : "pour améliorer la stabilité"
- Capacity buffer augmentée : "pour meilleure stabilité"
- Époques réduites : "pour un entraînement plus rapide tout en préservant la qualité"

## Structure améliorée

- Séparation claire entre Policy Gradient, InfoNCE, et composantes auxiliaires
- Explication détaillée de l'estimation d'avantage avec baseline
- Description du diversity-aware sampling dans replay buffer
- Ajout d'une sous-section sur sauvegarde et évaluation

## Corrections terminologiques

- "Bandit contextuel" → "Policy Gradient" dans le titre de la section 8
- Clarification que InfoNCE est la "bonne façon" de faire l'apprentissage contrastif
- Distinction claire entre apprentissage par imitation (attraction) et repoussement (répulsion)
