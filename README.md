# 📊 Segmentation des clients — Apprentissage non supervisé (DBSCAN)

Projet de **clustering de clients** d'une entreprise de télécommunications, réalisé dans le cadre du module *Unsupervised Learning* du **Sen HUB**. Trois algorithmes de clustering (KMeans, CAH, DBSCAN) sont comparés, le meilleur modèle est sélectionné puis **déployé sous forme d'application web interactive** (Streamlit) permettant d'attribuer une classe à un nouveau client à partir de ses caractéristiques.

## 🎯 Objectif

Identifier des groupes homogènes de clients à partir de leurs comportements d'usage (appels, SMS, ancienneté, valeur client, etc.), sans étiquette préalable, afin de :

- mieux comprendre les profils de clientèle,
- distinguer les clients fidèles des clients à risque de churn,
- détecter les clients au comportement atypique (anomalies),
- fournir un outil simple permettant de classer instantanément un nouveau client.

## 📁 Structure du dépôt

```
.
├── Customer_Segmentation.ipynb   # Notebook complet : EDA, PCA, modélisation, comparaison, déploiement
├── app.py                        # Application Streamlit de prédiction (modèle final : DBSCAN)
├── modele_dbscan.joblib          # Artefacts du modèle DBSCAN entraîné (points cœur, eps, colonnes, etc.)
├── requirements.txt              # Dépendances Python de l'application
└── README.md
```

## 🗂️ Jeu de données

Le jeu de données ("Télécommunications") contient **3 150 clients** décrits par **12 variables** numériques relatives à leur comportement d'usage :

| Variable | Description |
|---|---|
| `Call Failure` | Nombre d'appels échoués |
| `Complains` | Nombre de plaintes/réclamations |
| `Subscription Length` | Ancienneté de l'abonnement (mois) |
| `Charge Amount` | Montant total facturé |
| `Seconds of Use` | Temps total d'utilisation (secondes) |
| `Frequency of use` | Fréquence d'utilisation des services |
| `Frequency of SMS` | Fréquence d'envoi de SMS |
| `Distinct Called Numbers` | Nombre de numéros distincts appelés |
| `Age Group` | Catégorie d'âge (codée) |
| `Status` | Statut du client |
| `Age` | Âge du client |
| `Customer Value` | Score de valeur économique du client |

## 🧪 Méthodologie

1. **Prétraitement** : normalisation des observations (`normalize`, norme par ligne).
2. **Réduction de dimension** : ACP (PCA) à 2 composantes pour la visualisation — **97,5 %** de la variance expliquée conservée.
3. **Comparaison de trois algorithmes de clustering**, évalués par **score de silhouette** :

| Modèle | Paramètres | Score de silhouette |
|---|---|---|
| KMeans | `k=3` (choisi via la méthode du coude) | 0.785 |
| CAH (Classification Ascendante Hiérarchique) | `n_clusters=2`, `linkage='average'` | 0.885 |
| **DBSCAN** ✅ | `eps=0.2`, `min_samples=9` | **0.892** |

➡️ **DBSCAN** obtient le meilleur score de silhouette : c'est le modèle retenu pour le déploiement. Il présente aussi l'avantage de détecter automatiquement les **clients atypiques** (bruit / anomalies) plutôt que de les forcer dans un cluster.

## 📈 Résultats

Le clustering DBSCAN final identifie :

| Classe | Effectif | Interprétation |
|---|---|---|
| Classe 0 | 2 985 clients | **Client fidèle** — forte utilisation, forte valeur client |
| Classe 1 | 144 clients | **Client non fidèle** — faible utilisation, faible valeur client |
| -1 (bruit) | 21 clients | **Anomalie** — profil atypique, ne correspond à aucun groupe |

## 🚀 Déploiement

DBSCAN ne possède pas de méthode `predict()` native (contrairement à KMeans, il ne calcule pas de centres de clusters). La solution retenue s'appuie sur les **points cœur** (*core samples*) mémorisés par le modèle entraîné :

1. Le nouveau client est normalisé (même prétraitement qu'à l'entraînement).
2. On calcule sa distance euclidienne à tous les points cœur du modèle.
3. Si le point cœur le plus proche est à une distance ≤ `eps`, le client hérite de la classe de ce point cœur.
4. Sinon, il est classé comme **anomalie** (`-1`).

Cette logique est implémentée dans `app.py`. Le notebook contient également une version de démonstration avec **Gradio**, tandis que l'application déployée (GitHub + Streamlit) utilise **Streamlit**.

## ⚙️ Installation et utilisation

### Prérequis
- Python 3.9+

### Étapes

```bash
# 1. Cloner le dépôt
git clone <url-du-depot>
cd Sen_HUB_Non_Supervise_7_Segmentation_clients

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre dans le navigateur. Il suffit de choisir un préremplissage (valeurs médianes ou exemple), d'ajuster les caractéristiques du client, puis de cliquer sur **« Prédire la classe »** pour obtenir son segment.

## 🛠️ Technologies utilisées

- **Python** — pandas, numpy
- **Scikit-learn** — `PCA`, `KMeans`, `AgglomerativeClustering`, `DBSCAN`, `normalize`, `silhouette_score`
- **Scipy** — `linkage`, `dendrogram` (CAH)
- **Yellowbrick** — `KElbowVisualizer` (choix du nombre de clusters)
- **Plotly** — visualisations interactives (notebook)
- **Streamlit** — interface web de l'application déployée
- **Gradio** — prototype d'interface (notebook)
- **Joblib** — sérialisation du modèle

## 📌 Limites et pistes d'amélioration

- Le nombre de clusters/paramètres (`eps`, `min_samples`) a été fixé de façon empirique ; une recherche plus systématique (grid search sur la silhouette) pourrait affiner le résultat.
- Le jeu de données étant statique, le modèle n'est pas ré-entraîné automatiquement : toute dérive du comportement client nécessiterait un nouvel entraînement.
- L'interprétation des classes (« client fidèle » / « client non fidèle ») repose sur l'analyse du profil moyen par cluster et pourrait être enrichie avec des règles métier supplémentaires.

## 👤 Contexte

Projet réalisé dans le cadre du programme **Sen HUB** — module Apprentissage non supervisé / Clustering.
