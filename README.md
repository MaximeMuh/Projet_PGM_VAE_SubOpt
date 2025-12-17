# Inference Suboptimality in Variational Autoencoders  
**Reproduction et extensions de Cremer et al. (2018)**

## Présentation générale

Ce projet porte sur l’**inférence amortie dans les Variational Autoencoders (VAE)** et plus précisément sur le phénomène d’**inférence sous-optimale**, comme le montre l'article : 

> Cremer, Li, Duvenaud — *Inference Suboptimality in Variational Autoencoders* (2018)

L’objectif principal est de **comprendre d’où viennent les erreurs d’inférence**, comment elles se décomposent, et comment elles évoluent en fonction :
- de la famille de posterior choisie,
- de la capacité de l’encodeur et du décodeur,
- de la dynamique d’apprentissage.

Le projet combine :
- une **réimplémentation quasi complète** des expériences clés du papier,
- des **extensions méthodologiques** (Contextual Flow),
- et des **nouvelles expériences** (analyse des gradients, refitting sous bruit).

C'est donc un **travail expérimental et exploratoire** autour des VAE.


## Rappel théorique : les gaps d’inférence

Pour un VAE entraîné avec une ELBO $\mathcal{L}[q_\phi]$, on a la décomposition :

$$
\log p_\theta(x) - \mathcal{L}[q_\phi]
=
\big(\log p_\theta(x) - \mathcal{L}[q^*]\big)
+
\big(\mathcal{L}[q^*] - \mathcal{L}[q_\phi]\big).
$$

- **Approximation gap** : $\log p_\theta(x) - \mathcal{L}[q^*]$
- **Amortization gap** : $\mathcal{L}[q^*] - \mathcal{L}[q_\phi]$

où :
- $q^*(z|x)$ est le meilleur posterior possible (optimisé localement pour chaque $x$),
- $q_\phi(z|x)$ est le posterior amorti produit par l’encodeur.

Le papier montre que **le gap d’amortization est souvent dominant**.

## Expérience 5.2 — Test-set inference gap

Cette expérience est reproduite sur **MNIST et Fashion-MNIST**.

On compare plusieurs modèles amortis :
- **FFG (Gaussian)** :  
  $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma_\phi(x)^2))$
- **Flow** : posterior plus flexible via des transformations inversibles
- **Contextual Flow (extension)** : flow conditionné explicitement par l’entrée

Pour chaque modèle, on calcule :
- $\log \hat p(x)$ (estimé par IWAE et AIS, on garde le maximum),
- $L[q]$ (ELBO amortie),
- $L[q^*]$ (ELBO optimisée localement),
- les trois gaps : approximation, amortization et total.

Résultat principal : le **gap d’amortization domine**, mais les flows réduisent significativement les deux termes, surtout sur Fashion-MNIST.


## Contextual Flow (extension)

Le **Contextual Flow** est une extension des normalizing flows standards.

Dans un flow classique, on a :
$
z = f_\lambda(z_0)
$

Dans un contextual flow, les paramètres du flow dépendent explicitement de l’entrée :
$
z = f_{\lambda(x)}(z_0)
$

Autrement dit, l’encodeur ne prédit pas seulement les paramètres d’une gaussienne, mais aussi les paramètres des transformations du flow.

Effets observés :
- posterior plus adaptable,
- meilleure réduction des gaps,
- comportement plus stable sur des données complexes.


## Expérience 5.3 — Décodeur figé, encodeurs petits

On reproduit l’expérience où le **décodeur est gelé** afin de fixer le vrai posterior $p_\theta(z|x)$.

Procédure :
1. On entraîne un VAE de base avec un posterior gaussien.
2. On gèle le décodeur.
3. On réentraîne des **encodeurs de petite taille** avec différentes familles :
   - Gaussian,
   - Flow,
   - Contextual Flow.

Cela permet d’isoler l’effet de la **famille variationnelle seule**.

Observation clé : les flows réduisent fortement le **gap d’amortization**, même avec peu de paramètres, ce qui montre qu’ils améliorent la **généralisation de l’inférence**, pas seulement l’expressivité.


## Expérience 5.5 — Dynamique d’apprentissage et généralisation

On suit l’évolution des gaps **pendant l’entraînement**, sur train et validation.

Modèles comparés :
- Gaussian standard,
- Flow,
- Contextual Flow,
- Gaussian avec **grand encodeur**,
- Gaussian avec **grand décodeur**.

Résultats :
- le gap d’approximation reste stable entre train et val,
- le gap d’amortization explose sur la validation pour les grands encodeurs,
- les flows offrent le meilleur compromis entre capacité et généralisation.


## Nouvelle expérience — Analyse des gradients

On introduit une expérience originale :  **suivre les normes de gradient de l’encodeur et du décodeur pendant l’entraînement**.

On mesure :
- $\|\nabla_{\text{enc}}\|$ et $\|\nabla_{\text{dec}}\|$,
- les mêmes quantités normalisées par $\sqrt{\text{nb paramètres}}$,
- leur ratio.

Résultats importants :
- les grands décodeurs déséquilibrent fortement l’optimisation,
- les grands encodeurs créent une forte pression d’apprentissage,
- les flows ont des gradients plus équilibrés,
- le contextual flow est le plus stable.

Cela donne une **lecture mécanique** du gap d’amortization.

## Nouvelle expérience — Encoder refitting sous bruit

Inspirée de Mattei & Frellsen (2018), cette expérience étudie le **refitting de l’encodeur** sur des données bruitées.

Procédure :
- on corrompt les images (10, 50, 100 pixels),
- on refit uniquement l’encodeur,
- on suit l’IWAE sur :
  - un jeu A (refit),
  - un jeu B indépendant.

Observations :
- l’IWAE augmente d’abord sur A **et** B → meilleure estimation du posterior,
- puis continue surtout sur A → adaptation locale,
- ce qui montre le compromis entre généralisation et sur-apprentissage.

## Organisation du projet

- Les fichiers `vae`, `distributions`, `optimize_local_q`, `generator` et `inference_net`  sont **fortement inspirés du code du papier**. En effet, l’objectif n’est pas de **réinventer la roue** : le VAE standard et ses briques mathématiques sont bien établis et ne constituent pas en eux-mêmes une contribution nouvelle. Cependant, on a 
  - nettoyé,
  - refactorisé,
  - amélioré quand c'est possible
  - adapté pour Contextual Flow qui necessite une nouvelle classe et des adaptations dans les fonctions (forward, etc)
  On a recodé entièrement les expériences 5.2, 5.3, 5.5 qui sont organisées sous forme de notebook. 
- Des notebooks dédiés couvrent :
  - l’analyse des gradients qui complètent l'expérience 5.5,
  - le refitting.

## Apports du projet

Ce travail apporte :
- une reproduction fidèle des expériences clés du papier,
- l’introduction du Contextual Flow,
- une analyse fine des dynamiques de gradient,
- des expériences nouvelles sur le refitting sous bruit.

Conclusion générale :  **le principal problème des VAE est l’amortization**, et améliorer l’expressivité de l’inférence est plus robuste que d’augmenter brutalement la taille des réseaux.