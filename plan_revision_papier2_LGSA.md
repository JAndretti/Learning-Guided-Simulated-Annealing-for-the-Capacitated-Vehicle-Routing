# Plan de révision — LG-SA⁺ (papier 2)

Document de travail à activer en cas de refus AAAI-27, ou à utiliser pour préparer la rebuttal. Organisé par ordre de risque décroissant.

---

## A. Verdict

Le papier est **méthodologiquement au-dessus de la moyenne du domaine** (seed-pairing, noise floors, résultats négatifs publiés, contrôle blind-SA, benchmarks OOD avec optima prouvés). Ce n'est pas la rigueur qui le met en danger.

Ce qui le met en danger, c'est que **le cadrage promet ce que les tableaux ne montrent pas**. Trois revendications sont réfutables en dix secondes avec le Tableau 1 du papier lui-même. Un reviewer qui en vérifie une et la trouve fausse vérifie les deux autres, puis cesse d'accorder le bénéfice du doute au reste — y compris aux 40 pages d'annexes qui, elles, sont solides.

Mais deux découvertes postérieures à la première version de ce document déplacent le problème. Le papier ne souffre plus seulement d'un cadrage trop ambitieux : **sa revendication centrale est actuellement fausse**, et deux concurrents non cités le montrent.

- **Urgence 1** — un hybride CW+SA **non appris**, construit en une matinée, domine LG-SA⁺ sur toute la frontière qualité/temps (section A-bis). C'est l'objection la plus grave possible pour un papier dont la thèse est le compromis qualité/calcul.
- **Urgence 2** — Neural Deconstruction Search (2025), le voisin le plus proche possible du papier, n'est pas cité et bat HGS (section A-ter).

Le reste du document (B à F) reste valable, mais subordonné : corriger des revendications sur un résultat qui ne tient pas n'a pas de sens tant que A-bis n'est pas tranché.

---

## A-bis. Urgence 1 : le baseline CW+SA non appris

### Le résultat

Un solveur assemblé rapidement — Clarke–Wright, SA sur les trois voisinages, plusieurs points de départ CW, split optimal tous les 1 000 pas, biais vers les sommets mal placés, déplacements restreints aux 20 plus proches voisins — en C sur un CPU 12 cœurs :

| | Coût | Temps (10 000 inst.) | Gap HGS |
|---|---|---|---|
| CW+SA | **15.843** | **90 s** | 1.80 % |
| CW+SA | **15.67** (var. 15.70–15.76) | **1 000 s** | **0.69 %** |
| LG-SA⁺ $T{=}10^4$ | 16.183 | 180 s | 3.98 % |
| LG-SA⁺ $T{=}10^5$ | 15.923 | 1 794 s | 2.31 % |
| LG-SA⁺ $T{=}10^6$ | 15.805 | 17 280 s | 1.55 % |

Le point CW+SA à 1 000 s est **meilleur que le point LG-SA⁺ à 4,8 h, dix-sept fois plus vite**. À 0.69 % de HGS, il se placerait devant DACT (1.11 %) et près de NeuOpt $D{=}1$ (0.60 %) dans le Tableau 1, pour 0,1 s par instance sur CPU.

**Conséquence directe** : la revendication « les méthodes hybrides légères bien conçues atteignent un excellent compromis qualité/calcul » reste vraie, mais l'apprentissage n'est pas ce qui la produit. Le papier ne peut pas être soumis en l'état.

### Vérifications préalables (avant toute conclusion)

À faire en premier, parce que tout le reste en dépend :

1. **Même jeu de test ?** Les 10 000 instances Nazari du papier, pas un redraw.
2. **Même convention de coût ?** Euclidien exact, pas arrondi entier.
3. **Comptabilité du temps** identique (construction incluse ou exclue des deux côtés).
4. **Variance** : plusieurs runs, l'écart 15.67–15.76 signalé suggère une variabilité comparable à des effets qu'on discute ailleurs au centième.
5. **Comparabilité matérielle** : 12 cœurs CPU contre un GPU. Les deux sont « une machine de bureau », mais il faut une normalisation explicite, comme en D.3.

Si ces cinq points passent, le résultat tient et la suite s'applique.

### Ce qui sauve le papier, et le rend meilleur

Les trois ingrédients que ton directeur identifie comme décisifs — **CW, plusieurs voisinages, restriction aux plus proches voisins** — sont exactement les trois que LG-SA⁺ n'a pas :

| Ingrédient | CW+SA | LG-SA⁺ |
|---|---|---|
| Initialisation CW | oui | non (aléatoire, choisi à l'entraînement) |
| Plusieurs voisinages | 3 | 1 (insertion) |
| Restriction aux voisins proches | 20-NN | aucune (masque sur tous les partenaires faisables) |
| Split optimal périodique | oui | non |
| Départs multiples | oui | non (hors augmentation $A{=}8$) |

**La politique apprise n'a donc jamais été évaluée dans le régime fort.** La question du papier devient :

> **Que reste-t-il à l'apprentissage une fois la machinerie classique correctement construite ?**

C'est une question plus difficile, plus honnête et plus intéressante que celle du papier actuel, et les trois réponses possibles sont publiables :

1. **L'apprentissage ajoute beaucoup** → le papier devient nettement plus fort : guidage appris au-dessus d'une machinerie de niveau OR.
2. **L'apprentissage ajoute un peu** → papier honnête sur les rendements décroissants de l'apprentissage en OC, dans la lignée de Santana–Lodi–Vidal.
3. **L'apprentissage n'ajoute rien** → c'est le résultat « Possible Overkill ? » pour le SA sur CVRP. Publiable, utile, mais c'est un autre papier — et il vaut mieux l'écrire soi-même que se le faire écrire par un reviewer.

### A-bis.1 — L'expérience décisive : décomposition factorielle

C'est **la** expérience à faire, avant toutes les autres du document.

Plan factoriel sur quatre facteurs, à budget wall-clock apparié :

| Facteur | Niveaux |
|---|---|
| Initialisation | aléatoire / CW |
| Voisinages | insertion seule / insertion + 2-opt(*) |
| Restriction | aucune / $k$-NN ($k \in \{5, 10, 20\}$) |
| Proposition | uniforme / biais détour codé à la main / politique apprise |

$2 \times 2 \times 2 \times 3 = 24$ cellules, plus le balayage de $k$. À 3 min par run de 10 000 instances, c'est faisable en une journée GPU.

**Ce que ça produit** :
- l'effet principal de l'apprentissage **conditionnellement** au reste, qui est le chiffre que le papier doit rapporter ;
- les interactions — en particulier, la restriction $k$-NN rend-elle la politique apprise redondante, puisqu'elle encode déjà « bouger vers de bons voisins » ?
- une réponse anticipée à D.8 : le biais détour codé à la main *est* la règle de substitution, et ton directeur rapporte qu'il « aide un peu ». Il faut le chiffre exact.

**L'issue à surveiller** : si la politique apprise n'ajoute rien une fois la restriction $k$-NN active, l'explication probable est que ses features dominantes (détour, rangs de distance) encodaient surtout cette information — ce qui est un résultat, pas un échec, mais impose la réécriture n°3.

### A-bis.2 — Reconstruire la version forte de LG-SA⁺

Indépendamment du résultat, la méthode doit intégrer les ingrédients gagnants : CW, 2-opt(*) en plus de l'insertion, restriction $k$-NN, split optimal périodique, départs multiples. Puis **réentraîner** — la politique actuelle a été optimisée pour un régime qui n'existera plus.

Les points techniques correspondants sont détaillés en section G.

---

## A-ter. Urgence 2 : traiter NDS

**Référence** : Hottung, Wong-Chung, Tierney, *Neural Deconstruction Search for Vehicle Routing Problems*, TMLR 05/2025 (arXiv 2501.03715). Code public : `github.com/ahottung/NDS`.

### Ce que c'est

Une politique neuronale apprise **déconstruit** la solution courante (sélection séquentielle des clients à retirer), une insertion gloutonne la reconstruit, le tout à l'intérieur d'un *augmented simulated annealing* batché sur GPU n'utilisant qu'un seul cœur CPU. C'est-à-dire : **un composant appris à l'intérieur d'un recuit simulé parallélisé sur GPU** — le cadre exact de LG-SA⁺.

Résultats CVRP à budget wall-clock égal, gap relatif à HGS :

| | HGS | NDS |
|---|---|---|
| N=100 | 15.57 | 15.57 (0.04 %) |
| N=500 | 36.66 | 36.57 (**−0.20 %**) |
| N=1000 | 41.51 | 41.11 (**−0.90 %**) |
| N=2000 | 57.38 | 56.00 (**−2.34 %**) |

Les auteurs revendiquent la première méthode apprise à égaler ou dépasser les méthodes OR de l'état de l'art. NDS traite aussi VRPTW et PCVRP, et bat SISRs sur presque tous les régimes de temps.

### Ce que ça casse dans le papier

1. **Intro** — l'affirmation que les méthodes apprises ont un coût structurel les empêchant de rivaliser avec les métaheuristiques est réfutée depuis 2025. À réécrire.
2. **Related Works** — le paragraphe « learning-guided metaheuristics » présente cette voie comme peu explorée. NDS l'occupe et y gagne. Le paragraphe doit être refondu autour de NDS comme référence principale de la famille.
3. **Tableau 1** — NDS doit y figurer. Un reviewer PRS d'AAAI le verra.
4. **Conclusion** — la piste « augmenter HGS avec des composants appris » est à reformuler : NDS montre que la voie fonctionne, ce n'est plus une spéculation.

### Ce que ça ne casse pas — et qu'il faut revendiquer explicitement

- **Régime batché.** NDS traite les instances séquentiellement, 5 s par instance à N=100, soit ~14 h pour 10 000 instances. LG-SA⁺ est à 18 ms par instance. Le régime batché à très faible budget reste vide de concurrents, y compris de NDS.
- **Coût d'entraînement.** NDS est un transformer (128 dims, couches d'attention, décodeur GRU) entraîné **5 à 15 jours sur A100** selon la taille, avec un modèle par taille d'instance. LG-SA⁺ : deux MLP de 32 neurones, ~1,2 h sur une carte grand public, un seul modèle pour toutes les tailles.

Le positionnement correct devient donc : **NDS et LG-SA⁺ sont aux deux extrémités du même spectre.** NDS achète la qualité maximale au prix d'un entraînement massif et d'une exécution par instance ; LG-SA⁺ achète le débit et la quasi-gratuité de l'entraînement au prix de la qualité asymptotique. Le dire soi-même est infiniment plus solide que de laisser un reviewer le découvrir.

### Bonus : NDS écrit tes travaux futurs

Les deux limitations que les auteurs déclarent eux-mêmes sont exploitables :

- *dépendance au GPU pour exécuter le réseau ; explorer la distillation pour réduire le coût* → LG-SA⁺ est déjà une réponse partielle, à souligner.
- *dépendance à des données d'entraînement proches des instances de test ; explorer si le fine-tuning permet une adaptation rapide* → c'est l'expérience D.13 ci-dessous. À noter : le groupe leader a annoncé cette direction, donc si elle est menée, il faut le faire vite et ne pas en dépendre.

### Actions concrètes

| # | Action | Coût |
|---|---|---|
| A-ter.1 | Lire NDS en entier, y compris annexes E (hyperparamètres) et G (généralisation) | 1 jour |
| A-ter.2 | Réécrire intro + Related Works autour de NDS | 2 jours |
| A-ter.3 | Faire tourner NDS (code public) sur tes jeux de test, en régime batché **et** séquentiel, sur ta machine | 3–5 jours |
| A-ter.4 | Ajouter la ligne NDS au Tableau 1 et le paragraphe de positionnement à deux extrémités | 1 jour |

A-ter.3 fait d'une pierre deux coups : c'est aussi le baseline réexécuté sur ton matériel demandé en D.2, et il est plus pertinent que NLNS ou LIH puisque c'est le concurrent réel.

---

## B. Erreurs de revendication (à corriger quoi qu'il arrive)

### B.1 — « dominates LIH and NLNS » (Intro, contribution 4)

**Faux pour NLNS.** Vérification sur ton propre tableau :

| | NLNS ($T{=}5$k) | LG-SA⁺ ($T{=}10^5$) | Verdict |
|---|---|---|---|
| $N{=}20$ | 6.175 / 48 m | 6.182 / 3.5 m | NLNS meilleur, toi 13.7× plus rapide |
| $N{=}50$ | 10.506 / 1.4 h | 10.526 / 14.6 m | NLNS meilleur, toi 5.7× plus rapide |
| $N{=}100$ | 15.915 / 2.4 h | 15.923 / 29.9 m | NLNS meilleur, toi 4.8× plus rapide |

NLNS est meilleur en objectif **aux trois tailles**. « Outperforming » (abstract) et « dominates » (intro) ne sont soutenus nulle part sur la qualité.

**Vrai pour LIH** : 16.165 en 5 h contre 15.923 en 29.9 min — meilleure qualité *et* 10× plus rapide. Domination stricte, à conserver telle quelle.

**Formulation correcte** : « LG-SA⁺ strictly dominates LIH, and matches NLNS within 0.05 % at 5–14× less computation. »

### B.2 — « state-of-the-art performance » (Conclusion)

NeuOpt ($D{=}5$) et POMO+EAS+SGBS sont à 0.10 %. Ton meilleur point est 1.55 %. Il n'y a pas de lecture sous laquelle c'est du SOTA en qualité.

**Formulation correcte** : « a state-of-the-art quality/runtime trade-off in the low-budget regime ».

### B.3 — La revendication Pareto est fausse à l'extrémité haute

C'est l'erreur que je n'avais pas vue à la première lecture et c'est la plus dangereuse, parce qu'elle figure dans ton texte commenté (« at equal wall clock, none of them reaches its quality ») et risque de revenir en rédaction :

- NeuOpt ($D{=}1$, $T{=}10^4$) : **15.656 en 4.6 h**
- LG-SA⁺ ($T{=}10^6$) : **15.805 en 4.8 h**

À budget quasi identique, NeuOpt te domine. Tu n'es **pas** sur la frontière de Pareto au-delà de ~4 h.

**Mais** — et c'est le point à vendre — en dessous de ~2 h à $N{=}100$, il n'existe **aucune autre méthode dans le tableau**. Le concurrent le plus rapide (AM+LCP) est à 2.1 h pour 16.00 ; toi tu es à 16.183 en **3 minutes** et 15.923 en 30 minutes.

**Revendication à substituer, imprenable** : *LG-SA⁺ occupe seul le régime sous les deux heures ; au-delà de quatre heures, les méthodes lourdes reprennent l'avantage.* C'est plus étroit et beaucoup plus fort, parce qu'aucun reviewer ne peut le réfuter.

### B.4 — « comparable quality to DACT »

15.930 (2.36 %) contre 15.736 (1.11 %) : le gap double. Ton ancienne formulation commentée était juste — « concedes 0.19 units to DACT at ~100× less compute ». La reprendre.

### B.5 — « the leading learning-based improvement methods NLNS and LIH » (abstract)

NLNS et LIH ne sont pas les meilleures méthodes d'improvement de ton propre tableau : NeuOpt et DACT le sont. Sélectionner les deux baselines les plus faibles et les qualifier de « leading » est le type de glissement qu'un reviewer relève et qui coûte cher en crédibilité. Écrire « two widely used improvement baselines ».

### B.6 — Le gain end-to-end $17.36 \to 16.18$ est cross-draw

Tu le marques honnêtement comme indicatif dans la légende, mais c'est **le chiffre de l'abstract et de l'intro**. Une headline ne peut pas reposer sur une comparaison que le papier qualifie lui-même de non appariée.

**Correctif obligatoire** : refaire tourner l'ancien checkpoint LG-SA sur les jeux de test actuels. Si le checkpoint est perdu, réimplémenter — c'est deux MLP.

### B.7 — Incohérences numériques

- abstract 2.30 % / table 2.31 %
- abstract « 29 minutes » / table 29.9 min
- intro « from 7.95 % to 2.30 % » / table 2.31 %

Triviales isolément, mais elles suggèrent que le reste n'a pas été vérifié non plus.

### B.8 — Anonymat

« Our work substantially extends this framework » immédiatement après avoir cité `AndrettiEtAl26` te désigne. Neutraliser : « We build on this framework and extend it along three axes… ».

### B.9 — Portée de la généralisation surrevendiquée

L'abstract dit « transfers without retraining to instances with up to 10,000 customers and to different distributions ». À $N{=}10^4$ (Set XL) le gap est de **40 %**. C'est un résultat honnête et défendable *avec son contrôle* (blind SA à 675 %), mais « transfers » sans qualificatif laisse entendre une qualité utilisable. Écrire « remains feasible and steadily improving up to 10,000 customers, though the gap to best-known solutions grows to 40 % at that scale ».

---

## C. Problèmes méthodologiques

### C.1 — La comparaison temporelle n'est pas ancrée (risque de rejet à lui seul)

RTX 5070 (Blackwell, 2025) contre des baselines mesurées par \citet{ma_learning_2023} sur du matériel d'au moins deux générations antérieures. Le facteur « 5–10× moins de calcul » contient un facteur GPU inconnu, plausiblement 1.5–3×.

Le papier pose le caveat, puis l'abstract et l'intro s'appuient sur les chiffres sans réserve. C'est la configuration exacte qui irrite un reviewer : le papier sait, et revendique quand même.

### C.2 — Le Tableau 1 abandonne le protocole statistique du papier

Tu construis 40 pages sur le seed-pairing, puis la table principale rapporte **le meilleur checkpoint sur 5 seeds, sans écart-type**. Par ailleurs, « its best checkpoint provides our results » est ambigu : sélection sur validation (légitime) ou sur test (invalide) ? À clarifier explicitement.

### C.3 — Tu jettes 10 000 instances appariées

C'est le gaspillage le plus coûteux du papier. Tu justifies l'absence de test statistique par $n = 3$ seeds — mais **chaque comparaison porte sur 10 000 instances évaluées par les deux configurations**. Un bootstrap apparié au niveau instance te donne des intervalles de confiance minuscules, gratuitement, et transforme chaque « tie » en résultat définitif plutôt qu'en non-conclusion.

Le noise floor reste utile pour la variabilité inter-seed. Mais il faut rapporter les deux niveaux : variance entre seeds *et* IC apparié entre instances.

### C.4 — Les seeds sont bon marché et tu n'en profites pas

Tes runs coûtent 0.2–4.5 h (médiane 1.2 h). Passer de 3–5 à 15 seeds sur les décisions litigieuses coûte ~18 h GPU. Trois décisions le méritent :

- **curriculum** : $\Delta = -0.073 \pm 0.043$, 4/5 seeds, IC $[-0.027, +0.105]$ contenant zéro — et pourtant listé comme contribution ;
- **critic multi-statistique** : $+0.06$, marginal ;
- **PPO passes 5 vs 10** : sign-split.

Avec 15 seeds, ces trois verdicts deviennent nets. Sans, un reviewer peut légitimement écrire que la contribution « curriculum » n'est pas établie.

### C.5 — Plateau de capacité surrevendiqué

Balayage jusqu'à $64 \times 2$ seulement, et la conclusion concède ensuite que des encodeurs GNN légers pourraient aider — ce qui contredit « increasing model capacity brings none ». Soit étendre ($128\times2$, $256\times3$), soit écrire « within the tested range ».

### C.6 — Validation séquentielle : pas de vérification d'optimalité locale

Tu documentes honnêtement la limite, et tu documentes même une interaction ratée (critic LR × rollout × $T_{\text{final}}$ → divergence). C'est précisément ce qui rend légitime la question : la configuration finale est-elle seulement un optimum local du chemin de descente choisi ?

### C.7 — Toutes les décisions de design sont prises à $N{=}100$

Tu testes le transfert du *checkpoint*, jamais celui des *décisions*. Est-ce que $T_{\text{final}} = 0.01$, le curriculum, le jeu de features restent optimaux à $N{=}500$ ? Question inévitable en review.

### C.8 — L'explication du résultat négatif « pair descriptors » est spéculative

Tu affirmes que la limite est l'optimisation et non l'information. Plausible, mais non testé. Une seule expérience (le même variant entraîné avec largeur 64 ou budget doublé) tranche.

### C.9 — Protocole incohérent dans `tab:dimsfull`

Cette table passe en CPU, batch 1, `float32`, alors que tout le reste est GPU. Les ms/step ne sont donc pas comparables au reste du papier.

### C.10 — Benchmarks OOD en run unique

XML100, Set X, Set XL : un seul run chacun, aucune variance. Set X ne fait que 100 instances — 5 runs coûtent quelques minutes.

---

## D. Expériences à ajouter, par priorité

### Tier 1 — décident l'acceptation

**D.1. Baselines SA structurés (la plus urgente).**
« Blind SA » propose uniformément. Personne ne fait ça. Trois contrôles à ajouter, tous à coût quasi nul puisqu'ils réutilisent ta boucle :

- SA restreint aux paires (i, j) avec j parmi les 5 plus proches voisins de i — c'est la *candidate list* de LKH-3, la référence évidente ;
- SA avec probabilité de sélection du premier nœud proportionnelle au détour $\Delta_i$ ;
- SA combinant les deux (roulette sur $\Delta_i$ × rang de distance).

**Enjeu** : si LG-SA⁺ ne bat que l'uniforme, la contribution s'effondre. S'il bat ces trois-là, la contribution devient incontestable. Dans les deux cas il faut le savoir avant le reviewer. C'est aussi la comparaison la plus intéressante scientifiquement, puisque tes features dominantes (détour, rangs de voisinage) sont exactement ce que ces heuristiques encodent en dur.

**D.2. Un baseline neuronal réexécuté sur ta machine.**
NLNS ou LIH, checkpoints publics. Un seul point d'ancrage calibre toute la colonne temps et transforme le caveat C.1 en argument.

**D.3. Courbes anytime des solveurs classiques à budget par instance apparié.**
HGS reçoit 2.5 jours pour 10 000 instances, soit ~21.6 s par instance. Toi tu es à **18 ms par instance**. Personne dans le domaine ne trace HGS, LKH-3 et OR-Tools à 18 ms, 100 ms, 1 s, 10 s par instance.

C'est l'expérience qui te sert le plus, elle est bon marché (PyVRP accepte une limite de temps), et elle règle le problème de cadrage : au lieu de rapporter un gap à un oracle qui a eu 1200× ton budget, tu montres une frontière anytime complète où ton régime est vide de concurrents. Cette figure peut devenir la figure 1 du papier.

**D.4. Vérification d'optimalité locale du design.**
Re-balayer, **à la configuration finale**, les 4 dimensions les plus influentes (features, $T_{\text{final}}$, taux actor, curriculum). Si aucune ne bouge, tu réponds définitivement à C.6 pour ~20 runs.

**D.5. Bootstrap apparié au niveau instance** sur toutes les comparaisons headline (C.3). Coût : nul, c'est du post-traitement des CSV existants.

**D.6. Ancien LG-SA sur le jeu de test actuel** (B.6). Obligatoire.

### Tier 2 — font passer d'« accept » à « strong accept »

**D.7. Transfert des décisions de design.**
Re-balayer les 3 dimensions dominantes à $N{=}500$. Deux issues, toutes deux publiables : soit les décisions transfèrent (résultat fort, tu peux le dire), soit non (résultat plus intéressant encore, et c'est directement le papier 3).

**D.8. Extraction d'une règle de substitution.**
Ajuster un modèle interprétable (régression logistique ou arbre peu profond) sur les 23 features pour imiter la distribution d'actions de la politique, puis faire tourner SA avec cette règle.

- Si une règle à 3 paramètres récupère 90 % du gain : résultat marquant, honnête, et parfaitement aligné avec ta thèse « la capacité n'est pas la contrainte ».
- Sinon : justification directe de la politique apprise.

Combinée à D.1, cette expérience répond à la question que tout reviewer se pose sans l'écrire : *qu'est-ce que le réseau a appris que je ne pourrais pas coder à la main ?*

**D.9. Balayage de capacité étendu** ($128\times2$, $256\times3$, et un mini-GNN à 1 couche) pour établir le plateau proprement (C.5).

**D.10. Variance multi-run sur Set X et XML100** (C.10).

**D.11. Test de l'hypothèse « optimisation, pas information »** sur les pair descriptors (C.8).

### Tier 3 — s'il reste du temps

**D.13. Adaptation en ligne : est-ce que l'adaptation remplace le réentraînement ?**

*Expérience à faire seulement si les Tiers 1 et 2 sont bouclés — mais elle a une valeur double, décrite plus bas.*

**L'hypothèse.** Un modèle à 32 neurones peut être réentraîné **pendant** qu'il résout l'instance, sur les récompenses que le recuit produit déjà. Contrairement aux méthodes constructives, où l'active search exige des rollouts dédiés (c'est pourquoi EAS coûte des heures), ici la recherche *est* la collecte de données : chaque pas de recuit produit déjà un tuple (état, action, récompense) et le calcul de PPO est marginal.

**La question qui vaut le coup.** Pas « est-ce que ça améliore un peu », mais : **l'adaptation en ligne referme-t-elle l'écart du réentraînement ?**

Tu as déjà les tableaux de contrôle. Dans `tab:crossdim`, un modèle entraîné à N=100 évalué à N=200 donne 29.84 contre 30.01 pour un modèle entraîné à N=200 ; dans `tab:crossdist`, le modèle Nazari testé sur Uchoa donne 19.29 contre 19.11 pour le modèle Uchoa. Ces écarts sont les cibles. Si l'adaptation en ligne les annule **sans réentraînement et sans surcoût mesurable**, c'est un résultat qui se défend seul.

**Protocole minimal (~2 semaines).**
1. Reprendre les checkpoints cross-dim et cross-dist existants.
2. Activer la mise à jour PPO pendant l'inférence, sur la trajectoire de recuit elle-même, toute la (petite) politique, sans paramètre ajouté.
3. Mesurer : coût final, surcoût wall-clock, et l'écart résiduel aux modèles spécialisés.
4. Contrôles indispensables : adaptation désactivée, adaptation avec taux d'apprentissage nul, et une instance unique répétée (pour distinguer adaptation réelle et bruit).

**Valeur double.**
- *Pour le papier 2* : une section courte qui renforce directement la partie généralisation, et qui répond par anticipation à la limitation n°2 de NDS.
- *Pour le papier 3* : c'est le test décisif d'une des directions envisagées. S'il est positif, la direction devient viable ; s'il est négatif, elle se ferme pour deux semaines de travail au lieu d'un an.

**Antériorité à citer impérativement.** LRBS (Camerota Verdù et al., AAAI 2025, arXiv 2412.10163) fait de l'adaptation en ligne d'une politique d'improvement, sur TSP et variantes pickup-and-delivery, via des rollouts de beam search et des poids ajoutés façon EAS. Ta différenciation, si elle tient, est : mise à jour **totale** du modèle, **continue**, **sans rollout dédié ni paramètre ajouté** — ce que seule la petitesse rend possible. Il faut le formuler ainsi dès le début, pas se le faire dire en review.

**D.12. Lois d'échelle budget/taille.** Ajuster $T^*(N, \text{gap cible})$ à partir de `tab:stepbudget` et `tab:dimsfull`, puis **prédire** sur des couples $(N, \text{budget})$ non vus et vérifier. Tu as déjà toutes les données pour l'ajustement ; il manque la validation prédictive.

C'est aussi la première brique du papier 3, donc le travail n'est jamais perdu.

---

## E. Restructuration du texte

### E.1 — Faire remonter la thèse

Le papier a une vraie thèse scientifique, et elle est enterrée sous le design study :

> **À $N{=}100$, la capacité du modèle n'est pas la contrainte active. Ce qui l'est, c'est ce que la politique observe et la manière dont elle est entraînée.**

Tu la démontres avec des preuves rares et propres : encodeur partagé $+3.20$, contexte global $+0.51$, pair descriptors $+0.15$, plateau largeur×profondeur, alors que l'enrichissement de l'état vaut $3.41$ unités.

Reformuler les contributions autour de cette affirmation. Le design study cesse alors d'être une liste de réglages (lecture « ingénierie ») pour devenir la *méthode de preuve* d'une thèse (lecture « science »). C'est le même contenu, et ça change complètement la façon dont un reviewer le classe.

### E.2 — `tab:summary` induit en erreur

La colonne « Effect » mélange gains obtenus et pertes évitées. La légende le dit, mais un reviewer qui survole lit « +3.20 grâce aux scorers indépendants ». Scinder en deux colonnes : *gain à l'adoption* / *perte évitée par rejet de l'alternative*.

### E.3 — Promouvoir l'analyse anytime en section principale

`fig:frontier`, `fig:multistart`, `tab:stepbudget`, le paragraphe restart-vs-anneal et l'observation « the currency is steps per node, not steps » sont actuellement en annexe. C'est le matériau le plus original du papier et celui qui parle au-delà de la communauté CVRP. Avec D.3, ça fait une section « Where the compute goes » qui vaut mieux que la moitié du design study.

### E.4 — Recadrer les tableaux OOD

Mettre LG-SA⁺ à 40 % à côté d'AILS-II à 0.07 % invite la mauvaise lecture. Faire du contrôle (blind SA à 675 %) la référence primaire, et reléguer les métaheuristiques dans une colonne de contexte avec leur budget affiché.

---

## F. Bibliographie

Hors ton ICAPS, `zhu_refining_2025` et `queiroga_xl_2026`, il n'y a essentiellement rien de 2025–2026. Pour AAAI-27 c'est visible. À ajouter au minimum :

- **NDS** (Hottung et al., TMLR 2025) — voir section A-bis, c'est la citation manquante la plus grave.
- **LRBS** (Camerota Verdù et al., AAAI 2025) — adaptation en ligne de politiques d'improvement.
- **La famille active search** : EAS (ICLR 2022), COMPASS (NeurIPS 2023), Poppy, PolyNet (ICLR 2025), MEMENTO (2024). Pertinente parce que c'est la réponse standard du domaine au problème de généralisation que traite ta section scaling.
- **SISRs** (Christiaens & Vanden Berghe, 2020) — absent alors que c'est une des métaheuristiques de référence sur grandes instances, et le baseline principal de NDS.

- **Modèles généralistes / foundation** : RouteFinder, GOAL, BQ-NCO, MVMoE. Pertinents parce qu'ils incarnent exactement la thèse « plus de capacité » que tu réfutes — c'est ton opposition naturelle, pas un remplissage.
- **Large-scale** : GLOP, UDC, décomposition neuronale. Concurrents directs de ta section scaling.
- **« Rethinking light decoder-based solvers for VRPs »** (Huang et al., 2025) — frontalement dans ton sujet.
- **LLM-AHD** : EoH, ReEvo, MCTS-AHD, une phrase pour situer le learning-guided par rapport à eux.
- Les entrées ICLR-2026 pertinentes (RRNCO, contraintes, multi-tâches).

Un paragraphe suffit. C'est l'absence qui signale « papier écrit il y a deux ans ».

---

## G. Retour du directeur — points techniques

Chaque point est rattaché à sa priorité. Ceux marqués **[fort]** sont ceux dont l'expérience CW+SA indique qu'ils comptent beaucoup.

### G.1 — Restriction aux plus proches voisins **[fort, Tier 1]**

Limiter les relocations aux $k$ plus proches voisins. Confirmé comme un des trois gains majeurs du CW+SA.

Deux effets à séparer proprement : le **gain de vitesse** (le masque devient $O(k)$ au lieu de $O(N)$, et le second étage de la politique ne score plus que $k$ candidats) et le **gain de qualité** (l'espace de recherche exclut des mouvements presque toujours mauvais). Balayer $k \in \{5, 10, 20, 50, \infty\}$, en séparant les deux axes.

Enjeu conceptuel à traiter explicitement : c'est la *candidate list* classique, celle de LKH-3. Si elle suffit, une partie de ce que la politique apprenait devient redondante — c'est le cœur de A-bis.1.

### G.2 — Plusieurs voisinages, 2-opt(*) **[fort, Tier 1]**

Deuxième gain majeur du CW+SA. L'argument de ton directeur sur SWAP est juste : un échange est décomposable en deux relocations, donc l'ajouter apporte peu à un SA qui fait déjà des relocations. 2-opt(*) est en revanche structurellement différent, il modifie la topologie des routes.

Le papier avait écarté 2-opt(*) dans `tab:v1grid` (18.93 contre 16.68) — mais l'annexe `app:v1` reconnaît honnêtement que **la représentation favorise l'insertion** : le détour $\Delta_i$ est exactement la moitié d'une relocation, tandis que le gain marginal d'un 2-opt dépend de quatre arêtes qu'aucune feature par nœud n'encode.

Donc : ajouter une feature de gain marginal spécifique à 2-opt(*) avant de conclure. La comparaison actuelle porte sur des couples (opérateur, représentation), pas sur les opérateurs. Et le point est de **mélanger**, pas de choisir : politique à deux niveaux, choix de l'opérateur puis de ses arguments.

### G.3 — Entraîner sur des solutions CW **[fort, Tier 1]**

**Trou réel dans le design study.** `tab:v8grid` croise construction d'entraînement × construction d'inférence, mais les lignes sont : aléatoire, sweep, plus proche voisin, mono-client, mixte. **CW n'apparaît qu'en colonne, jamais en ligne.** La construction la plus performante n'a donc jamais été testée à l'entraînement.

Cela invalide partiellement la conclusion « random est la meilleure construction d'entraînement », qui n'a été établie que contre des constructions plus faibles. À refaire, notamment la cellule CW→CW.

Mécanisme à surveiller : le papier explique le succès de random par la couverture (états les plus éloignés d'un optimum local, donc les plus divers). Mais la section curriculum montre l'effet inverse — les états trop pré-optimisés dégradent l'apprentissage. CW construit à 16.5 pour un final à 15.9 : c'est du pré-optimisé. Les deux mécanismes s'opposent, et seule l'expérience tranche.

### G.4 — Features de densité : $\mu_5$ contre $\mu_{N/10}$, $\mu_{N/3}$ **[Tier 2]**

Ton directeur a raison, et le papier le sait à moitié : le paragraphe *« Limitation: the mixed scale has a cost »*, actuellement **commenté** dans le source LaTeX, dit exactement ça. Il faut le décommenter et le traiter.

Le problème précis : $\mu_5$ (compte absolu) n'est pas invariant en $N$ — pour des clients uniformes, la distance moyenne aux $k$ plus proches décroît en $O(N^{-1/2})$, donc à $N = 10^4$ la feature vaut un ordre de grandeur de moins que ce que le réseau a vu à l'entraînement. Les deux autres ($N/10$, $N/3$) sont proportionnelles et restent calibrées. **C'est un candidat sérieux pour les « effets zarbs » à grande dimension.**

À tester : (a) tout proportionnel, (b) tout absolu, (c) le mixte actuel, (d) $\mu_5$ renormalisée par $\sqrt{N}$ — et surtout **avec réentraînement**, parce que redéfinir une feature à l'inférence seule est un décalage de distribution. Le papier note d'ailleurs que le patch sans réentraînement dégrade (12.71 % → 13.08 % sur Set X).

Second défaut à corriger : les comptes $N/10$ et $N/3$ utilisent le plus petit $N$ du batch d'évaluation, pas celui de chaque instance. Sur un batch de tailles mixtes, les features d'une instance dépendent des instances avec lesquelles elle est batchée. C'est un bug, sans justification.

### G.5 — Supprimer la feature de charge inutile **[Tier 1, gratuit]**

`route_pct` ($L_{r(i)}$). Le papier le mesure déjà comme le seul retrait neutre en qualité ($-0.035 \pm 0.106$, et $\Delta_{\mathrm{LOO}} = -0.06$). À enlever : une feature de moins, un argument de parcimonie de plus, aucun coût.

Attention à ne pas généraliser : l'ablation thématique montre que retirer **les trois** signaux de capacité coûte $+0.343$. Ils sont mutuellement substituables ; on en enlève un, pas le groupe.

### G.6 — Centroïdes sans le dépôt **[Tier 2, très bon marché]**

Actuellement le centroïde de route inclut le dépôt, que **toutes** les routes partagent. Il tire donc tous les centroïdes vers un point commun et réduit le pouvoir discriminant de $d_{i,\mathrm{cent}}$ — précisément l'une des deux features les plus utiles du papier ($\Delta_{\mathrm{iso}} = +0.48$).

Une ligne de code, un réentraînement. À faire aussi : variante pondérée par la demande.

### G.7 — Structure du réseau **[Tier 2]**

Ton directeur observe ce que `tab:v3arch` montre : la tendance est monotone en faveur des réseaux plus larges ($64\times2$ : $-0.049$ ; $64\times1$ : $-0.047$ ; $32\times2$ : $-0.037$), sans qu'aucun ne franchisse le noise floor de $0.053$. Le signal est faible mais **jamais de signe contraire**.

Trois seeds ne suffisent pas à trancher ça. Deux corrections : (a) porter à 15 seeds sur $32\times1$ contre $64\times2$ — le coût est dérisoire ; (b) étendre à $128\times2$ et $256\times3$ pour savoir si le plateau existe vraiment.

En attendant, la revendication « increasing model capacity brings none » doit devenir « within the tested range, no configuration cleared the noise floor » (cf. C.5).

### G.8 — Implémentation GPU et compilation **[Tier 1]**

Le point est juste et il est stratégique : quand toute la thèse du papier est un compromis temps/qualité, laisser un facteur 2 à 4 dans l'implémentation est indéfendable — surtout maintenant qu'un CPU 12 cœurs bat la méthode.

À faire :
- **`torch.compile`** en mode `reduce-overhead` avec CUDA graphs. Le gain vient du nombre d'appels de kernels : une boucle de recuit qui enchaîne des dizaines de petites opérations par pas est dominée par le lancement, pas par le calcul. C'est cohérent avec `tab:batch`, où le débit plafonne dès 4 000 instances.
- **Profiler** (`torch.profiler`) avant d'optimiser, pour savoir où passe réellement le temps : features, masque de faisabilité, forward, acceptation.
- **Éliminer les redondances** accumulées : le papier a été construit en ajoutant des briques rapidement, certaines features sont probablement recalculées à chaque pas alors qu'elles ne changent que localement (topologie, statistiques de route).
- **Cas limites** : petit/grand $N$, faible/forte charge. C'est le meilleur détecteur de bugs silencieux et d'écarts entre ce qui est décrit et ce qui est implémenté.

Un facteur 2 déplace tous les points du Tableau 1 vers la gauche, et c'est le seul travail du document qui améliore *toutes* les revendications à la fois.

---

### G.9 — Features d'état de recherche : stagnation et écart à l'incumbent **[Tier 1, bon marché]**

**L'idée.** Deux scalaires supplémentaires : le nombre de pas depuis la dernière amélioration de l'incumbent, et l'écart entre la solution courante et l'incumbent.

**Pourquoi c'est qualitativement nouveau.** Les méta-features actuelles (température normalisée, fraction de budget restante) sont **en boucle ouverte** : elles ne dépendent que du compteur de pas. Ces deux-là sont les premières à être **en boucle fermée** — elles décrivent l'état de la recherche, pas la position dans le schedule. C'est un changement de nature, pas un ajout de dimension.

Elles encodent exactement l'information que les métaheuristiques adaptatives classiques utilisent explicitement : réchauffage déclenché par stagnation, *record-to-record travel* (accepter si l'on reste dans une déviation du record), *great deluge*, *late acceptance hill climbing*, et les mécanismes de stagnation de FILO. Un bon signe : ce n'est pas une intuition isolée, c'est un mécanisme dont l'efficacité est établie sous forme codée à la main.

**Normalisation — le piège principal.** L'écart doit être **relatif** : $(C(s_t) - B_t)/B_t$, pas absolu, sinon il n'est pas invariant en taille ni en échelle d'instance.

Le compteur de stagnation est plus délicat, et pour une raison que le papier connaît déjà. Les rollouts d'entraînement font $L = 200$ pas ; à l'inférence, la stagnation peut atteindre des milliers de pas. **C'est exactement le problème d'échelle de $\mu_5$** (G.4) : le réseau lirait au test une plage jamais vue à l'entraînement. Trois options à comparer : fraction du budget restant, échelle logarithmique, ou plafonnement. À décider par l'expérience, pas par défaut.

**Conséquences sur le MDP.** Ces features rendent l'état **non markovien par rapport à la solution seule** — il dépend de l'historique. Ce n'est pas un problème formel (l'état augmenté reste un MDP), mais trois choses en découlent :
- le **critique doit les voir aussi**, sinon les estimations de valeur sont incohérentes avec ce que voit l'acteur ;
- le **curriculum** interagit : après un warm-up de 250 pas, le compteur de stagnation est hérité. Décider explicitement s'il est remis à zéro au début de la collecte, et documenter le choix ;
- même question pour l'**augmentation d'instance** ($A{=}8$), où les huit symétries ont leur propre historique.

**Sur la taille du réseau — la question de ton directeur.** Les données du papier suggèrent que ce n'est pas nécessaire, mais avec une nuance importante.

Contre l'agrandissement : `tab:v3arch` montre un plateau de capacité ($32{\times}1 \to 64{\times}2$ ne gagne que $-0.049$, sous le floor), et le résultat négatif des pair descriptors s'accompagne de l'explication « ce qui limite la politique n'est pas l'information disponible ».

Mais **la nature de l'information diffère** : les pair descriptors étaient largement *redondants* (le coût d'insertion est calculable depuis des coordonnées déjà présentes). La stagnation et l'écart à l'incumbent sont **strictement nouveaux** — aucune combinaison des 23 features actuelles ne les reconstruit, puisqu'ils dépendent de l'historique. Le précédent négatif ne s'applique donc pas directement.

Protocole propre : plan $2 \times 2$, {features off, on} × {$32{\times}1$, $64{\times}2$}, sur 10 seeds. Ça répond à la question de la taille **conditionnellement** aux nouvelles features, ce qui est la bonne façon de la poser — et ça se combine avec G.7.

### G.10 — La version ambitieuse : contrôle appris de l'acceptation **[Tier 2, à fort potentiel]**

L'objectif que ton directeur formule — « gérer la température / le critère de Metropolis tout seul » — n'est **pas atteignable** par G.9 seule, et il faut être clair là-dessus : avec les features en boucle fermée, la politique peut *réagir* à la stagnation en proposant autrement, mais elle ne peut pas modifier la probabilité d'acceptation, qui reste dans l'environnement.

Deux voies bien distinctes :

**(a) Adaptation passive** = G.9. La politique voit l'état de recherche, le schedule reste fixe. Bon marché, pas de changement d'architecture.

**(b) Contrôle actif.** La température devient une **action**. Concrètement : une seconde tête sur l'acteur qui produit un multiplicateur discret de $T$ (par exemple $\{\times 0.5, \times 1, \times 2\}$), ou directement $T$ dans un intervalle. Le MDP change — l'action devient (paire de nœuds, réglage de température).

C'est là qu'il y a un vrai papier, pour une raison précise : **cela répond à la question ouverte de ta propre conclusion**, *« The annealing budget should thus be set from $N$ and the target quality, ideally automatically »*. Un contrôleur d'acceptation appris, conditionné à l'état de recherche, est littéralement cette automatisation.

**⚠ Le piège de reward hacking, à traiter avant d'implémenter.** Avec la récompense actuelle (`immediate`, $r_t = \Delta_t$), si la politique contrôle l'acceptation, la stratégie optimale est de mettre $T \to 0$ immédiatement : n'accepter que les mouvements améliorants garantit une récompense toujours positive. La politique maximiserait sa récompense en détruisant la recherche — descente immédiate dans le premier optimum local.

Le couplage correct est donc avec la récompense **`global_best`** ($r_t = \max(0, B_{t-1} - B_t)$), qui ne récompense que les nouveaux incumbents et rend l'exploration rentable. L'ablation du papier montre qu'elle est viable ($16.510$ contre $16.459$, dans le cluster dense), contrairement à la récompense terminale qui s'effondre ($19.411$). **C'est un cas où deux dimensions du design study, validées séparément, doivent être rejouées ensemble** — exactement le type d'interaction que la validation séquentielle ne voit pas (cf. `app:stats`).

**Baselines obligatoires**, et ce point est non négociable au vu du résultat CW+SA :
- réchauffage sur stagnation, à seuil fixe ;
- *record-to-record travel* : accepter si $C(s_t) \le B_t (1 + \delta)$, un seul paramètre ;
- *late acceptance hill climbing*, un seul paramètre.

Si une règle adaptative classique à un ou deux paramètres égale le contrôleur appris, c'est encore la réponse « Possible Overkill ? ». Il vaut mieux la produire soi-même.

**Contrôle anti-artefact** : les deux features pourraient n'agir que comme un meilleur proxy du budget restant. Contrôle à prévoir — les remplacer par un signal de statistiques comparables mais décorrélé de la recherche réelle.

**Le résultat qui vaudrait le déplacement** : un contrôleur appris à $N{=}100$ qui transfère à $N{=}1000$ ou $10\,000$ **sans réglage**, là où le schedule géométrique doit être recalibré par taille. Ce serait la réponse directe à l'observation « the currency is steps per node, not steps », et le seul élément du papier qui parle au-delà du CVRP.

---

## G-bis. Additions « vendeuses »

### G-bis.1 — Résoudre **une seule** instance vite **[fort]**

C'est le meilleur argument commercial disponible, et le papier l'effleure sans le développer. Aujourd'hui le débit vient du batch de 10 000 instances ; en pratique on veut souvent résoudre **une** instance vite.

La dimension batch est alors libre : départs multiples × augmentations dihédrales × restarts, tous en parallèle sur une instance. `fig:multistart` montre déjà que best-of-10 récupère l'essentiel de l'écart, et l'analyse $A{=}8$ montre que la largeur parallèle s'échange contre les pas séquentiels à raison de $10\times$.

Résultat visé : *une instance CVRP-100 résolue à $x$ % en $y$ millisecondes*. C'est un chiffre qu'un industriel comprend immédiatement, et où ni NDS (5 s/instance) ni HGS ne sont compétitifs.

### G-bis.2 — Transfert vers une variante légèrement plus générale **[fort]**

Flotte hétérogène : capacité différente par camion, nombre de véhicules limité, idéalement matrice de distances par véhicule.

C'est le meilleur rapport valeur/coût du papier, parce que **c'est là que l'architecture gagne structurellement** : ajouter une contrainte à un SA masqué, c'est modifier le masque ; pour une méthode constructive, c'est réentraîner un décodeur. Les features à ajouter sont peu nombreuses (capacité du véhicule de la route, nombre de routes ouvertes contre limite).

Et ça répond par anticipation au reproche « encore du CVRP euclidien uniforme ».

### G-bis.3 — Heuristiques rapides supplémentaires **[Tier 2]**

Mesurer ce qu'on laisse sur la table : split optimal périodique (ton directeur le rapporte utile — il est linéaire, cf. Vidal 2016), 2-opt intra-route en post-traitement, or-opt, nettoyage final. Si une passe de post-traitement à coût négligeable gagne 0,5 %, il vaut mieux le savoir et l'intégrer que le voir apparaître en review.

---

## H. Ordre d'exécution proposé

Si refus AAAI, en supposant ~8 semaines avant la deadline suivante.

Le calendrier est réorganisé autour d'un principe : **on ne réécrit rien avant de savoir ce que l'apprentissage apporte réellement.**

### Phase 0 — Établir les faits (semaines 1–2)

| # | Travail |
|---|---|
| A-bis (vérif.) | Valider le résultat CW+SA : même jeu de test, même convention de coût, même comptabilité du temps, variance, normalisation matérielle |
| G.1, G.2, G.3 | Implémenter restriction $k$-NN, 2-opt(*) mélangé à l'insertion, construction CW à l'entraînement |
| G.8 | Profiler, puis `torch.compile` — à faire tôt, tout le reste tourne plus vite ensuite |
| G.5, G.6, **G.9** | Retrait de `route_pct`, centroïdes sans dépôt, **features de stagnation et d'écart à l'incumbent** — à grouper dans un même réentraînement |

### Phase 1 — L'expérience décisive (semaines 3–4)

| # | Travail |
|---|---|
| **A-bis.1** | **Plan factoriel : init × voisinages × restriction × proposition.** Tout le papier en dépend |
| A-bis.2 | Réentraînement de la version forte de LG-SA⁺ |
| D.3 | Courbes anytime HGS / LKH-3 / OR-Tools / CW+SA à budget par instance apparié |

**Point de décision.** À l'issue de la phase 1, la nature du papier est fixée : guidage appris au-dessus d'une machinerie forte (cas 1), rendements décroissants (cas 2), ou « Possible Overkill ? » pour le SA (cas 3). La suite en dépend, et il faut en discuter avec ton directeur avant de rédiger.

### Phase 2 — Consolidation (semaines 5–7)

| # | Travail |
|---|---|
| A-ter.1–4 | NDS : lecture, réécriture intro + related works, réexécution, ligne dans le Tableau 1 |
| B.1–B.9 | Revendications, cohérence numérique, anonymat — à refaire **après** la phase 1, les chiffres auront changé |
| D.5, D.6 | Bootstrap apparié au niveau instance ; ancien LG-SA sur le jeu de test actuel |
| C.4, G.7 | Seeds portés à 15 sur curriculum, critique, PPO passes, et $32{\times}1$ vs $64{\times}2$ (avec et sans les features G.9) |
| G.4 | Features de densité : les quatre variantes, avec réentraînement |
| **G.10** | **Contrôle appris de l'acceptation** — avec la récompense `global_best`, et les baselines adaptatifs classiques |
| F | Bibliographie 2025–2026 |

### Phase 3 — Valorisation (semaines 8–10)

| # | Travail |
|---|---|
| **G-bis.1** | Résolution rapide d'une instance unique — le meilleur argument commercial |
| **G-bis.2** | Flotte hétérogène — le meilleur rapport valeur/coût |
| G-bis.3, D.9 | Heuristiques rapides supplémentaires ; capacité étendue ($128{\times}2$, $256{\times}3$) |
| E.1–E.4 | Réécriture autour de la thèse |
| D.13 | Adaptation en ligne, **s'il reste du temps** — sa vraie valeur est de servir de test décisif au papier 3 |

### Si le temps manque

Par ordre strict : **A-bis** (sans négociation, la revendication centrale en dépend), puis **G.1–G.3** (les trois ingrédients forts), puis **A-ter** (NDS), puis **B** (gratuit), puis **G-bis.1 et G-bis.2** (ce qui vend le papier). Tout le reste est amélioration marginale par comparaison.