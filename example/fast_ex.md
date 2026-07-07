# `fast_ex.py` — inférence LG-SA pour un solveur externe (Rust)

Ce script exécute notre modèle entraîné de **Recuit Simulé Guidé par l'Apprentissage**
(Learning-Guided Simulated Annealing) sur des instances CVRP **que tu fournis**. Il ne
génère *pas* les instances : ton solveur Rust passe les coordonnées, les demandes, la
capacité et une solution initiale, et récupère une solution améliorée.

Vois-le comme une boîte noire :

```
  coords + demandes + capacité + routes initiales  ─▶  [ LG-SA ]  ─▶  meilleures routes + coût
```

Les données passent de Python↔Rust sous forme de **fichiers `.npz` NumPy**
(un `.npz` est simplement un zip de tableaux binaires). Le contrôle / les métadonnées
traversent sous forme de **JSON**.

---

## 1. Ce que fait le code, étape par étape

À chaque appel il :

1. Lit le fichier `.npz` → `coords`, `demands`, `capacity`, `init`.
2. **Analyse ta solution initiale.** Ton `init` est une liste plate d'indices de clients
   où `0` marque un retour au dépôt. Le code découpe sur les `0` pour retrouver les
   routes individuelles.
3. **Valide l'entrée** (échoue bruyamment si c'est faux) :
   - chaque client `1..DIM` doit apparaître **exactement une fois** ;
   - chaque route doit respecter la `capacity` du véhicule.
4. Charge le modèle entraîné (le réseau *actor*) et lance `sa_test` — la boucle rapide de
   Recuit Simulé où, à chaque pas, le réseau de neurones propose un mouvement local et un
   critère de Metropolis l'accepte/rejette.
5. Écrit la meilleure solution trouvée dans un `.npz` de sortie.

Tout le code lourd d'apprentissage est dans `src/` ; `fast_ex.py` n'est que la fine
enveloppe qui gère la frontière fichiers/JSON et la validation des entrées.

---

## 2. Format d'entrée (ce que Rust doit écrire)

Un seul fichier `.npz` avec **4 tableaux**. `N` = nombre d'instances dans le lot,
`DIM` = nombre de clients.

| Tableau    | dtype     | forme            | signification                                                    |
|------------|-----------|------------------|------------------------------------------------------------------|
| `coords`   | `float32` | `[N, DIM+1, 2]`  | coordonnées des nœuds ; **l'indice 0 est le dépôt**, 1..DIM les clients |
| `demands`  | `int64`   | `[N, DIM+1]`     | demande par nœud ; **la demande du dépôt doit être 0**          |
| `capacity` | `int64`   | `[N]` ou `[N,1]` | capacité du véhicule pour chaque instance                       |
| `init`     | `int64`   | `[N, L]`         | solution initiale, une ligne par instance (voir ci-dessous)    |

Ajoute 40% de zéro à la fin de la solution

### Le tableau `init` en détail

Chaque **ligne** est la solution initiale d'une instance : une séquence d'indices de
clients (`1..DIM`) où **`0` signifie « retour au dépôt »** (donc `0` sépare les routes).
Complète la droite avec des `0` pour que toutes les lignes aient la même longueur `L`.

Exemple — instance avec les routes `[3, 1, 7]`, `[2, 5]`, `[4, 6]` :

```
[0 ,3 , 1, 7, 0, 2, 5, 0, 4, 6, 0, 0, 0, ...]
```

Tu as besoin d'un `0` en tête, et les `0` de padding sont important.
Ce qui compte aussi:
- chaque client `1..DIM` apparaît **exactement une fois**, et
- la demande totale de chaque route ≤ `capacity`.

> **D'où vient `init` ?** C'est ton solveur Rust qui le produit

### ⚠️ Important : rester proche de la distribution d'entraînement

Le modèle a été entraîné sur un réglage précis, et il est **sensible à l'échelle des
coordonnées** (les x/y bruts sont des variables d'entrée). Pour de bons résultats :
- **Normalise les coordonnées dans `[0, 1] × [0, 1]`** avant de les envoyer.

---

## 3. Format de sortie (ce que Rust relit)

Le `.npz` de sortie contient **3 tableaux** :

| Tableau       | dtype     | forme      | signification                                                        |
|---------------|-----------|------------|----------------------------------------------------------------------|
| `best_routes` | `int64`   | `[N, L']`  | meilleure solution trouvée, **même format plat délimité par `0` que `init`** |
| `cost`        | `float32` | `[N]`      | longueur totale des routes de `best_routes` par instance            |
| `init_cost`   | `float32` | `[N]`      | longueur totale des routes de l'`init` envoyé (pour référence)      |

`best_routes` commence par un `0` (dépôt) et est complété par des `0` à droite. Décode-le
comme tu as encodé `init` : **découpe chaque ligne sur les `0`** pour obtenir les routes.

---

## 4. Deux façons de le lancer

### Mode A — one-shot (le plus simple)

Tourne une fois puis quitte. Recharge le modèle à chaque appel (~2–5 s de démarrage fixe),
donc à utiliser quand tu envoies **un gros lot** (`N` grand) par appel.

```bash
uv run example/fast_ex.py --input in.npz --output out.npz --outer-steps 1000
```

### Mode B — `--serve` (pour de nombreux appels répétés)

Charge le modèle **une seule fois**, puis reste actif à répondre aux requêtes. À utiliser
si ton solveur appelle LG-SA de nombreuses fois dans une boucle — ça supprime le coût de
démarrage par appel (après chauffe, 3 résolutions ont pris ~0,7 s contre ~3 s chacune en
one-shot).

```bash
uv run example/fast_ex.py --serve
```

Ensuite tu lui parles via **stdin/stdout** (voir 6).

---

## 5. Paramètres en ligne de commande

| Option          | Défaut             | Signification                                                     |
|-----------------|--------------------|-------------------------------------------------------------------|
| `--serve`       | désactivé          | Lance la boucle de requêtes persistante au lieu du one-shot.     |
| `--input`       | —                  | Chemin du `.npz` d'entrée (requis en mode one-shot).            |
| `--output`      | —                  | Chemin du `.npz` de sortie (requis en mode one-shot).          |
| `--model`       | le dossier dans `example/models/` | Dossier contenant le checkpoint (`*.pt`) et `HP.yaml`. |
| `--device`      | `cpu`              | `cpu`, `cuda` ou `mps`.                                          |
| `--outer-steps` | `1000`             | Nombre de pas de RS. Plus = meilleur & plus lent. Surchargeable par requête en mode serve. |
| `--seed`        | `1`                | Graine du générateur aléatoire (fixée une fois au démarrage).   |

---

## 6. Le protocole `--serve`

Tout est du **JSON ligne par ligne** : un objet JSON par ligne.

- **stdout** ne transporte *que* les messages du protocole (un JSON par ligne).
- **stderr** transporte les barres de progression / avertissements — ton code Rust peut
  les ignorer.

**Poignée de main.** Au démarrage, le serveur charge le modèle et affiche :

```json
{"status": "ready", "device": "cpu", "model": "20260129_224725_x2uj6g8k"}
```

Attends cette ligne avant d'envoyer des requêtes.

**Requête** (une ligne). `input`/`output` sont requis ; `outer_steps`/`greedy` sont des
surcharges optionnelles par appel :

```json
{"input": "in.npz", "output": "out.npz", "outer_steps": 1000, "greedy": false}
```

**Réponse** (une ligne), en cas de succès :

```json
{"status": "ok", "output": "out.npz", "n": 4, "init_cost_mean": 20.34, "cost_mean": 18.29}
```

Les tableaux complets sont dans le `.npz` `output` ; les nombres de la réponse ne sont
qu'un résumé rapide. En cas de requête invalide, le serveur répond et **continue à
tourner** :

```json
{"status": "error", "message": "ValueError: instance 0: init must visit customers 1..100 exactly once (missing=[5], ...)"}
```

**Arrêt.** Envoie `{"cmd": "shutdown"}` ou ferme simplement stdin.

> ⚠️ **Note sur la reproductibilité :** en mode serve, la graine est fixée une seule fois
> au démarrage, donc envoyer *deux fois* la même instance donne des résultats *légèrement
> différents* (l'échantillonnage du RS fait avancer le RNG partagé). C'est normal. Si tu
> as besoin d'une sortie identique par appel, dis-le moi et j'ajouterai un champ `"seed"`
> optionnel à la requête.

---

## 7. Utilisation depuis Rust

Crates suggérées : [`ndarray`], [`ndarray-npy`] (lire/écrire des `.npz`), [`serde_json`]
(le protocole), et `std::process`.

### 7a. Écrire le `.npz` d'entrée

```rust
use ndarray::{Array2, Array3};
use ndarray_npy::NpzWriter;
use std::fs::File;

// coords: [N, DIM+1, 2] f32, dépôt à la ligne 0, coords normalisées dans [0,1]
// demands: [N, DIM+1] i64, demands[_, 0] = 0
// capacity: [N] i64
// init: [N, L] i64, clients 1..DIM avec 0 = séparateur de dépôt, padding 0

let mut npz = NpzWriter::new(File::create("in.npz")?);
npz.add_array("coords",   &coords)?;    // Array3<f32>
npz.add_array("demands",  &demands)?;   // Array2<i64>
npz.add_array("capacity", &capacity)?;  // Array1<i64>
npz.add_array("init",     &init)?;      // Array2<i64>
npz.finish()?;
```

### 7b. Appel one-shot

```rust
use std::process::Command;

let status = Command::new("uv")
    .args(["run", "example/fast_ex.py",
           "--input", "in.npz", "--output", "out.npz",
           "--outer-steps", "1000"])
    .current_dir("/chemin/vers/repo")   // racine du dépôt
    .status()?;
assert!(status.success());

// puis lire out.npz :
use ndarray_npy::NpzReader;
let mut out = NpzReader::new(File::open("out.npz")?)?;
let best_routes: Array2<i64> = out.by_name("best_routes")?;
let cost:        Array1<f32> = out.by_name("cost")?;
// découper chaque ligne de best_routes sur les 0 pour retrouver les routes
```

### 7c. Appel `--serve` (spawn une fois, boucle)

```rust
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

let mut child = Command::new("uv")
    .args(["run", "example/fast_ex.py", "--serve", "--device", "cpu"])
    .current_dir("/chemin/vers/repo")
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null())          // ignorer les barres de progression
    .spawn()?;

let mut stdin  = child.stdin.take().unwrap();
let mut stdout = BufReader::new(child.stdout.take().unwrap());

// 1. attendre la ligne "ready"
let mut line = String::new();
stdout.read_line(&mut line)?;       // {"status":"ready",...}

// 2. pour chaque instance : écrire in.npz, envoyer une requête, lire la réponse, lire out.npz
let req = r#"{"input":"in.npz","output":"out.npz","outer_steps":1000}"#;
writeln!(stdin, "{}", req)?;
stdin.flush()?;

line.clear();
stdout.read_line(&mut line)?;       // {"status":"ok",...} ou {"status":"error",...}
// parser `line` avec serde_json, puis lire out.npz comme en 7b.

// 3. à la fin :
writeln!(stdin, r#"{{"cmd":"shutdown"}}"#)?;
```

Tu peux réutiliser les mêmes noms `in.npz`/`out.npz` à chaque itération, ou utiliser des
fichiers temporaires uniques si tu parallélises.

[`ndarray`]: https://crates.io/crates/ndarray
[`ndarray-npy`]: https://crates.io/crates/ndarray-npy
[`serde_json`]: https://crates.io/crates/serde_json

---

## 8. Fichiers de test fournis

Des `.npz` d'exemple prêts à l'emploi sont dans `example/test_data/` :

| Fichier                 | N | DIM | capacité | note                               |
|-------------------------|---|-----|----------|-------------------------------------|
| `instance_small.npz`    | 1 | 20  | 30       | petit, pour un test rapide          |
| `instance_dim100.npz`   | 1 | 100 | 50       | proche de la distribution d'entraînement |

> Chaque fichier contient un lot de `N` instances ; augmente `N` dans
> `make_test_npz.py` pour tester des lots plus gros.

Pour les (re)générer : `uv run example/make_test_npz.py`. Le script montre aussi, en
Python, comment construire chaque tableau au bon format.

Test rapide :

```bash
uv run example/fast_ex.py \
    --input example/test_data/instance_dim100.npz \
    --output example/test_data/out_dim100.npz \
    --outer-steps 1000
```

---

## 9. Checklist

- [ ] Coordonnées normalisées dans `[0, 1]`, dépôt à l'indice `0`.
- [ ] `demands[_, 0] == 0` (le dépôt n'a pas de demande).
- [ ] `init` visite chaque client `1..DIM` exactement une fois, `0` entre les routes, capacité respectée.
- [ ] Les quatre tableaux d'entrée présents avec les bons dtypes (`float32` pour coords, `int64` pour le reste).
- [ ] Décoder `best_routes` en découpant chaque ligne sur les `0`.
- [ ] Beaucoup d'appels dans une boucle → utiliser `--serve` ; un gros lot → one-shot suffit.
- [ ] Lire `stdout` pour la réponse JSON, ignorer `stderr`.
