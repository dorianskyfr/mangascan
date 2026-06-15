# Mangascan 🈂️ → 🗨️ — v0.6

Application de bureau (Python) qui **capture l'écran en temps réel**, **détecte
le texte des planches de manga** (japonais en priorité, coréen/chinois en
repli), le **traduit** dans la langue de votre choix, et affiche la traduction
via une **overlay transparente** posée par-dessus l'écran.

Idéal pour lire un manga/manhwa/manhua dans une visionneuse, un navigateur ou
un lecteur d'images sans quitter la page.

---

## ✨ Fonctionnalités

- **Sélection de zone** : dessinez un rectangle à surveiller (façon ShareX), ou
  capturez un monitor entier.
- **OCR spécialisé manga** : [`manga-ocr`](https://github.com/kha-white/manga-ocr)
  pour le japonais, [`EasyOCR`](https://github.com/JaidedAI/EasyOCR) en repli
  pour le coréen et le chinois.
- **Détection automatique de la langue source** (`langdetect`). Si la langue
  source correspond à la langue cible, rien n'est affiché.
- **Traduction Google (gratuit) ou DeepL** (clé API requise) via
  `deep-translator`, avec **cache mémoire** pour éviter les appels répétés.
- **Overlay transparente** : fenêtre sans décoration, toujours au premier plan,
  transparente aux clics (`WA_TransparentForMouseEvents`), texte positionné
  par-dessus la bulle d'origine, **police adaptative**.
- **Hotkey globale** `Ctrl+Shift+T` pour activer/désactiver l'overlay.
- **Fenêtre de contrôle** : Start/Stop, statut temps réel (FPS, nombre de
  bulles, erreurs API), sélecteur de langue et de moteur, logs de debug.
- **Thread de capture/OCR séparé du thread UI** (`QThread`) : l'interface ne
  gèle jamais.

---

## 📁 Structure du projet

```
mangascan/
├── main.py            # Point d'entrée + fenêtre de contrôle
├── worker.py          # Thread de capture/OCR/traduction (QThread)
├── capture.py         # Capture d'écran (mss)
├── ocr.py             # Wrapper manga-ocr + EasyOCR
├── translator.py      # Wrapper deep-translator (Google/DeepL) + cache
├── overlay.py         # Overlay PyQt5 transparente + sélecteur de zone
├── detector.py        # Détection des bulles (OpenCV)
├── config.py          # Chargement/sauvegarde de config.json
├── config.json        # Configuration utilisateur
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

> Prérequis : **Python 3.10+**. Testé en priorité sur **Windows 10/11**.

1. (Recommandé) Créez un environnement virtuel :

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate
   ```

2. Installez les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

   > ℹ️ `torch` est volumineux. Pour une version CPU/GPU spécifique, suivez les
   > instructions sur https://pytorch.org/get-started/locally/ avant
   > d'installer le reste.

### Setup de `manga-ocr`

`manga-ocr` télécharge automatiquement son modèle (~400 Mo) depuis Hugging Face
**au premier lancement de l'OCR**. Aucune étape manuelle n'est nécessaire,
assurez-vous simplement d'avoir une connexion Internet lors de la première
utilisation. Le modèle est ensuite mis en cache localement.

Pour tester l'installation :

```bash
python -c "from manga_ocr import MangaOcr; MangaOcr()"
```

Si `manga-ocr` n'est pas installé ou échoue, l'application bascule
automatiquement sur EasyOCR sans planter.

### Configuration DeepL (optionnelle)

Par défaut, Mangascan utilise **Google Translate (gratuit)**. Pour utiliser
**DeepL** :

1. Récupérez une clé API sur https://www.deepl.com/pro-api
2. Renseignez-la, au choix :
   - dans la variable d'environnement `DEEPL_API_KEY` :

     ```bash
     # Windows (PowerShell)
     $env:DEEPL_API_KEY="votre_cle"
     # Linux / macOS
     export DEEPL_API_KEY="votre_cle"
     ```

   - ou directement dans `config.json` (champ `deepl_api_key`).
3. Sélectionnez **DeepL** dans le menu déroulant « Traducteur » de l'interface
   (ou mettez `"translator": "deepl"` dans `config.json`).

---

## 🚀 Lancement

**Option 0 — exécutable Windows (.exe)** : aucune installation Python requise.
Téléchargez `Mangascan.exe` depuis la page
[Releases](https://github.com/dorianskyfr/mangascan/releases) et double-cliquez.
Au premier lancement de l'OCR, le modèle manga-ocr (~400 Mo) se télécharge
automatiquement (connexion Internet requise).

> 🔧 L'exe est construit automatiquement par GitHub Actions (PyInstaller sur un
> runner Windows). Pour le générer soi-même :
> ```bash
> pip install -r requirements.txt pyinstaller
> pyinstaller --onefile --windowed --name Mangascan \
>   --collect-all manga_ocr --collect-all easyocr \
>   --collect-all torch --collect-all torchvision --collect-all cv2 \
>   mangascan.py
> # -> dist/Mangascan.exe
> ```

**Option 1 — fichier unique (recommandé)** : toute l'application tient dans un
seul fichier exécutable.

```bash
python mangascan.py
```

**Option 2 — version modulaire** (mêmes fonctionnalités, code réparti) :

```bash
python main.py
```

1. Cliquez sur **✎ Sélectionner une zone** pour délimiter la région à
   surveiller (ou **🖵 Monitor entier**).
2. Choisissez la **langue cible** et le **traducteur**.
3. Cliquez sur **▶ Démarrer**.
4. Les traductions apparaissent par-dessus les bulles. Utilisez `Ctrl+Shift+T`
   (ou le bouton **👁**) pour masquer/afficher l'overlay.

> 🐧 **Linux** : la hotkey globale (module `keyboard`) nécessite généralement
> les droits root (`sudo`). Sur Wayland, la capture/overlay peut être limitée ;
> X11 est recommandé.

---

## ⚙️ Configuration (`config.json`)

```json
{
  "target_language": "fr",
  "capture_fps": 2,
  "translator": "google",
  "deepl_api_key": "",
  "hotkey_toggle": "ctrl+shift+t",
  "monitor_index": 1,
  "capture_region": null,
  "debug_logs": true
}
```

| Champ | Description |
|-------|-------------|
| `target_language` | Code ISO de la langue cible (`fr`, `en`, `es`, …). |
| `capture_fps` | Nombre de captures analysées par seconde. |
| `translator` | `google` (gratuit) ou `deepl` (clé requise). |
| `deepl_api_key` | Clé DeepL (sinon, variable `DEEPL_API_KEY`). |
| `hotkey_toggle` | Raccourci global d'activation de l'overlay. |
| `monitor_index` | Monitor capturé (1 = écran principal) si pas de zone. |
| `capture_region` | Zone `{top,left,width,height}` ou `null`. |
| `debug_logs` | Affiche les logs de debug dans l'interface. |

Le fichier est créé automatiquement avec ces valeurs par défaut au premier
lancement et mis à jour à chaque changement via l'interface.

---

## 🧩 Dépannage

| Problème | Solution |
|----------|----------|
| `manga-ocr` ne se charge pas | Vérifiez votre connexion (téléchargement du modèle) ; l'app bascule sur EasyOCR. |
| Aucune traduction n'apparaît | Source = cible ? Vérifiez la zone et la langue cible. |
| Erreur API DeepL | Vérifiez la clé et votre quota ; sinon repassez sur Google. |
| Hotkey inopérante (Linux) | Lancez avec `sudo` ou rebindez via `config.json`. |
| Overlay invisible | Vérifiez qu'elle n'est pas masquée (`Ctrl+Shift+T`). |

---

## 📜 Licence

Projet fourni à but éducatif. Respectez les droits d'auteur des œuvres traduites
et les conditions d'utilisation des API de traduction.
