# Contributing to PSM-NeuroAI-Final

New here? Never really coded before? **You are in the right place.** This guide walks you
through everything, one baby step at a time. If a step doesn't work, that's normal — copy
the error into a message and spam a teammate. 🙂

---

## Table of contents

1. [The big idea (read this first)](#1-the-big-idea-read-this-first)
2. [Get set up](#2-get-set-up)
   - [Step 1 — Download the data](#step-1--download-the-data)
   - [Step 2 — Get the code and install it](#step-2--get-the-code-and-install-it)
   - [Step 3 — Tell the code where the data lives](#step-3--tell-the-code-where-the-data-lives)
3. [Map of the repo (what all the folders are)](#3-map-of-the-repo-what-all-the-folders-are)
4. [⭐ Add your own model (the main event)](#4--add-your-own-model-the-main-event)
5. [Save your work and open a pull request](#5-save-your-work-and-open-a-pull-request)
6. [Recommended reading](#6-recommended-reading)
7. [Things we still need to do (roadmap)](#7-things-we-still-need-to-do-roadmap)

---

## 1. The big idea (read this first)

We are asking one question: **do AI vision models "see" the world the way brains do?**

We have two kinds of brain data:

- **Algonauts / NSD** — human brain scans (fMRI) of people looking at photos.
- **Triple-N** — recordings from individual neurons in macaque monkeys looking at the same photos.

Here's the trick that lets us compare a brain to an AI model. For any system — a brain
region *or* a neural network — we can build an **RDM** ("Representational Dissimilarity
Matrix"). Think of it as a big grid that answers, for every pair of images: *"how
differently did this system react to image A vs. image B?"* If two systems produce
similar-looking RDMs, they are organizing visual information in a similar way.

So the whole project boils down to:

```
your model  ──▶  its RDM  ──▶  compare to  ──▶  brain RDMs (human + monkey)
```

**Your job as a contributor is tiny: plug in a new model.** All the hard parts — loading
the brain data, building the brain RDMs, and running the comparison — are already written.
You just implement the model, and the shared machinery compares it to *both* brain datasets
in the exact same way. (See the [ABSTRACT in the README](./README.md) for the science.)

---

## 2. Get set up

### Step 1 — Download the data

The brain datasets are too big for GitHub, so they live on Google Drive:

**➡️ [Download the data here](https://drive.google.com/drive/folders/1bWpBi-X9iN8x-HsuRwXDUX-GzELAK3sC)**

Download it and put it somewhere you'll remember (an external SSD is great — these files are
large). You'll point the code at these folders in [Step 3](#step-3--tell-the-code-where-the-data-lives).
**Do not** put the data inside the project folder and **do not** commit it to GitHub — it's
already ignored on purpose.

### Step 2 — Get the code and install it

Pick the path that sounds like you:

| Path | Who it's for |
|------|--------------|
| [🏃 I run it on my own computer](#-i-run-it-on-my-own-computer) | You have a decent laptop/desktop and want it local. |
| [☁️ I run it on my computer, but let Google's servers do the heavy lifting](#-i-run-it-on-my-computer-but-let-googles-servers-do-the-heavy-lifting) | Same as above, but you don't want to melt your CPU. |
| [🤷 My laptop is a potato — just use Colab](#-my-laptop-is-a-potato--just-use-colab) | You'd rather do everything in the browser. |

#### 🏃 I run it on my own computer

**1. Get the code (do this once):**

```bash
git clone git@github.com:lucas-nunn/PSM-NeuroAI-Final.git
cd PSM-NeuroAI-Final
```

**2. Install VS Code** (the editor everyone here uses): https://code.visualstudio.com/

**3. Install two VS Code extensions** (search for them in VS Code's Extensions panel):

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

**4. Install `uv`** (the tool that installs all our Python packages for us):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**5. Install everything the project needs:**

```bash
uv sync --all-extras
```

This reads [pyproject.toml](./pyproject.toml) and creates a `.venv` folder with the exact
right versions of PyTorch, NumPy, etc.

**6. Turn the environment on:**

```bash
source .venv/bin/activate
```

**7. Point a notebook at the environment:** open (or create) a notebook in
[notebooks/](./notebooks), click **"Select Kernel"** in the top-right, and choose
**`psm-final`**. You're ready.

#### ☁️ I run it on my computer, but let Google's servers do the heavy lifting

Do everything in [🏃 I run it on my own computer](#-i-run-it-on-my-own-computer) above,
then also install the **[Colab VS Code extension](https://marketplace.visualstudio.com/items?itemName=Google.colab)**.
Now you write code in VS Code as usual, but it runs on a free Google GPU instead of your
laptop.

#### 🤷 My laptop is a potato — just use Colab

You can use everything in this repo straight from a browser on
[Google Colab](https://colab.research.google.com/). Paste this at the very top of your
Colab notebook:

```python
# Replace YOUR_BRANCH with the name of your branch (e.g. main, or your feature branch)
!pip install git+https://github.com/lucas-nunn/PSM-NeuroAI-Final.git@YOUR_BRANCH
```

Heads up: Colab pulls the code **from GitHub**, so any change you make must be
[pushed to GitHub first](#5-save-your-work-and-open-a-pull-request) before Colab can see it.
A good starting point is [notebooks/data_exploration.ipynb](./notebooks/data_exploration.ipynb).

### Step 3 — Tell the code where the data lives

The code finds the datasets through two "environment variables." Create a file named
**`.env`** in the top folder of the project (right next to this file) and put your download
paths in it:

```bash
# .env  — use the real folders you downloaded in Step 1
ALGONAUTS_DIR=/path/to/algonauts/train
TRIPLE_N_DIR=/path/to/triple-N
```

VS Code automatically loads `.env` when it starts a notebook, so `os.environ["TRIPLE_N_DIR"]`
in the notebooks will just work. (On Colab there's no `.env`; set them by hand instead, e.g.
`import os; os.environ["TRIPLE_N_DIR"] = "/content/drive/..."`.)

> `FREESURFER_HOME` is a third, **optional** variable only needed if you do brain-surface
> plotting. You can ignore it for the model comparison.

---

## 3. Map of the repo (what all the folders are)

Most of the important code is under [src/psm_final/](./src/psm_final). Here's the tour:

| Location | What it does |
|----------|--------------|
| [dataset/algonauts.py](./src/psm_final/dataset/algonauts.py) | Loads the **human fMRI** (Algonauts/NSD) data and builds its brain RDMs. |
| [dataset/triple_n.py](./src/psm_final/dataset/triple_n.py) | Same idea, but for the **macaque neuron** (Triple-N) data. |
| [analysis/model.py](./src/psm_final/analysis/model.py) | The **general base class** (`ModelAnalysisBase`) that turns *any* model into an RDM and compares it to the datasets. **This is the one you build on.** |
| [analysis/beta_vae_analysis.py](./src/psm_final/analysis/beta_vae_analysis.py) | The **worked example**: turns a trained β-VAE into an RDM. Copy this to make your own. |
| [analysis/correlating.py](./src/psm_final/analysis/correlating.py) | The math for building RDMs and comparing them (you probably won't touch this). |
| [models/beta_vae.py](./src/psm_final/models/beta_vae.py) | Defines a β-VAE **and** its training script. Run this if you want to train a VAE from scratch. |
| [helpers/](./src/psm_final/helpers) | Leftover odds-and-ends from class. You can ignore this folder. |
| [notebooks/](./notebooks) | Where the actual "fun analysis" happens. Copy an existing notebook to start yours. |

**The key idea:** everyone adds new models the *same* way — by subclassing
`ModelAnalysisBase` in the `analysis/` folder. Because every model plugs into the same base
class, the project can run them all in one go and compare each one to both brain datasets in
exactly the same way. **So all you have to do is implement your model.**

---

## 4. ⭐ Add your own model (the main event)

This is the actual contribution. The goal: pick a pretrained AI vision model, wrap it in a
small class, and let the shared machinery compare it to the brains.

Follow along with the finished example while you read: the class
[analysis/beta_vae_analysis.py](./src/psm_final/analysis/beta_vae_analysis.py) and the
notebook [notebooks/beta_vae.ipynb](./notebooks/beta_vae.ipynb).

### What you're building, in plain words

The base class `ModelAnalysisBase` already knows how to:

1. find the 1000 Triple-N stimulus images on disk,
2. hand you each image, one at a time,
3. collect what your model says about each image, and
4. turn all of that into an RDM (via its `.rdm()` method).

It only needs **two things from you**, which is why you subclass it:

- **`__init__`** — download / load your pretrained model and get it ready.
- **`embedding(self, image)`** — given one image, return your model's "opinion" of it as a
  flat list of numbers (a 1-D vector). That's the layer you want to study.

That's it. Everything else is inherited.

### Step 1 — Pick a pretrained model

Grab something off the internet — ideally an **autoencoder (AE), VAE, or diffusion model**,
and ideally one **trained on MS COCO** (the same kind of natural photos the brains looked
at). Almost any pretrained vision model works, as long as you can download it and pull
activations out of one of its layers.

### Step 2 — Create a new analysis file

Make a new file next to the example:

```
src/psm_final/analysis/your_model_analysis.py
```

### Step 3 — Write the class

Copy this template and swap in your model. Every line is commented:

```python
import numpy as np
import torch
import torchvision.transforms as transforms

from psm_final.analysis.model import ModelAnalysisBase


class YourModelAnalysis(ModelAnalysisBase):
    def __init__(self, triple_n_path, device=None):
        # This line lets the base class find the Triple-N stimulus images later.
        super().__init__(triple_n_path)

        # Use the GPU if there is one, otherwise the CPU.
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # (1) DOWNLOAD your pretrained model from the internet.
        #     torch.hub downloads the weights the first time and caches them.
        #     Swap this line for whatever AE / VAE / diffusion model you chose.
        self.model = torch.hub.load("pytorch/vision", "resnet50", weights="DEFAULT")
        self.model.to(self.device).eval()   # .eval() = "just make predictions, don't train"

        # (2) Pick ONE layer to study and grab its output with a "hook".
        #     A hook is a little spy that stashes a layer's activations every
        #     time the model runs. Change `avgpool` to the layer you care about.
        self._features = {}
        self.model.avgpool.register_forward_hook(
            lambda module, inp, out: self._features.__setitem__("layer", out)
        )

        # (3) How to turn a PIL image into the tensor your model expects.
        #     Match whatever preprocessing your chosen model was trained with.
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def embedding(self, image):
        # The base class hands you one PIL image at a time. Convert to RGB in case
        # a stimulus is grayscale, preprocess it, and add a fake "batch" dimension.
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():          # don't waste memory tracking gradients
            self.model(x)              # run the model; the hook fills self._features

        # Return a flat 1-D numpy vector: this is the embedding for this one image.
        return self._features["layer"].squeeze().cpu().numpy().ravel()
```

Two rules for `embedding` and you can't go wrong:

- It receives **one** [PIL](https://pillow.readthedocs.io/) image and returns **one** 1-D
  NumPy array.
- Every image must return a vector of the **same length**.

### Step 4 — Make a notebook and get your RDM

Copy [notebooks/beta_vae.ipynb](./notebooks/beta_vae.ipynb) to
`notebooks/your_model.ipynb`, then create your class and ask it for its RDM:

```python
import os
from psm_final.analysis.your_model_analysis import YourModelAnalysis

analysis = YourModelAnalysis(triple_n_path=os.environ["TRIPLE_N_DIR"])
model_rdm = analysis.rdm()   # <- the base class does all the work here
```

### Step 5 — Compare it to the brains 🧠

Now go wild. Build the brain RDMs and correlate them against yours (higher `rho` = more
brain-like):

```python
import os
import numpy as np
from scipy.stats import spearmanr
from psm_final import Algonauts, TripleN, shared_stimuli

# The images shared across every dataset, so everything lines up.
shared_ids = shared_stimuli("../nsd_expdesign.mat")
algonauts = Algonauts(os.environ["ALGONAUTS_DIR"], shared_ids)
triple_n = TripleN(os.environ["TRIPLE_N_DIR"])

# vs. macaque area V4:
tn_rdm = triple_n.compute_rdm(area_label="V4")
rho, p = spearmanr(model_rdm, tn_rdm)
print(f"your model vs. Triple-N V4: rho = {rho:.3f}")
```

The bottom of [notebooks/beta_vae.ipynb](./notebooks/beta_vae.ipynb) has a full example that
compares a model against fMRI ROIs, monkey brain areas, individual monkeys, and
neuron-preference types — copy whatever's useful.

> **Want to train your own β-VAE instead of downloading a model?** Just run the training
> script: `python -m psm_final.models.beta_vae --coco_root /path/to/coco/images`. It saves a
> `vae.pth` you can point [BetaVAEAnalysis](./src/psm_final/analysis/beta_vae_analysis.py)
> at. But for a new contribution, **downloading a pretrained model is the easy path** — start
> there.

### Step 6 — (optional) register your class for easy imports

The example classes are exported so you can write `from psm_final import BetaVAEAnalysis`.
If you want the same convenience, add your class to the `__all__` and `_EXPORTS` lists in
[src/psm_final/analysis/__init__.py](./src/psm_final/analysis/__init__.py) and
[src/psm_final/__init__.py](./src/psm_final/__init__.py). Totally optional — importing
straight from your file (as in Step 4) works fine.

When it looks good, [open a pull request](#5-save-your-work-and-open-a-pull-request) and a
maintainer will help you get it merged. 🎉

---

## 5. Save your work and open a pull request

New to Git? This 12-minute video explains the whole idea:
**[Git & GitHub for beginners](https://youtu.be/hZS96dwKvt0?si=ZzU0I7dEIgLdXZ2L)**.

Git is just "save game" for code. Here's the whole loop.

**Start your own workspace (a "branch") so you don't disturb `main`:**

```bash
git checkout -b your-branch-name
```

Now make your changes (add your model, run your notebook, etc.).

**Check what you've changed / which branch you're on at any time:**

```bash
git status
```

**Save a checkpoint as you go** (do this often — every time you finish a little piece):

```bash
git add .
git commit -m "short description of what you did"
```

**Grab teammates' latest work** so you don't fall behind:

```bash
git pull origin main
```

**Send your branch up to GitHub:**

```bash
git add .
git commit -m "short description of what you did"
git push origin your-branch-name
```

**When your branch is ready for the big leagues — open a pull request:**

1. Take a breath.
2. Pull the latest code so you're up to date: `git pull origin main`
   - If you get a **merge conflict**, don't panic — Git is just asking you to choose which
     version of a clashing line to keep. Fix the marked spots, then `git add .` and
     `git commit`. Ask for help if it's confusing; everyone hits these.
3. Do a final `git add .`, `git commit -m "..."`, and `git push origin your-branch-name`.
4. Go to the [repo on GitHub](https://github.com/lucas-nunn/PSM-NeuroAI-Final) → **Pull
   requests** → **New pull request**.
5. Set **base = `main`** and **compare = `your-branch-name`**.
6. Skim the changed files to make sure they're what you expect.
7. Write a short, friendly description of what you did and why.
8. Submit it, then **add a teammate as a reviewer and gently spam them.** 🎉

---

## 6. Recommended reading

If you want the science behind this project, start with the paper that inspired it:

- I. Higgins et al., *"Unsupervised deep learning identifies semantic disentanglement in
  single inferotemporal face patch neurons,"* Nature Communications, 2021 —
  **[read it here](https://www.nature.com/articles/s41467-021-26751-5)**.

More background and the full reference list are in the [README](./README.md).

---

## 7. Things we still need to do (roadmap)

Looking for something to work on? Pick something from here. The **mandatory** list is what the
project needs to be complete; the **big leagues** list is the ambitious, stretchy stuff. Not
sure where to start? A pixel-space baseline (below) is the friendliest first pull request.

### ✅ Mandatory — the baseline models to compare

Right now we mostly have the β-VAE. The whole point is a *comparison*, so we need a few
reference points to compare it against. Each of these is a new model class in `analysis/`,
exactly like [Step 4 above](#4--add-your-own-model-the-main-event):

- [ ] **Pixel space** — the dumbest possible "model": use the raw image pixels themselves as
  the embedding (just flatten the image into one long vector). This is the floor everything
  else has to beat, and it's the easiest one to implement — great first PR.
- [ ] **A non-autoencoder vision model** — a normal *supervised* image model (e.g. a pretrained
  CNN or vision transformer). The ResNet-18 in
  [notebooks/baseline_starter.ipynb](./notebooks/baseline_starter.ipynb) already does this —
  turn it into a proper class under `analysis/` and it's done.
- [ ] **A pretrained AE / VAE / diffusion model** — an *unsupervised* model someone else already
  trained (ideally on MS COCO). This is the headline comparison the paper is built around.

### 🏆 Big leagues — stretch goals

Bigger lifts, more interesting science. Several of these should live in the shared base class
(`ModelAnalysisBase`) so that **every** model gets them for free, the same way every model
already gets `.rdm()`:

- [ ] **Non-parametric statistical tests.** Our current p-values (from `spearmanr`) assume things
  about the data that RDMs don't satisfy. Swap in permutation / bootstrap tests so the
  "is this correlation real?" answer is actually trustworthy.
- [ ] **Encoding models.** Instead of only comparing whole-pattern RDMs, fit a model that predicts
  each brain response *from* the model's features and measure how much variance it explains.
  **Generalize this into `ModelAnalysisBase`** (e.g. an `.encoding()` method alongside `.rdm()`)
  so it works for any model automatically.
- [ ] **Custom statistics from the Higgins paper.** The single-unit / disentanglement analyses
  from the [reference paper](https://www.nature.com/articles/s41467-021-26751-5) — e.g. matching
  individual latent units to individual IT neurons — not just population RDMs.
- [ ] **Searchlight analyses.** Slide a small window across the brain and compute the model↔brain
  match everywhere, producing a brain *map* of where a model fits best. (We already have β-VAE
  searchlight figures for subj01 — generalize the approach so it runs for any model.)
- [ ] **Comparisons across varying ROIs / segments.** Systematically sweep the comparison across
  many brain regions, cortical segments, and unit groupings (area, macaque, firing preference,
  depth, …) instead of a hand-picked few — and line them up across datasets.
