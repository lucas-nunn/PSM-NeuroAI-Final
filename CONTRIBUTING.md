# SETUP

**choose your path:**

- [I am cool](#i-am-cool): running the code locally on your computer
- [I am cool but I don't want to stress my precious CPU](#i-am-cool-but-i-dont-want-to-stress-my-precious-cpu): running the code via Colab inside vscode on your computer
- [I prefer easy things in life](#i-prefer-easy-things-in-life): running the code on Colab

## I am cool

**clone the repo**

- `git clone git@github.com:lucas-nunn/PSM-NeuroAI-Final.git"`

**make a data and results folder**

- `mkdir data results`
- store data in `../data`
- store results in `../results`
- don't push them ^^ to github
- put figures in `../figures`
- put OG source code in `../src/psm_final`
- put analysis notebooks in `../notebooks`

**install VS Code**

- https://code.visualstudio.com/

**install some VS Code extensions**

- [python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

**install uv**

- `curl -LsSf https://astral.sh/uv/install.sh | sh`

**install dependencies**

- `uv sync --all-extras`

**activate the environment**

- `source .venv/bin/activate`

**create a jupyter notebook in ../notebooks**

- open it
- top right corner "select kernel"
- find "psm-final"

## I am cool but I don't want to stress my precious CPU

**follow the above**

- ...

**install VS Code Colab extension**

- https://marketplace.visualstudio.com/items?itemName=Google.colab

## I prefer the simple things in life / my klunkbook2000 can't run this s%$t

**put this blob of vomit at the top of your Colab so you can use the project code in the Colab**

```
# replace YOUR_BRANCH with the name of your branch
!pip install git+https://github.com/lucas-nunn/PSM-NeuroAI-Final.git@infra
```

- your new changes must be on Github, so you have to push your branch (see below)
- somehow get your data into google drive

# CONTRIBUTING

**watch this**

- https://youtu.be/hZS96dwKvt0?si=ZzU0I7dEIgLdXZ2L

**I need the code!! - clone the repo (only once)**

- `git clone git@github.com:lucas-nunn/PSM-NeuroAI-Final.git"`

**I want to make some changes of my own!! - make a branch**

- `git checkout -b YOUR_BRANCH`

_..._
_make some changes_
_..._

**what have I changed?? / is there any new stuff I don't have?? / what branch am I on??**

- `git status`

**there is new stuff I don't have!! - get the latest code**

- `git pull origin main`

**I completed a subtask of my branch's overall goal!! - intermittently commit your changes**

- `git add .`
- `git commit -m "short description of what you did"`

**I want my changes to be on Github!! - push your branch**

- `git add .`
- `git commit -m "short description of what you did"`
- `git push origin YOUR_BRANCH`

**OMG my branch is ready for the big league main branch!!! - create a pull request**

1. take a breath
2. PULL THE LATEST CODE NOW!!!!
   - `git pull origin main`
3. merge conflict ... F&%K
   - sorry boss, figure it out
4. final commit
   - `git add .`
   - `git commit -m "short description of what you did"`
5. push!!!
   - `git push origin YOUR_BRANCH`
6. YOOOOOO!!!!!
7. go to the github webpage
8. click on "pull requests"
9. click on "new pull request"
10. base = main
11. compare = your branch
12. review the files
13. write a nice little description
14. submit
15. YOOOOOOOO!!!!!
16. add somebody as a reviewer and spam them
