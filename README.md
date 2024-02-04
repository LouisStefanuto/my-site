# Welcome to my website 👋

This repo defines all you see on my personal website here: <https://louisstefanuto.github.io/my-site/>.

## 🤖 Installation

### With Poetry

All dependencies are listed in the `pyproject.toml`. To create an environment, go to the project directory, install dependencies and activate your venv:

```console
poetry install
poetry shell
```

### With conda / pip

Create a Conda environment and install dependencies from the `requirements.txt` file:

```console
conda create --name env_mysite python=3.11
conda activate env_mysite
pip install -r requirements.txt  
```

## 🧪 Host the site locally for dev

Launch the server:

```console
mkdocs serve
```

## 🚀 Deployment

Github actions deploy the site o nGithub Pages for you. If you want to deploy manually, use:

```console
mkdocs gh-deploy
```
