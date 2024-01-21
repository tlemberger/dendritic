from setuptools import find_packages, setup

setup(
    name="dendritic",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "spacy",
        "nltk",
        # "torch==1.12.1+cu116q",
        "tensorboard",
        "requests",
        "celery",
        # check it out https://github.com/jd/tenacity
        "tenacity",
        "lxml",
        "transformers[torch]",
        "datasets",
        "accelerate",
        "jupyterlab",
        "ipykernel>5.5.0",
        # for jupyter lab
        "ipywidgets>=7.6",
        "jupyterlab-widgets",
        # for plotting
        "plotly==5.18.0",
        # to save images from plotly
        "kaleido",
        # "scipy==1.10.1",
        "scikit-learn==1.3.0",
        "tqdm==4.65.0",
    ],
)
