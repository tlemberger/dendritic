from setuptools import find_packages, setup

setup(
    name="dendritic",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "spacy",
        "nltk",
        "torch",
        "tensorboard",
        "requests",
        "celery",
        # check it out https://github.com/jd/tenacity
        "tenacity",
        "lxml",
        "transformers[torch]==4.36.2",
        "datasets==2.16.1",
        "accelerate",
        "jupyterlab",
        "ipykernel",
        # for jupyter lab
        "ipywidgets>=7.6",
        "jupyterlab-widgets",
        # for plotting
        "plotly==5.18.0",
        # to save images from plotly
        "kaleido",
        # "scipy==1.10.1",
        # "scikit-learn==1.3.0",
        "tqdm==4.65.0",
    ],
)
