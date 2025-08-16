# Introduction To Machine Learning - WebApplication
This project contains a simple Flask REST-API with a webapplication for testing your machine learning models. The user interface will give you the ability to draw symbols and send a classification request to the REST-API.

The front-end application has already been finished, it is your task to implement the `classify_sample()` method in `server.py` with your best classifier.

## Project structure
The following folders are used within the project:
- `static`: contains a simple webapplication that you can use to draw symbols and test your REST-API. You should not edit the files in this folder.
- `classes`: place your (well-crafted) classes here.
- `models`: this is where you place your exported machine learning and preprocessing models.
- `temp`: this folder is used to store the uploaded images (temporarily). Don't remove or place files here.

## Installing and starting the project
We have used [Poetry](https://python-poetry.org/) for creating a Python project with specific dependencies (see `pyproject.toml`). This is nowadays the standard way of dealing with professional Python projects. If you want to understand the difference between pip, conda and poetry, please read [this article](https://blog.inedo.com/python/managing-python-packages/). Please make sure poetry is installed on your system. This is how you can work with your poetry project:

1. Installing all dependencies: `poetry install`
2. Adding your own custom libraries `poetry add <name_of_package>`. Please note: the important libraries for machine learning, data analysis and webdevelopment have already been added to the project. In most cases you don't have add any packages.
3. Start the webserver by running `poetry run python3 server.py`. This will start the Flask webserver. If you browse to the URL (shown in the terminal) you should see the webapplication.

## Your task
Please implement the `classify_sample()` method in `server.py`. This method receives the drawn image and should return a dictionary object (in JSON) with the predicted label and the accuracy of the prediction.

## Questions
If you have any questions about the project or the code, please contact Evert Duipmans <a href="mailto:e.f.duipmans@saxion.nl">(e.f.duipmans@saxion.nl)</a>.