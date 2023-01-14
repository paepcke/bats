# Why a project template?

This project is a starter template for new student
projects with Andreas Paepcke. It sets up a file 
structure with a few files that illustrate style
and some Python/Unittest facilities.

The template contains a small example project that
pretends to predict ancient Greek professions from 
names.

Features that are demonstrated are:

    o Class/method setup
    o Unit testing in an object-oriented context
    o Argparse
    o Creation of executable script using #!
    o Use of setup.py

# Creating new project from this template

    o Create your own repo on github, and note its clone URL,
        say git@github.com:john_doe/your_project.git
    o On your machine, clone this project_template:
        `git clone git@github.com:paepcke/project_template.git`
    o cd project_template
    o git push --mirror git@github.com:john_doe/your_project.git
    o cd ..
    o git clone git@github.com:john_doe/your_project.git
    o cd your_project

You are now in your project root directory (<proj-root>). 
    
# Setup and install the example

After cloning:
    
    o cd your_project
    o Optionally create an anaconda environment (highly recommended)
         conda new -s your_project
    o python setup.py install
  

Replace the .py and test data files as needed.
