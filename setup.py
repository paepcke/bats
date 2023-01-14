from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "greeknames",
    version = "0.0.1",

    package_dir = {'': 'src'},
    packages = ['greeknames'],
    install_requires = ['nose2>=0.9.2'],     # For testing

    # metadata for upload to PyPI
    author = "Nick Gardner",
    author_email = "gardner.nicholas.d@gmail.com",
    description = "Classifies ancient Greek names to status",
    long_description_content_type = "text/markdown",
    long_description = long_description,
    license = "BSD",
    url = "git@github.com:paepcke/greeknames.git"
)
print("To run tests, type 'nose2'")

