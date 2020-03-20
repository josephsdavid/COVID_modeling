import setuptools
def readme():
    with open('README.md') as readme_file:
        return readme_file.read()


setuptools.setup(
    name = "covid-modeling",
    version = "0.0.3",
    description = "COVID-19 Modeling",
    long_description = readme(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/josephsdavid/COVID-modeling",
    maintainer = "David Josephs",
    maintainer_email = "josephsd@smu.edu",
   #$ packages = setuptools.find_packages(exclude = [
    #    "*weights*", "*viz*", "*data*"
   # ]),
    packages = ['comodels'],
    license = 'MIT',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = ['numpy']
)
