import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name = 'sentida',
    version = '0.3.4',
    author = "Esben Kran, Søren Orm",
    author_email = "contact@esbenkc.com",
    description = "The Sentida Danish sentiment analysis package",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/esbenkc/emma",
    packages = ['sentida'],
    # packages=setuptools.find_packages(),
    classifiers = [
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
    package_data = {"sentida":['aarup.csv', 'intensifier.csv'],},
)
