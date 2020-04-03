import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name = 'sentida',
    version = '0.4.2',
    author = "Esben Kran, Søren Orm",
    author_email = "contact@esbenkc.com",
    description = "The Sentida Danish sentiment analysis package",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/esbenkc/emma",
    # packages = ['sentida'],
    licence = 'MIT',
    packages=setuptools.find_packages(),
    include_package_data = True,
    keywords = 'Natural Language Processing Sentiment Analysis',
    classifiers = [
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
         "Development Status :: 4 - Beta",
         "Intended Audience :: Science/Research",
         "Topic :: Scientific/Engineering"
    ],
    package_data = {"sentida":['aarup.csv', 'intensifier.csv'],},
)
