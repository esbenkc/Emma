import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='SentidaV2',
    version='0.2.1',
    scripts=['SentidaV2.py'],
    author="Søren Orm, Esben Kran",
    author_email="contact@esbenkc.com",
    description="A Danish sentiment analysis package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/esbenkc/emma",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
)
