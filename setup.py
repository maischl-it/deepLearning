import setuptools

description = open("readme.md").read()

setuptools.setup(
    name="DeepLearningHouseprices",
    version="1.0",
    packages=setuptools.find_packages(),
    long_description=description,
    install_requires=[
        'pandas',
        'sklearn',
        'matplotlib',
        'keras',
        'tensorflow',
        'jupyter'
    ],
    python_requires=">=3.8"
)
