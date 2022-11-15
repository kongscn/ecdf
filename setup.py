import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ecdf",
    version="0.6.1",
    author="Shel Kong",
    author_email="kongscn@gmail.com",
    description="Visualize dataframes with echarts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kongscn/ecdf",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pandas"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
