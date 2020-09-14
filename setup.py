import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="machlearn",
    version="0.0.1",
    author="Daniel Yang",
    author_email="daniel.yj.yang@gmail.com",
    description="Machine Learning Python Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daniel-yj-yang/pyml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
