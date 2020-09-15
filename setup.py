# https://packaging.python.org/guides/distributing-packages-using-setuptools/#setup-py
# https://github.com/pypa/sampleproject/blob/master/setup.py

import setuptools
import machlearn

README_file = "README.rst" # "README.md"
with open(README_file, "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="machlearn",
    version=machlearn.__version__,
    author="Daniel Yang",
    author_email="daniel.yj.yang@gmail.com",
    description="Machine Learning Python Library",
    long_description=long_description,
    long_description_content_type="text/x-rst",  # "text/markdown",
    url="https://github.com/daniel-yj-yang/pyml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # https://pypi.org/classifiers/
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.18.5',
        'seaborn>=0.10.1',
        'matplotlib>=3.2.2',
    ],
    python_requires='>=3.6',
)
