# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.18.5',
        'seaborn>=0.10.1',
        'matplotlib>=3.2.2',
    ],
    python_requires='>=3.6',
)
