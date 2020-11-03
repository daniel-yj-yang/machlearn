[Date]

2020-11-03


[Fix]

install-cmake-3.18.4-Linux-x86_64.sh was needed for Python 3.8, as Travis CI's CMake was 3.12.4 and generated the following error message.


[Bug]

Installed /home/travis/virtualenv/python3.8.0/lib/python3.8/site-packages/pyspark-3.0.1-py3.8.egg
Searching for xgboost
Reading https://pypi.org/simple/xgboost/
Downloading https://files.pythonhosted.org/packages/2a/8f/3b21e1a65ef54b1f536f5e31e6bae1eff72ab9e08655743a7a5435ce71fc/xgboost-1.2.0.tar.gz#sha256=e5f4abcd5df6767293f31b7c58d67ea38b2689641a95b2cd8ca8935295097a36
Best match: xgboost 1.2.0
Processing xgboost-1.2.0.tar.gz
Writing /tmp/easy_install-8efltxzu/xgboost-1.2.0/setup.cfg
Running xgboost-1.2.0/setup.py -q bdist_egg --dist-dir /tmp/easy_install-8efltxzu/xgboost-1.2.0/egg-dist-tmp-oa_nt9ob
warning: no previously-included files matching '*.py[oc]' found anywhere in distribution
INFO:XGBoost build_ext:Building from source. /tmp/easy_install-8efltxzu/lib/libxgboost.so
INFO:XGBoost build_ext:Run CMake command: ['cmake', 'xgboost', '-GUnix Makefiles', '-DUSE_OPENMP=1', '-DUSE_CUDA=0', '-DUSE_NCCL=0', '-DBUILD_WITH_SHARED_NCCL=0', '-DHIDE_CXX_SYMBOLS=1', '-DUSE_HDFS=0', '-DUSE_AZURE=0', '-DUSE_S3=0', '-DPLUGIN_LZ4=0', '-DPLUGIN_DENSE_PARSER=0']
CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
  CMake 3.13 or higher is required.  You are running version 3.12.4
  