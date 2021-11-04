@ECHO OFF

pip3 install tensorflow==2.6.0
pip3 install Cython==0.29.24
pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
python -m pip install ./research/object_detection/packages/tf2
pip3 install keras==2.6.*