# GroWell ML Server

## Set Up the project
1. clone the repository
2. clone from tensorflow model `git clone --depth 1 https://github.com/tensorflow/models`
3. write this command
```bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```
4. update your Numpy version `pip install numpy --upgrade`
5. You should now able to run the app