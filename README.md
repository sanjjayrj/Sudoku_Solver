# Sudoku Solver
This project uses deep learning to train a model to recognize digits, and takes the help of OpenCV to process an image of a sudoku to separate out each cells for prediction. The extracted values are then solved using a backtracking algorithm. The solution is plotted on a sudoku template image and output to the user.

- [OpenCV] - open source image-processing framework
- [Flask] - web framework
- [Deep Learning] - Tensorflow and Keras

## Installation
Go the project directory and run the command to install all the required packages. This might take some time.
```bash
pip3 install -r requirements.txt
```
## Running
Go to the directory, and run the command:
```bash
$ python3 app.py
```
Follow the instructions in the website.

   [OpenCV]: <https://opencv.org/>
   [Flask]: <https://flask.palletsprojects.com/en/2.0.x/>
   [Deep Learning]: <https://www.tensorflow.org/>