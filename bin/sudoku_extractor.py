import cv2, imutils, os
import numpy as np
import solve_sudoku as sudoku
from tensorflow import keras
from skimage.segmentation import clear_border
from imutils.perspective import four_point_transform
from tensorflow.keras.preprocessing.image import img_to_array
# from sudoku import Sudoku
class GetPuzzle:
    def __init__(self):
        self.MODEL_FOLDER="./models/"
    def find_puzzle(self, image):
        # using adaptive thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7,7), 3)
        th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        th = cv2.bitwise_not(th)
        
        #using the output from adapative thresholding to find contours
        contours = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        puzzle_shape = None
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4:
                puzzle_shape = approx
                break
        
        if puzzle_shape is None:
            raise Exception(("Could not find puzzle!!!"))
        puzzle = four_point_transform(image, puzzle_shape.reshape(4,2))
        warped = four_point_transform(gray, puzzle_shape.reshape(4,2))
        return (puzzle,warped)

    def get_digits(self, cell):
        threshold = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) [1]
        # in each cell, the following function removes edges in each cell
        threshold = clear_border(threshold)
        # cv2.imshow("thresholding", threshold)
        # cv2.waitKey(0)
        contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        # if contour doesnt exist
        if len(contours) == 0:
            return None
        # else return largest contour in cell
        max_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(threshold.shape, dtype="uint8")
        cv2.drawContours(mask, [max_contour], -1, 255, -1)
        
        (h,w) = threshold.shape
        percentFilled = cv2.countNonZero(mask) / float(w*h)
        # making a few assumptions now
        # if less than 5% of mask filled, then its noise
        if percentFilled < 0.03:
            return None
        # else apply mask
        digit_img = cv2.bitwise_and(threshold, threshold, mask=mask)
        return digit_img

def sudoku_extractor(img_path):
    get_puzzle = GetPuzzle()
    model = keras.models.load_model(os.path.join(get_puzzle.MODEL_FOLDER, "sudoku_model_2"))
    img = cv2.imread(img_path)
    img = imutils.resize(img, width=600)
    (puzzle, warped) = get_puzzle.find_puzzle(img)
    # the warped image is used for processing and,
    # outputs are plotted on the "puzzle" image
    board = np.zeros((9,9), dtype="int")
    # we divide the warped image, which was the grayscale image on which mask was applied,
    # into 9x9 grid like sudoku, to get each individual cells
    X = warped.shape[1] // 9
    Y = warped.shape[0] // 9
    locations = []
    # Traversing the grid and predicting the digits by cell number.
    for y in range(0,9):
        row = []
        for x in range(0, 9):
            x1 = x * X
            y1 = y * Y
            x2 = (x+1) * X
            y2 = (y+1) * Y
            row.append((x1,y1, x2,y2))
            cell = warped[y1:y2, x1:x2]
            digit = get_puzzle.get_digits(cell)
            if digit is not None:
                digit_image = cv2.resize(digit, (28,28))
                digit_image = digit_image.astype("float") / 255.0
                # converting to numpy array instance
                digit_image = img_to_array(digit_image)
                # since we need to specify that the image is grayscale, we expand the input to model
                digit_image = np.expand_dims(digit_image, axis=0)

                prediction = model.predict(digit_image).argmax(axis=1)[0]
                board[y,x] = prediction

        locations.append(row)
    return board, warped

def show_solution(solution):
    # pass numpy array as solution
    template = cv2.imread("template.jpg")
    cv2.imshow("template", template)
    cv2.waitKey(0)
    X = template.shape[1] // 9
    Y = template.shape[0] // 9
    for y in range(0,9):
        for x in range(0, 9):
            x1 = x * X
            y1 = y * Y
            x2 = (x+1) * X
            y2 = (y+1) * Y
            x_center = (x2-x1)/2 + x1
            y_center = (y2-y1)/2 + y1
            cv2.putText(template, str(solution[y][x]), x_center, y_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            
def solver(board):
    return sudoku.solve(board)

if __name__=="__main__":
    sudoku_extractor("../examples/sudoku.jpg")