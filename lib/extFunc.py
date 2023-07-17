import numpy as np
import json

def saveArrToFile(filename, arr):
    fp = open(f"{filename}", "wb")
    np.save(fp, arr)
    fp.close()

def loadArrFromFile(filename):
    fp = open(f"{filename}", "rb")
    arr = np.load(fp)
    fp.close()

    return arr

def writeToFile(filename, info):
    fp = open(f"{filename}", "w")
    fp.writelines([f"{x}\n" for x in info])
    fp.close()

def readFromFile(filename):
    fp = open(f"{filename}", "r")
    info = fp.readlines()
    fp.close()

    return info

def save_dictionary(dictionary, file_path):
    with open(file_path, 'w') as file:
        json.dump(dictionary, file)

def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        dictionary = json.load(file)
    return dictionary

