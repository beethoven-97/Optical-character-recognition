import os
import numpy as np
import random
import cv2

words_path = 'C:/Users/User/PycharmProjects/ocr/words.txt'
images_path = 'C:/Users/User/PycharmProjects/ocr/words/'


class Image:
    def __init__(self, text, path):
        self.text, self.path = text, path


class Data_load:
    def __init__(self):
        # define variables
        self.images, f, maxtextlen = [], open(words_path), 31
        lines = f.readlines()

        # read images file
        for line in lines[:80]:
            # ignore empty lines and line begin with '#'
            if not line or line[0] == '#':
                continue

            # split line
            line_words = line.strip().split(' ')

            # get image name
            image_name_split = line_words[0].split('-')
            image_name = images_path + image_name_split[0] + '/' + '-'.join(image_name_split[:2]) + '/' + line_words[
                0] + '.png'

            # get texts with length<=32
            text = self.truncate_label(''.join(line_words[-1]), maxtextlen)

            # check empty image
            if not os.path.getsize(image_name):
                continue

            # insert image into images
            self.images.append(Image(text, image_name))

        f.close()

    def truncate_label(self, text, maxtextlen):
        cost = 0
        for i in range(len(text)):
            if i > 0 and text[i - 1] == text[i]:
                cost += 2
            else:
                cost += 1
            if cost > maxtextlen:
                return text[:i]
        return text
