import os
import cv2
import math
import time
import numpy as np
from flask import Flask, render_template, request

__author__ = 'andhijes'
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return render_template("uploads.html")


@app.route("/upload", methods=['POST'])
def main():  # Mulai program ENKRIPSI
    # img = cv2.imread("coin.jpg")                     # Masukkan gambar yang ingin dimasukkan pesan tersembunyi

    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        img = file

    word = request.files.getlist("secretWord")  # Masukkan kalimat yang ingin disembunyikan
    count = word.__len__()  # Hitung panjang kalimatnya
    encrypted_word = []  # Deklarasi array kosong untuk dimasukkan pesan yang terenkrip dalam binary (rasio 2:2:3)

    start_time = time.time()  # Mulai stopwatch untuk menghitung waktu operasi algoritme
    for char in word:  # Looping per karakter
        binary_char = chr_to_bin(char)  # Ubah karakter ke binary
        split_char = split_binChar(binary_char)  # Memisahkan hasil binary tadi (rasio 2:2:3) ke dalam bentuk (3:3:3)
        encrypted_word.append([split_char[0], split_char[1], split_char[2]])  # Masukkan hasil string ke array

    cipher_img = encrypt_steganography(img, encrypted_word, count)  # Masukkan hasil gambar ke 'cipher_img'
    cv2.imwrite("encrypted_image.png", cipher_img)  # Membuat file baru (di direktori) dengan nama 'encrypted_image.png'
    print("Total chars inserted:", count)  # Mem-print jumlah kata dalam pesan yang ingin disembunyikan
    print("--- %s seconds ---" % (time.time() - start_time))  # Perlihatkan kompleksitas waktu algoritme

    return render_template("complete.html")


# def upload():
#     target  = os.path.join(APP_ROOT, 'images/')
#     print(target)

#     if not os.path.isdir(target):
#         os.mkdir(target)

#     for file in request.files.getlist("file"):
#         print(file)
#         filename    = file.filename
#         destination = "/".join([target, filename])
#         print(destination)
#         file.save(destination)

def dec_to_bin(x):  # Fungsi mengubah desimal menjadi binary (8-bit)
    return '%08d' % int(bin(x)[2:])


def chr_to_bin(x):  # Fungsi mengubah karakter (char) menjadi binary (7-bit)
    return '%07d' % int(bin(ord(x))[2:])


def split_binChar(string):  # Fungsi memisahkan binary menjadi rasio 2:2:3 (3-bit)
    return '%03d' % int(string[:2]), '%03d' % int(string[2:][:2]), '%03d' % int(string[4:])  # Ganti RASIO disini


def encrypt_steganography(img, word, word_len):  # Fungsi Enkripsi Steganografi (word)
    row, col, ch = img.shape
    k = 0
    i = 0
    w = int(math.floor(word_len / col))
    while (k <= word_len - 1):
        if (i == w):
            col = word_len % col

        for j in range(0, col):
            blue, green, red = img[i][j]
            blue = int(str(int(dec_to_bin(blue)[:5]) * (10 ** 3)), 2)
            green = int(str(int(dec_to_bin(green)[:5]) * (10 ** 3)), 2)
            red = int(str(int(dec_to_bin(red)[:5]) * (10 ** 3)), 2)

            blue = blue + int(word[k][2], 2)
            green = green + int(word[k][1], 2)
            red = red + int(word[k][0], 2)
            k = k + 1

            img.itemset((i, j, 0), blue)
            img.itemset((i, j, 1), green)
            img.itemset((i, j, 2), red)

        i = i + 1

    return img


if __name__ == "__main__":
    # app.run(port=4555, debug=True)
    app.run()