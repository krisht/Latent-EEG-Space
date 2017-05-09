import os

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import numpy as np
from sklearn import neighbors

saver = None
sess = None

loss_mem = []


bckg_num = 0
artf_num = 1
eybl_num = 2
gped_num = 3
spsw_num = 4
pled_num = 5


def get_loss(loss_mem):
    print(loss_mem)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(loss_mem, 'r--')
    plt.xlabel("100 Iterations")
    plt.ylabel("Average Loss in 100 Iterations")
    plt.title("Iterations vs. Average Loss")
    plt.show()


def save_model(sess, saver):
    save_path = saver.save(sess, "./latest_weights.ckpt")
    print("Model saved in file: %s" % save_path)


class BrainNet:
    def __init__(self, sess, input_shape=[None, 71, 125], num_output=64, num_classes=6, restore_dir=None):

        path = os.path.abspath('/media/krishna/My Passport/DataForUsage/labeled')

        self.ARTF = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
                     'artf' in os.path.join(dp, f) and 'npz' in os.path.join(dp, f)]
        self.BCKG = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
                     'bckg' in os.path.join(dp, f) and 'npz' in os.path.join(dp, f)]
        self.SPSW = ['/media/krishna/My Passport/DataForUsage/labeled/session10/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session11/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session112/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session114/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session115/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session116/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session118/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session119/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session12/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session121/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session122/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session123/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session216/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session219/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session220/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session222/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session225/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session226/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session227/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session228/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session229/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session230/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session231/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session232/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session233/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session54/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session55/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session57/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session59/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session73/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session76/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session78/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session79/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session81/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session83/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session85/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session87/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session89/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session9/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session91/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session92/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session94/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session95/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session96/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session99/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session127/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session129/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session130/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session131/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session132/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session133/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session135/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session136/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session137/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session138/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session139/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session14/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session140/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session141/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session142/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session143/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session144/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session17/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session234/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session255/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session276/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session298/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session32/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session358/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session53/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session80/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session146/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session147/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session148/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session149/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session150/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session152/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session154/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session155/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session157/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session166/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session168/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session178/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session179/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session180/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session181/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session185/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session19/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session197/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session199/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session2/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session200/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session201/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session203/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session205/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session206/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session207/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session212/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session213/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session235/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session237/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session24/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session241/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session244/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session245/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session246/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session247/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session248/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session249/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session25/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session254/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session256/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session258/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session259/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session261/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session262/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session264/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session269/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session27/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session270/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session274/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session277/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session279/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session28/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session280/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session281/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session282/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session283/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session284/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session285/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session287/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session288/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session289/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session29/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session291/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session295/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session296/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session297/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session299/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session30/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session300/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session301/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session302/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session304/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session305/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session306/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session307/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session308/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session309/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session31/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session310/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session314/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session317/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session319/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session320/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session321/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session322/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session323/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session324/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session325/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session326/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session327/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session328/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session329/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session33/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session331/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session332/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session333/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session334/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session335/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session34/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session359/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session36/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session360/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session363/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session364/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session365/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session369/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session371/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session376/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session39/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session46/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session48/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session49/spsw0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session50/spsw0.npz']
        self.PLED = ['/media/krishna/My Passport/DataForUsage/labeled/session120/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session232/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session233/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session139/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session140/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session141/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session181/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session244/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session245/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session247/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session248/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session299/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session300/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session301/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session31/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session317/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session319/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session320/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session322/pled0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session324/pled0.npz']
        self.GPED = ['/media/krishna/My Passport/DataForUsage/labeled/session119/gped0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session121/gped0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session122/gped0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session123/gped0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session125/gped0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session168/gped0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session181/gped0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session283/gped0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session284/gped0.npz']
        self.EYBL = ['/media/krishna/My Passport/DataForUsage/labeled/session0/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session1/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session10/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session104/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session11/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session112/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session114/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session115/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session116/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session117/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session118/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session119/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session12/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session120/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session121/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session122/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session123/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session125/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session215/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session216/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session217/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session218/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session219/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session220/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session221/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session222/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session223/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session224/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session225/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session226/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session227/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session228/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session229/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session230/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session231/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session232/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session233/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session54/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session55/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session56/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session57/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session58/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session59/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session60/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session61/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session63/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session64/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session65/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session66/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session73/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session74/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session75/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session76/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session77/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session78/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session79/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session81/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session82/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session83/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session84/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session85/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session86/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session87/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session88/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session89/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session9/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session90/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session91/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session92/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session93/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session94/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session95/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session96/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session97/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session99/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session127/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session129/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session13/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session130/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session131/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session132/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session133/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session134/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session135/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session136/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session137/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session138/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session139/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session14/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session140/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session141/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session142/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session143/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session144/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session126/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session145/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session17/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session214/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session234/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session255/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session276/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session298/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session32/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session358/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session53/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session80/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session146/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session147/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session148/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session149/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session150/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session151/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session152/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session153/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session154/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session155/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session156/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session157/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session161/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session162/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session164/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session165/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session166/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session168/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session178/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session179/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session180/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session181/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session185/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session187/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session19/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session196/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session199/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session200/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session201/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session203/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session205/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session206/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session207/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session209/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session210/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session212/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session213/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session235/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session236/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session237/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session238/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session24/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session241/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session242/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session243/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session244/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session245/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session246/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session247/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session248/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session249/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session25/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session250/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session252/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session253/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session254/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session256/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session257/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session258/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session259/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session260/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session261/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session262/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session263/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session264/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session268/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session269/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session27/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session270/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session271/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session272/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session273/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session274/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session275/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session277/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session279/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session28/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session280/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session281/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session282/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session283/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session284/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session285/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session287/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session289/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session29/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session291/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session292/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session295/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session296/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session297/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session299/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session30/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session300/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session301/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session302/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session304/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session305/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session306/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session307/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session308/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session309/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session31/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session310/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session313/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session314/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session317/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session318/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session319/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session320/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session321/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session322/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session323/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session324/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session325/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session326/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session327/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session328/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session329/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session33/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session330/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session331/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session332/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session333/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session334/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session335/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session34/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session35/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session359/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session36/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session360/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session363/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session364/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session365/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session366/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session367/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session369/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session37/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session371/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session375/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session376/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session38/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session39/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session40/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session44/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session45/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session46/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session47/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session48/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session49/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session50/eybl0.npz',
                     '/media/krishna/My Passport/DataForUsage/labeled/session52/eybl0.npz']

        self.sess = sess
        self.num_classes = num_classes
        self.num_output = num_output
        self.input_shape = input_shape
        self.inference_input = tf.placeholder(tf.float32, shape=input_shape)
        self.inference_model = self.get_model(self.inference_input, reuse=False)
        if restore_dir is not None:
            dir = tf.train.Saver()
            dir.restore(self.sess, restore_dir)

        print(len(self.ARTF))
        print(len(self.BCKG))
        print(len(self.EYBL))
        print(len(self.SPSW))
        print(len(self.PLED))
        print(len(self.GPED))
        self.load_files()

    def triplet_loss(self, alpha):
        self.anchor = tf.placeholder(tf.float32, shape=self.input_shape)
        self.positive = tf.placeholder(tf.float32, shape=self.input_shape)
        self.negative = tf.placeholder(tf.float32, shape=self.input_shape)
        self.anchor_out = self.get_model(self.anchor, reuse=True)
        self.positive_out = self.get_model(self.positive, reuse=True)
        self.negative_out = self.get_model(self.negative, reuse=True)
        with tf.variable_scope('triplet_loss'):
            pos_dist = tf.reduce_sum(tf.square(self.anchor_out - self.positive_out))
            neg_dist = tf.reduce_sum(tf.square(self.anchor_out - self.negative_out))

            basic_loss = tf.maximum(0., alpha + pos_dist - neg_dist)
            loss = tf.reduce_mean(basic_loss)
            return loss

    def load_files(self):
        print("Loading New Source Files...")
        self.bckg = np.load(random.choice(self.BCKG))['arr_0']
        self.eybl = np.load(random.choice(self.EYBL))['arr_0']
        self.artf = np.load(random.choice(self.ARTF))['arr_0']
        self.gped = np.load(random.choice(self.GPED))['arr_0']
        self.pled = np.load(random.choice(self.PLED))['arr_0']
        self.spsw = np.load(random.choice(self.SPSW))['arr_0']

    def get_triplets(self):

        choices = ['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf']
        neg_choices = choices

        choice = random.choice(choices)

        if choice in neg_choices: neg_choices.remove(choice)

        if choice == 'bckg':
            ii = random.randint(0, len(self.bckg) - 1)
            a = self.bckg[ii]

            jj = random.randint(0, len(self.bckg) - 1)
            p = self.bckg[jj]

        elif choice == 'eybl':
            ii = random.randint(0, len(self.eybl) - 1)
            a = self.eybl[ii]

            jj = random.randint(0, len(self.eybl) - 1)
            p = self.eybl[jj]

        elif choice == 'gped':
            ii = random.randint(0, len(self.gped) - 1)
            a = self.gped[ii]

            jj = random.randint(0, len(self.gped) - 1)
            p = self.gped[jj]

        elif choice == 'spsw':
            ii = random.randint(0, len(self.spsw) - 1)
            a = self.spsw[ii]

            jj = random.randint(0, len(self.spsw) - 1)
            p = self.spsw[jj]

        elif choice == 'pled':
            ii = random.randint(0, len(self.pled) - 1)
            a = self.pled[ii]

            jj = random.randint(0, len(self.pled) - 1)
            p = self.pled[jj]

        else:
            ii = random.randint(0, len(self.artf) - 1)
            a = self.artf[ii]

            jj = random.randint(0, len(self.artf) - 1)
            p = self.artf[jj]

        neg_choice = random.choice(neg_choices)

        if neg_choice == 'bckg':
            ii = random.randint(0, len(self.bckg) - 1)
            n = self.bckg[ii]
        elif neg_choice == 'eybl':
            ii = random.randint(0, len(self.eybl) - 1)
            n = self.eybl[ii]
        elif neg_choice == 'gped':
            ii = random.randint(0, len(self.gped) - 1)
            n = self.gped[ii]
        elif neg_choice == 'spsw':
            ii = random.randint(0, len(self.spsw) - 1)
            n = self.spsw[ii]
        elif neg_choice == 'pled':
            ii = random.randint(0, len(self.pled) - 1)
            n = self.pled[ii]
        else:
            ii = random.randint(0, len(self.artf) - 1)
            n = self.artf[ii]

        a = np.expand_dims(a, 0) * 10e4
        p = np.expand_dims(p, 0) * 10e4
        n = np.expand_dims(n, 0) * 10e4

        return np.vstack([a, p, n])

    def get_model(self, input, reuse=False):
        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(seed=random.random(),
                                                                                     uniform=True),
                            weights_regularizer=slim.l2_regularizer(0.05), reuse=reuse):
            net = tf.expand_dims(input, axis=3)
            net = slim.layers.conv2d(net, num_outputs=32, kernel_size=4, scope='conv1', trainable=True)
            net = slim.layers.max_pool2d(net, kernel_size=3, scope='maxpool1')
            net = slim.layers.conv2d(net, num_outputs=64, kernel_size=5, scope='conv2', trainable=True)
            net = slim.layers.max_pool2d(net, kernel_size=3, scope='maxpool2')
            net = slim.layers.flatten(net, scope='flatten')
            net = slim.layers.fully_connected(net, 256, scope='fc1', trainable=True)
            net = slim.layers.fully_connected(net, 1024, scope='fc2', trainable=True)
            net = slim.layers.fully_connected(net, self.num_output, activation_fn=None, weights_regularizer=None,
                                              scope='output')
            return net

    def train_model(self, learning_rate, keep_prob, batch_size, train_epoch, outdir=None):
        loss = self.triplet_loss(alpha=0.5)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optim = self.optimizer.minimize(loss=loss)
        self.sess.run(tf.global_variables_initializer())

        count = 0
        ii = 0

        for epoch in range(0, train_epoch):
            ii = 0
            count = 0
            full_loss = 0
            while ii <= batch_size:
                ii += 1
                feeder = self.get_triplets()

                anchor = feeder[0]
                anchor = np.expand_dims(anchor, 0)
                positive = feeder[1]
                positive = np.expand_dims(positive, 0)
                negative = feeder[2]
                negative = np.expand_dims(negative, 0)

                temploss = self.sess.run(loss, feed_dict={self.anchor: anchor, self.positive: positive,
                                                          self.negative: negative})

                if temploss == 0:
                    print(temploss)
                    ii -= 1
                    count += 1
                    continue

                full_loss += temploss

                if ii % 100 == 0:
                    loss_mem.append(full_loss /(100+ count))
                    full_loss = 0

                _, anchor, positive, negative = self.sess.run([self.optim,
                                                               self.anchor_out,
                                                               self.positive_out,
                                                               self.negative_out], feed_dict={self.anchor: anchor,
                                                                                              self.positive: positive,
                                                                                              self.negative: negative})

                d1 = np.linalg.norm(positive - anchor)
                d2 = np.linalg.norm(negative - anchor)

                print("Epoch: ", epoch, "Iteration:", ii, ", Loss: ", temploss, ", Positive Diff: ", d1,
                      ", Negative diff: ", d2)
                print("Iterations skipped: ", count)
            self.validate()
            self.load_files()

    def get_sample(self, choice = None, size = 1):
        data_list=[]
        class_list=[]

        for ii in range(0, size):

            if choice == None:
                choices = ['bckg', 'eybl', 'gped', 'spsw', 'pled', 'artf']
                choice = random.choice(choices)

            if choice == 'bckg':
                ii = random.randint(0, len(self.bckg) - 1)
                data_list.append(self.bckg[ii])
                class_list.append(bckg_num)

            elif choice=='eybl':
                ii = random.randint(0, len(self.eybl) - 1)
                data_list.append(self.eybl[ii])
                class_list.append(eybl_num)
            elif choice =='gped':
                ii = random.randint(0, len(self.gped) - 1)
                data_list.append(self.gped[ii])
                class_list.append(gped_num)
            elif choice == 'spsw':
                ii = random.randint(0, len(self.spsw) -1)
                data_list.append(self.spsw[ii])
                class_list.append(spsw_num)
            elif choice=='pled':
                ii = random.randint(0, len(self.spsw) - 1)
                data_list.append(self.pled[ii])
                class_list.append(pled_num)
            else:
                ii = random.randint(0, len(self.artf) -1)
                data_list.append(self.artf[ii])
                class_list.append(artf_num)


        return data_list, class_list



    def validate(self):
        self.load_files()

        inputs, classes = self.get_sample(size=10000)

        vector_inputs = self.sess.run(self.inference_model, feed_dict={self.inference_input: inputs})

        knn = neighbors.KNeighborsClassifier()
        knn.fit(vector_inputs, classes)

        val_inputs, val_classes = self.get_sample(size=1000)


        pred_class = knn.predict(val_inputs)

        percentage = len([i for i, j in zip(val_classes, pred_class) if i==j])

        print("Validation Results: %d out of 1000 correct" %  percentage )

if __name__ == "__main__":
    try:
        sess = tf.Session()
        model = BrainNet(sess=sess, restore_dir='previous_run/latest_weights.ckpt')
        print('Krisna you\'re a dolt')
    except (KeyboardInterrupt, SystemError, SystemExit):
        save_model(sess, saver)
        get_loss(loss_mem)
    save_model(sess, saver)
    get_loss(loss_mem)

#
# sess = tf.Session()
# model = BrainNet(sess=sess, restore_dir='./latest_weights.ckpt')
