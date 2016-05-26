#!/usr/bin/env python3

import multitron

def main():
    m = multitron.Multitron()

    print("-----------------------LOAD DATA----------------")
    m.loadData()

    print("----------------TRAIN DATA----------------------")
    m.train()

    print("----------------VALIDATE DATA-------------------")
    m.validate()

    print("--------------------TEST DATA-------------------")
    m.test()

    # m.printTrainingDataInfo();

    return

main()
