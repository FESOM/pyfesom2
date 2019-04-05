# -*- coding: utf-8 -*-

import argparse

def pfinterp():
     
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    # args.func(args)
    print(args.name)



# parser.set_defaults(func=pfinterp)  


                 



if __name__ == '__main__':
    # args = parser.parse_args()
    # args.func(args)
    pfinterp()