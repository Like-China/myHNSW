'''
Descripttion: Rika's code
version: 1.0.0
Author: Rika
Date: 2024-07-15 11:01:03
LastEditors: Rika
LastEditTime: 2024-07-15 11:01:08
'''
import sys
# python test.py 55 44 88
if __name__=='__main__':
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    print("Parameter")
    print('\t type:',type(sys.argv),'\n\t value:', sys.argv)
    print("$$$$$$$$$$")
    print(sys.argv[0])
    print(sys.argv[-1])
    print("##########")
    print(n+m)
    print("**********")
    for item in sys.argv:
        print(type(item))
# python test.py 11111 22222 88888

'''
"args": ["11111","22222","88888"]
'''

