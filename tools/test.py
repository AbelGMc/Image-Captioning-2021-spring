import os
import sys
print(os.getcwd())
print(sys.path)
sys.path.append((os.getcwd()))
print(os.path.dirname(os.getcwd()))
import captioning
