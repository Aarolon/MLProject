import preprocessing
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

try:
    filename = str(sys.argv[1])
    preprocessing.getDfFromJSON(filename).to_csv(filename[:-8]+'.csv',index=False)
except(KeyError, IndexError):
    print("No argument given, or file not found")
