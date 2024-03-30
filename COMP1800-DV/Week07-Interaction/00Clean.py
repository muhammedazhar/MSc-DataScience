import os

fileNames = os.listdir('.')
print('deleting html files in ' + os.path.abspath('.'))
for filename in fileNames:
    if not filename.endswith('.html'):
        continue
    print(' deleting \'' + filename + '\'')
    os.remove(filename)
