import os

os.system('python3 setup.py bdist_wheel')


path = 'dist'

for filename in os.listdir(path):
    filename_splitext = os.path.splitext(filename)
    if filename_splitext[1] in ['.rtf', '.py', '.whl', '.out']:
        os.rename(os.path.join(path, filename), 
                os.path.join(path, filename_splitext[0] +  '.zip'))
print()
print('Wheel for custom packages converted to .zip and sent to workers; submit job when ready!')
print()