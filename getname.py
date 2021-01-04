from imfloader import get_dataset_struct

key2des = get_dataset_struct('IFS')['Indicator']

while 1:
    i = input()
    print(key2des[i]+'\n')
