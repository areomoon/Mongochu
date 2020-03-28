from urllib.request import urlretrieve

url = 'https://cnndmpretrained.s3-ap-southeast-1.amazonaws.com/Mongochu-master.zip'
dst = 'Mongochu-master.zip'
urlretrieve(url, dst)
print('{} is downloaded'.format(dst))