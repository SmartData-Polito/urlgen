import random

baseurl = "www.example.com/position/search?q={},{}&apikey={}&language=it\n"
urls = []

with open(r'./example-dataset.txt', 'w') as fp:
    for i in range(20000):
        fp.write(baseurl.format(random.randint(454365384, 464365384)/10000000, random.randint(86266474, 87266474)/10000000, random.getrandbits(128)))

