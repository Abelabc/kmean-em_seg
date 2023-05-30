from scipy.spatial.distance import cosine
from gaborpy import gaborcls

fea=gaborcls('mapB.bmp')
print(fea.shape)
filename = 'kmean.png'
newfea = gaborcls(filename)
print(newfea.shape)
tmp=cosine(newfea,fea)
print(newfea)
print(tmp)

filename = 'em.png'
newfea = gaborcls(filename)
tmp=cosine(newfea,fea)
print(newfea)
print(tmp)