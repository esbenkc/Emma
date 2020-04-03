from timeit import timeit

from sentida import Sentida

setup = """
from sentida.sentida2 import Sentida2, sentida2_examples
SV = Sentida()
"""

time1 = []
time2 = []

iterations = 1000

for i in list(range(iterations)):
    print("iteration: ", i)
    time1.append(
            timeit("SV.sentida(\"Jeg er rigtig glad. Jeg hader dig!!!!\", output = \"by_sentence_total\", speed = \"fast\")", setup = setup, number = 10))
    time2.append(
            timeit("SV.sentida(\"Jeg er rigtig glad. Jeg hader dig!!!!\", output = \"by_sentence_total\", speed = \"normal\")", setup = setup, number = 10))

print(sum(time1)/len(time1))
print(sum(time2)/len(time2))

SV = Sentida()
print(SV.sentida("Jeg er rigtig glad. Død.", output = "by_sentence_total", normal = True, speed = "fast"))
