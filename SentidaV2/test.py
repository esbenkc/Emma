from timeit import timeit

setup = """
from sentida.sentida2 import Sentida2, sentida2_examples
SV2 = Sentida2()
"""
time1 = []
time2 = []
iterations = 1000
for i in list(range(iterations)):
    print("iteration: ", i)
    time1.append(
            timeit("SV2.sentida2(\"Jeg er rigtig glad. Jeg hader dig!!!!\", output = \"by_sentence_total\", speed = \"fast\")", setup = setup, number = 10))
    time2.append(
            timeit("SV2.sentida2(\"Jeg er rigtig glad. Jeg hader dig!!!!\", output = \"by_sentence_total\", speed = \"normal\")", setup = setup, number = 10))


print(sum(time1)/len(time1))
print(sum(time2)/len(time2))

from sentida.sentida2 import Sentida2, sentida2_examples
SV = Sentida2()
print(SV.sentida2("Jeg er rigtig glad. Død.", output = "by_sentence_total", normal = True, speed = "fast"))