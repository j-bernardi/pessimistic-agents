import timeit

from collections import deque
import itertools

def create_a_batch(batch_size=64):

    a = ["a"] * batch_size
    b = ["b"] * batch_size
    c = ["c"] * batch_size


    return a, b, c

def append_as_tuples_naive(eps=200):
    memory = deque(maxlen=100000)
    for _ in range(eps):
        tup = create_a_batch()
        for i in range(len(tup[0])):
            memory.append((tup[0][i], tup[1][i], tup[2][i]))
    return memory

def append_as_tuples(eps=200):

    memory = deque(maxlen=100000)
    for _ in range(eps):
        tup = create_a_batch()
        memory.extend(tuple(zip(*tup)))
    return memory

def append_as_array(eps=200):
    memory = [deque(maxlen=100000), deque(maxlen=100000), deque(maxlen=100000)]
    for _ in range(eps):
        tup = create_a_batch()
        for i in range(3):
            memory[i].extend(tup[i])
    return memory

test_num = 1000

naive_tup_time = timeit.timeit(append_as_tuples_naive, number=test_num) 
tup_time = timeit.timeit(append_as_tuples, number=test_num)
arr_time = timeit.timeit(append_as_array, number=test_num)

n_as_tups = append_as_tuples_naive()
as_tups = append_as_tuples()
as_arrays = append_as_array()

print("NAIVE TUP TIME", naive_tup_time, "Very slow")
print("TUP TIME", tup_time, "1/4 speedup")
print(list(itertools.islice(as_tups, 0, 5)))
print("ARR TIME", arr_time, "faster by around 1/3 but less conceptually sound")
for i in range(3):
    print(list(itertools.islice(as_arrays[i], 0, 5)))

assert len(as_tups) == len(as_arrays[0])  # same len
assert [tup[0] for tup in as_tups] == list(as_arrays[0])  # same elements
assert [tup[0] for tup in as_tups] == [tup[0] for tup in n_as_tups]