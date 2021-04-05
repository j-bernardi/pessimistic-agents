import math
import numpy as np

def int_to_k_ints(i, k):
    sums = []
    i_prime = i
    for j in range(k, 0, -1):
        n = highest_n_choose_k_below_x(i_prime, j)
        sums.append(n - j + 1) # +/- 1
        i_prime -= n_choose_k(n, j)
    sums.append(0)
    return [sums[j] - sums[j+1] for j in range(k)]

def k_ints_to_int(ints):
    pass

def highest_n_choose_k_below_x(x, k):
    n = k
    while True:
        if n_choose_k(n, k) > x:
            break
        n += 1
    n -= 1
    return n

def n_choose_k(n, k):
    prod = 1
    for i in range(k):
        prod *= n - i
    return prod // math.factorial(k)

def two_ints_to_int_skew(int1, int2, skew):
    """
    maps a pair of ints to an int
    keeping int1 + int2 constant, output is ~minimized when int2 ~= skew * int1
    """
    int2_quo, int2_rem = divmod(int2, skew)
    return skew * two_ints_to_int(int1, int2_quo) + int2_rem

def int_to_two_ints_skew(i, skew):
    i_quo, int2_rem = divmod(i, skew)
    int1, int2_quo = int_to_two_ints(i_quo)
    return int1, int2_quo * skew + int2_rem

def two_ints_to_int(int1, int2):
    total = int1 + int2
    return ((total + 1) * total // 2) + int1

def int_to_two_ints(i):
    total = int((1 + math.sqrt(8*(i) + 1)) / 2) - 1 # thanks stackexchange
    int1 = i - ((total + 1) * total // 2)
    return int1, total-int1

def encode_nk_ints_1(ints, k):
    n_ints = len(ints) // k
    n_ints_unary = int_to_unary(n_ints)
    bitstrings = [int_to_bitstring(i) for i in ints]
    lengths = [len(bits) for bits in bitstrings]
    length_unaries = [int_to_unary(l) for l in lengths]
    return "".join([n_ints_unary] + \
                    length_unaries + \
                    bitstrings
                )

def safe_decode_nk_ints_1(string, k):
    try:
        return decode_nk_ints_1(string, k)
    except IndexError:
        return None

def decode_nk_ints_1(string, k):
    pointer = 0
    n_ints, pointer = read_unary(string, pointer)
    n_ints *= k
    lengths = []
    for _ in range(n_ints):
        length, pointer = read_unary(string, pointer)
        lengths.append(length)
    ints = []
    for length in lengths:
        i, pointer = read_bitstring(string, length, pointer)
        ints.append(i)
    return tuple(ints)

def encode_nk_ints(ints, k):
    n_ints = len(ints) // k
    n_ints_bitstring = int_to_bitstring(n_ints)
    n_ints_length = len(n_ints_bitstring)
    n_ints_length_bitstring = int_to_bitstring(n_ints_length)
    n_ints_length_length = len(n_ints_length_bitstring)
    n_ints_length_length_unary = int_to_unary(n_ints_length_length)
    bitstrings = [int_to_bitstring(i) for i in ints]
    lengths = [len(bits) for bits in bitstrings]
    length_bitstrings = [int_to_bitstring(l) for l in lengths]
    length_lengths = [len(lbits) for lbits in length_bitstrings]
    length_length_unaries = [int_to_unary(ll) for ll in length_lengths]
    return "".join([n_ints_length_length_unary,
                    n_ints_length_bitstring,
                    n_ints_bitstring] + \
                    length_length_unaries + \
                    length_bitstrings + \
                    bitstrings
                )

def safe_decode_nk_ints(string, k):
    try:
        return decode_nk_ints(string, k)
    except IndexError:
        return None

def decode_nk_ints(string, k):
    pointer = 0
    n_ints_length_length, pointer = read_unary(string, pointer)
    n_ints_length, pointer = read_bitstring(string, n_ints_length_length, pointer)
    n_ints, pointer = read_bitstring(string, n_ints_length, pointer)
    n_ints *= k
    length_lengths = []
    for _ in range(n_ints):
        length_length, pointer = read_unary(string, pointer)
        length_lengths.append(length_length)
    lengths = []
    for ll in length_lengths:
        length, pointer = read_bitstring(string, ll, pointer)
        lengths.append(length)
    ints = []
    for length in lengths:
        i, pointer = read_bitstring(string, length, pointer)
        ints.append(i)
    return tuple(ints)

def int_to_bitstring(n):
    length = int(math.log2(n+1))
    prev_cutoff = 2**length - 1
    diff = n - prev_cutoff
    no_lead_zeros = bin(diff)[2:]
    if no_lead_zeros == '0':
        no_lead_zeros = ''
    lead_zeros = "".join(['0' for _ in range(length - len(no_lead_zeros))])
    return lead_zeros + no_lead_zeros

def bitstring_to_int(string):
    length = len(string)
    if length == 0:
        return 0
    return 2**length - 1 + int(string, 2)

def int_to_unary(n):
    return "".join(['0' for _ in range(n)] + ['1'])

def read_unary(string, pointer):
    pointer_start = pointer
    while string[pointer] == '0':
        pointer += 1
    return pointer - pointer_start, pointer + 1

def read_bitstring(string, length, pointer):
    bitstring = string[pointer:pointer+length]
    if pointer+length > len(string):
        raise IndexError
    pointer = pointer + length
    return bitstring_to_int(bitstring), pointer

def integers_from(start):
    i = start
    while True:
        yield i
        i += 1

# generate coefficients for many layers of boundaries
def coefficients(num):
    n = 0
    for offset in [2]:# integers_from(1):
        num_edges = np.floor(np.pi*2*offset)
        angles = (np.pi * 2 * np.arange(num_edges)) / num_edges
        x = np.cos(angles)
        y = np.sin(angles)
        for coeffs in np.c_[x, y]:
            yield coeffs, offset
            n += 1
            if n >= num:
                return
