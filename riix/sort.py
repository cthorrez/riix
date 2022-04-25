# you can write to stdout for debugging purposes, e.g.
print("This is a debug message")

def mergesort_and_dedup(a):
    if len(a) == 1:
        return [(a[0],1)]
    if len(a) == 2:
        if a[0] < a[1]:
            return [(a[0],1), (a[1],1)]
        elif a[1] < a[0]:
            return [(a[1],1), (a[0],1)]
        else:
            return [(a[0],2)]

    midx = len(a) // 2
    left, right = a[:midx], a[midx:]

    left_sorted = mergesort_and_dedup(left)
    right_sorted = mergesort_and_dedup(right)

    out_list = []
    li, ri = 0, 0
    while li + ri < len(left_sorted) + len(right_sorted):
        if li == len(left_sorted):
            out_list.append(right_sorted[ri])
            ri += 1
        elif ri == len(right_sorted):
            out_list.append(left_sorted[li])
            li += 1
        elif left_sorted[li][0] < right_sorted[ri][0]:
            out_list.append(left_sorted[li])
            li += 1
        elif left_sorted[li][0] > right_sorted[ri][0]:
            out_list.append(right_sorted[ri])
            ri += 1
        else:
            out_list.append((left_sorted[li][0], left_sorted[li][1] + right_sorted[ri][1]))
            li += 1
            ri += 1

    return out_list
import random
a = [random.randint(0,10000) for x in range(10000000)]

print(mergesort_and_dedup(a))