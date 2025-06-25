
# PROGRAM 1 ─ NUMPY ARRAYS

import numpy as np

# 1-A. Creating and inspecting a 2-D array
arr = np.array([[1, 2, 3], [4, 2, 5]])
print("Array is of type :", type(arr))          # <class 'numpy.ndarray'>
print("No. of dimensions :", arr.ndim)          # 2
print("Shape of array    :", arr.shape)         # (2, 3)
print("Size of array     :", arr.size)          # 6
print("Element dtype     :", arr.dtype)         # int32 (or int64 depending on OS)

print()  # spacer line

# 1-B. Simple row slicing
a = np.array([[1, 2, 3],
              [3, 4, 5],
              [4, 5, 6]])
print("Full array:\n", a)
print("\nAfter slicing (rows 1 → end):\n", a[1:])

print()  # spacer line

# 1-C. Ellipsis (…) indexing tricks
print("Original array:\n", a)
print("\nItems in 2nd column:", a[..., 1])      # [2 4 5]
print("Items in 2nd row   :", a[1, ...])        # [3 4 5]
print("Column 2 repeated  :", a[..., 1])  

# PROGRAM 2 ─ PANDAS DATAFRAMES

import pandas as pd

print("\n" + "="*40)
print("Pandas: building DataFrames\n")

# 2-A. From a 2-D NumPy array with custom row/col labels
array_data = np.array([['Row1', 1, 2],
                       ['Row2', 3, 4]])
df1 = pd.DataFrame(data=array_data[:, 1:],      # numeric cols
                   index=array_data[:, 0],      # row labels
                   columns=['Col1', 'Col2'])    # col labels
print("DataFrame from 2-D array:")
print(df1, "\n")

# 2-B. From a plain 2-D array (default indices/cols)
my_2darray = np.array([[1, 2, 3],
                       [4, 5, 6]])
print("DataFrame from plain 2-D array:")
print(pd.DataFrame(my_2darray), "\n")

# 2-C. From a dictionary of lists (keys → column names)
my_dict = {1: ['A', 'C'],
           2: ['A', 'B'],
           3: ['B', 'D']}
print("DataFrame from dict of lists:")
print(pd.DataFrame(my_dict), "\n")

# 2-D. From an existing Series
my_series = pd.Series({
    "United Kingdom": "London",
    "India": "New Delhi",
    "United States": "Washington DC",
    "Belgium": "Brussels"})
print("DataFrame from Series:")
print(pd.DataFrame(my_series, columns=['Capital']), "\n")

# 2-E. Shape vs. length demo
df_tmp = pd.DataFrame(np.array([[1, 2, 3],
                                [4, 5, 6]]))
print("Shape  :", df_tmp.shape)   # (2, 3)
print("Len(idx):", len(df_tmp))   # 2 rows

# PROGRAM 3 ─ BASIC PLOTS WITH MATPLOTLIB

import matplotlib.pyplot as plt

# 3-A. A single simple line plot
x = [1, 2, 3]
y = [2, 4, 1]

plt.figure(figsize=(5, 3))
plt.plot(x, y, marker='o')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('My first graph')
plt.tight_layout()
plt.show()

# 3-B. Over-plotting several series on the same axes
days   = list(range(1, 9))          # 1 … 8
temp_A = [ 0,  6,  2, 15, 10,  8, 16, 21]
temp_B = [ 4,  2,  6,  8,  3, 20, 13, 15]

plt.figure(figsize=(6, 3))
plt.plot(days, temp_A, "or", label="Sensor A")  # red circles
plt.plot(days, temp_B, "sb", label="Sensor B")  # blue squares
plt.xlabel("Day →")
plt.ylabel("Temp (°C) →")
plt.title("Two sensors over one week")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# 3-C. 2×2 grid of independent sub-plots
fig = plt.figure(figsize=(10, 8))

sub1 = plt.subplot(2, 2, 1)
sub1.plot(days, temp_A, 'sb')
sub1.set_title('1st Rep')
sub1.set_xticks(range(1, 9))

sub2 = plt.subplot(2, 2, 2)
sub2.plot(days, temp_B, 'or')
sub2.set_title('2nd Rep')
sub2.set_xticks(range(1, 9, 2))

sub3 = plt.subplot(2, 2, 3)
sub3.plot(range(0, 22, 3), 'vg')
sub3.set_title('3rd Rep')

sub4 = plt.subplot(2, 2, 4)
sub4.plot(days, temp_B[::-1], 'Dm')   # just for variety
sub4.set_title('4th Rep')
sub4.set_yticks(range(0, 25, 2))

fig.suptitle("Matplotlib multi-panel example", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for main title
plt.show()