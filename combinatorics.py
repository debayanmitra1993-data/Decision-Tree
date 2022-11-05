def factorial(x):
  if x == 1 or x == 0:
    return 1
  else:
    return x*factorial(x-1)

def permutation(n,r):
  return factorial(n)/factorial(n - r)

def sum_permutations(n):
  sum = 0
  for i in range(n + 1):
    sum += permutation(n,i)
  return sum
