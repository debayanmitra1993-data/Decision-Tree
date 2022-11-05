import numpy as np

def label_counter(x):
  count_labels = {}
  print('inside label counter x shape = ',x.shape)
  for pt in x:
    if pt in count_labels:
      count_labels[pt] += 1
    else:
      count_labels[pt] = 1
  return count_labels


def calculate_entropy(x):
  count_labels = label_counter(x)
  ent = 0
  for label in count_labels.keys():
    p = round(count_labels[label]/np.sum(list(count_labels.values())),4)
    ent += (p*np.log2(p))
  return -round(ent,4)


def calculate_information_gain(x,y):
  entropy_parent = calculate_entropy(y)
  count_labels = label_counter(x)

  weighted_entropy_child = 0
  for label in count_labels.keys():
    label_weight = round(count_labels[label]/np.sum(list(count_labels.values())),4)
    label_indices = np.where(x == label)
    label_entropy = calculate_entropy(y[label_indices])
    weighted_entropy_child += (label_weight*label_entropy)
  
  return round(entropy_parent - weighted_entropy_child,4)
