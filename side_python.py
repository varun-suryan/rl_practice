import numpy as np
class Person:
  def __init__(self, name = 'Varun'):
    self.name = name
    self.age = 26

  def test_func(self):
  	print('My name is ' + self.name + ' and I am ' + str(self.age) + ' years old.')   
  

action = 5 if 7>9 else 9
print(action)
a = np.array([3, 4, 5])
print(a.T.shape)