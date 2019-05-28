class Person:
  def __init__(self, name = 'Varun'):
    self.name = name
    self.age = 26

  def test_func(self):
  	print('My name is ' + self.name + ' and I am ' + str(self.age) + ' years old.')   
  

p1 = Person()

p1.test_func()
