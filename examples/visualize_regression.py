from mola import Matrix, LabeledMatrix
from mola import regression
from mola import utils
import matplotlib.pyplot as plt
import random
import math


# Let's imagine we have a data from grocery store shopping carts, where each row is a shopping cart.
# The first column is the number of items in the shopping cart and the second column is the total price of the shopping cart.

# Let's generate this data first, assuming that each item has a price between 0.10 and 20.

# First, let's generate 100 shopping carts with 1 to 30 items in each cart.
random.seed(2)
n_items = [random.randrange(1,30,1) for x in range(100)]

# Second, let's generate a price for each item in each cart and calculate the total price.
prices_total = []
for cart in n_items:
    price = 0
    for item in range(cart):
        price += random.randrange(10,200,1)/10
    prices_total.append(price)

# Now let's build a labeled matrix out of the data to serve as our dataset, and print it.
data = LabeledMatrix({'num_items': n_items, 'cost': prices_total})
print(data)

# Next, let's fit a first-order polynomial with an intercept to the data. We treat the number of items as the independent variable and the final cost as the dependent variable.
# A first order-polynomial with an intercept is just a function of the form y = a*x + b, where y is the dependent variable, x is the independent variable, and a and b are parameters that we are estimating by fitting the function.
params = regression.fit_univariate_polynomial(independent_values=data.get_column('num_items',as_list=False), dependent_values=data.get_column('cost',as_list=False), degrees=[1], intercept=True)

# To demonstrate nonlinear regression, let's fit a nonlinear function to the data. This function is of the form a*x + c*sin(b*x) + d, where a, b, c, and d are parameters that we are estimating by fitting the function. We can see that a is the slope of a linear term, b is the period of a sine term, c is the amplitude of the sine term, and d is the intercept.
# First, we define the function in a form that can be used by the regression algorithm. We use a lambda function, where theta constains the parameters and x contains the independent variables. The function takes as input the list of parameters and the list of independent variables, and returns the value of the function for the given parameters and independent variables.
h = lambda theta,x: theta[0]*x[0] + theta[2]*math.sin(theta[1]*x[0]) + theta[3]
h_mat = Matrix([h])
# Second, we define the Jacobian matrix of the function. The Jacobian matrix contains the partial derivatives of the function with respect to each parameter (a, b, c, and d in vector theta). In effect, we use a row vector as the Jacobian matrix, where each column corresponds to the partial derivative of one unknown parameter.
J = [lambda theta,x: x[0], lambda theta,x: x[0]*theta[2]*math.cos(theta[1]*x[0]), lambda theta,x : math.sin(theta[1]*x[0]), lambda theta,x: 1]
J_mat = Matrix(J)

# Optionally, we can define an initial guess for the unknown parameters. Here, our initial guess is for the function is a linear slope of 1, sine period of 1, sine amplitude of 0.25, and no intercept (intercept is zero).
initial_guess = utils.column([1, 1, 0.25, 0])

# Finally, let's call the actual function to fit the nonlinear function to the data. The function returns the estimated parameters (a, b, c, and d in vector theta).
params_nl = regression.fit_nonlinear(independent_values=data.get_column('num_items',as_list=False), dependent_values=data.get_column('cost',as_list=False), h = h_mat, J = J_mat, initial=initial_guess)

# Also, print the parameters.
print("Parameters of nonlinear fit: " + str(params))
print("Parameters of nonlinear fit: " + str(params_nl))

# Let's plot the data and the fitted function (first-order polynomial with an intercept).
plt.plot(n_items, prices_total, 'b.', label='data points')
plt.plot([0,30],[params[1],params[1]+30*params[0]],'r-', label=f'linear fit y={round(params[0],2)}x+{round(params[1],2)}')

# Let's also plot the fitted nonlinear function.
x = [i/100 for i in range(3000)]
y = [h(params_nl,[i]) for i in x]
plt.plot(x,y,'g--', label=f'nonlinear fit y={round(params_nl[0],2)}x+{round(params_nl[2],2)}*sin({round(params[1],2)}x)+{round(params_nl[3],2)}')

plt.legend()
plt.xlabel('number of items in shopping cart')
plt.ylabel('total cost of shopping cart')
plt.show()

