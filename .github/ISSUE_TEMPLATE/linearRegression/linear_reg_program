import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data1.csv', header = None)

plt.scatter(df[0], df[1])
plt.xticks(np.arange(0, 30, 5))
plt.yticks(np.arange(0, 30, 5))
plt.title("Food truck and their profit")
plt.show()

def computeCost(X, y, theta):
    
    m, n = X.shape
    
    X = np.hstack((np.ones((m,1)), X))

    predictions = X@theta
    error = sum((predictions-y)**2)
    J = (1/(2*m))*error
    
    grad = (1/m) * (X.T @ (predictions-y))
    
    return J, grad

X = df.iloc[:,0:1].values
y = df.iloc[:,1:2].values
theta = np.zeros((X.shape[1]+1, 1))

J, grad = computeCost(X, y, theta);
print('Cost J: ',J)
print('grad value: ', grad)

def gradientDescent(X, y, alpha, number_iterations):
    
    m, n = X.shape
    j_history = []
    theta = np.zeros((n+1, 1))   
    for i in range(number_iterations):
    
        J, grad = computeCost(X, y, theta)
        
        theta = theta - (alpha * grad)
        j_history.append(J)
    
    return j_history, theta
j_history, theta_final = gradientDescent(X, y, alpha = 0.01, number_iterations=1500)
print('Cost function j_history size: ', np.size(j_history))
print('Final value of Theta size: ', theta_final)

plt.plot(j_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost values J')
plt.title('Cost Vs Iterations')
plt.show()

X_values = [X for X in range(25)]
y_values = [y*theta_final[1]+theta_final[0] for y in X_values]
plt.plot(X_values, y_values)
plt.xlabel('Goods vslues')
plt.ylabel('Profit')
plt.title('Fitting prediction line on Data')
plt.show()
