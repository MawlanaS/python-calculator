import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# this is created by GamerYousha and i did for just fun purpose and 
#want to share this with others so that people can implement their projects in easy way
#surely you can't claim this code as your this is made by me and its opensource you can use this anywhere for Commercial or personal purpose i dont care xD
# just please dont claim to your friends that this is your code thanks 🥱
#Instagram: its_mawlanas
#YouTube: GamerYousha
#Facebook: ArafinHoqueYousha
class MatrixSolver:
    
    def __init__(self, matrix):
        self.matrix = matrix
    
    def row_reduce(self):
        for i in range(len(self.matrix)):
            # find the row with the largest first element
            max_row = max(range(i, len(self.matrix)), key=lambda x: abs(self.matrix[x][i]))
            # swap the current row with the max row
            self.matrix[i], self.matrix[max_row] = self.matrix[max_row], self.matrix[i]
            # make the diagonal element equal to 1
            diagonal = self.matrix[i][i]
            self.matrix[i] = [elem / diagonal for elem in self.matrix[i]]
            # zero out all elements below the diagonal element
            for j in range(i + 1, len(self.matrix)):
                multiple = self.matrix[j][i] / self.matrix[i][i]
                self.matrix[j] = [elem_j - multiple * elem_i for elem_i, elem_j in zip(self.matrix[i], self.matrix[j])]
            # zero out all elements above the diagonal element
            for j in range(i):
                multiple = self.matrix[j][i] / self.matrix[i][i]
                self.matrix[j] = [elem_j - multiple * elem_i for elem_i, elem_j in zip(self.matrix[i], self.matrix[j])]
                
    def solve(self):
        self.row_reduce()
        return [row[-1] for row in self.matrix]
        
class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def __str__(self):
        return '\n'.join([' '.join([str(num) for num in row]) for row in self.matrix])

    def __add__(self, other):
        if len(self.matrix) != len(other.matrix) or len(self.matrix[0]) != len(other.matrix[0]):
            raise ValueError('Matrices must be the same size to add them together.')
        result = []
        for i in range(len(self.matrix)):
            row = []
            for j in range(len(self.matrix[0])):
                row.append(self.matrix[i][j] + other.matrix[i][j])
            result.append(row)
        return Matrix(result)

    def __sub__(self, other):
        if len(self.matrix) != len(other.matrix) or len(self.matrix[0]) != len(other.matrix[0]):
            raise ValueError('Matrices must be the same size to subtract them.')
        result = []
        for i in range(len(self.matrix)):
            row = []
            for j in range(len(self.matrix[0])):
                row.append(self.matrix[i][j] - other.matrix[i][j])
            result.append(row)
        return Matrix(result)

    def transpose(self):
        result = []
        for j in range(len(self.matrix[0])):
            row = []
            for i in range(len(self.matrix)):
                row.append(self.matrix[i][j])
            result.append(row)
        return Matrix(result)
    def __mul__(self, other):
        if len(self.matrix[0]) != len(other.matrix):
            raise ValueError('Number of columns in first matrix must match number of rows in second matrix for multiplication.')
        result = []
        other_transpose = other.transpose().matrix
        for i in range(len(self.matrix)):
            row = []
            for j in range(len(other_transpose)):
                row.append(sum([a * b for a, b in zip(self.matrix[i], other_transpose[j])]))
            result.append(row)
        return Matrix(result)

class Polynomial: #quadritic equation can be solved
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __repr__(self):
        return "Polynomial({})".format(self.coeffs)

    def __add__(self, other):
        # Make sure the two polynomials have the same length
        n = max(len(self.coeffs), len(other.coeffs))
        coeffs1 = self.coeffs + [0] * (n - len(self.coeffs))
        coeffs2 = other.coeffs + [0] * (n - len(other.coeffs))

        # Add the coefficients of the two polynomials
        coeffs = [a + b for a, b in zip(coeffs1, coeffs2)]

        # Remove any trailing zeros
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()

        return Polynomial(coeffs)

    def __sub__(self, other):
        # Make sure the two polynomials have the same length
        n = max(len(self.coeffs), len(other.coeffs))
        coeffs1 = self.coeffs + [0] * (n - len(self.coeffs))
        coeffs2 = other.coeffs + [0] * (n - len(other.coeffs))

        # Subtract the coefficients of the two polynomials
        coeffs = [a - b for a, b in zip(coeffs1, coeffs2)]

        # Remove any trailing zeros
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()

        return Polynomial(coeffs)

    def __mul__(self, other):
        # Multiply the coefficients of the two polynomials
        coeffs = [0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                coeffs[i + j] += a * b

        # Remove any trailing zeros
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()

        return Polynomial(coeffs)

    def __call__(self, x):
        # Evaluate the polynomial at x
        result = 0
        for i, a in enumerate(self.coeffs):
            result += a * (x ** i)
        return result


    def sin(x):
        return math.sin(x)

    def cos(x):
        return math.cos(x)

    def tan(x):
        return math.tan(x)

    def asin(x):
        return math.asin(x)

    def acos(x):
        return math.acos(x)

    def atan(x):
        return math.atan(x)

    def radians(x):
        return math.radians(x)

    def degrees(x):
        return math.degrees(x)




class ParabolaSolver:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        
    def solve(self):
        x = np.linspace(-100, 100, 1000)
        y = self.a * x**2 + self.b * x + self.c
        return (x, y)
        
    def plot(self):
        x, y = self.solve()
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Parabola Graph')
        plt.show()


class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"
    
    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3D(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vector3D):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            raise TypeError("Multiplication not supported between vector and other type.")
        
    def cross_product(self, other):
        return Vector3D(self.y * other.z - self.z * other.y,
                        self.z * other.x - self.x * other.z,
                        self.x * other.y - self.y * other.x)
    
    def magnitude(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
    
    def normalize(self):
        mag = self.magnitude()
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)


class Mathl:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def div(a, b):
        if b == 0:
            raise ValueError('Cannot divide by zero')
        return a / b




class ComplexNumber:
    def __init__(self, real_part, imaginary_part):
        self.real_part = real_part
        self.imaginary_part = imaginary_part

    def __str__(self):
        if self.imaginary_part >= 0:
            return str(self.real_part) + "+" + str(self.imaginary_part) + "i"
        else:
            return str(self.real_part) + str(self.imaginary_part) + "i"

    def __add__(self, other):
        return ComplexNumber(self.real_part + other.real_part, self.imaginary_part + other.imaginary_part)

    def __sub__(self, other):
        return ComplexNumber(self.real_part - other.real_part, self.imaginary_part - other.imaginary_part)

    def __mul__(self, other):
        real = self.real_part * other.real_part - self.imaginary_part * other.imaginary_part
        imaginary = self.real_part * other.imaginary_part + self.imaginary_part * other.real_part
        return ComplexNumber(real, imaginary)

    def __truediv__(self, other):
        denominator = other.real_part**2 + other.imaginary_part**2
        real = (self.real_part * other.real_part + self.imaginary_part * other.imaginary_part) / denominator
        imaginary = (self.imaginary_part * other.real_part - self.real_part * other.imaginary_part) / denominator
        return ComplexNumber(real, imaginary)

    def conjugate(self):
        return ComplexNumber(self.real_part, -self.imaginary_part)

    def modulus(self):
        return math.sqrt(self.real_part**2 + self.imaginary_part**2)

    def polar_coordinates(self):
        modulus = self.modulus()
        if modulus == 0:
            return (0, 0)
        else:
            theta = math.atan2(self.imaginary_part, self.real_part)
            return (modulus, theta)

    def from_polar_coordinates(modulus, theta):
        real = modulus * math.cos(theta)
        imaginary = modulus * math.sin(theta)
        return ComplexNumber(real, imaginary)
        

def differentiate(expression, variable):
    """
    Takes a symbolic expression and a variable with respect to which
    differentiation is to be performed, and returns the differentiated expression.
    """
    return sp.diff(expression, variable)

def integrate(expression, variable):
    """
    Takes a symbolic expression and a variable with respect to which
    integration is to be performed, and returns the integrated expression.
    """
    return sp.integrate(expression, variable)

def partial_differentiate(expression, variables):
    """
    Takes a symbolic expression and a list of variables with respect to which
    partial differentiation is to be performed, and returns the partially differentiated expression.
    """
    for var in variables:
        expression = sp.diff(expression, var)
    return expression

def double_integral(expression, variable1, variable2, limits1, limits2):
    """
    Takes a symbolic expression, two variables with respect to which integration
    is to be performed, and their limits of integration, and returns the double integrated expression.
    """
    return sp.integrate(expression, (variable1, limits1[0], limits1[1]), (variable2, limits2[0], limits2[1]))

def triple_integral(expression, variable1, variable2, variable3, limits1, limits2, limits3):
    """
    Takes a symbolic expression, three variables with respect to which integration
    is to be performed, and their limits of integration, and returns the triple integrated expression.
    """
    return sp.integrate(expression, (variable1, limits1[0], limits1[1]), (variable2, limits2[0], limits2[1]), (variable3, limits3[0], limits3[1]))


