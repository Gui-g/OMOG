import numpy as np
from scipy.special import comb

K = 4

class Curve():
    def __init__(self):
        self.points = np.empty((0, 4))

class Nurb(Curve):

    degree = K-1

    def knot_vector(self):
        '''
        For an open nonuniform curve that interpolated the end points, the t_j (knot value) are calculated using K: (Mortensen, 2006)
        t_j = 0 if j < K
        t_j = j - K + 1 if K <= j <= n
        t_j = n - K + 2 if j > n
        '''

        n = len(self.points)
        knot_vector = np.zeros(self.degree + 1)
        middle = np.arange((self.degree + 1), n + 1) - (self.degree + 1) + 1
        knot_vector = np.concatenate((knot_vector, middle, np.full((self.degree + 1), n - (self.degree + 1) + 2)))
        
        return knot_vector

    def normalized_knot(self, knots):
        # knot_vector [0,1]
        min_val = np.min(knots)
        max_val = np.max(knots)
        
        normalized_knots = (knots - min_val) / (max_val - min_val)
        return normalized_knots

    def deboor(self, param_u, point_index, degree, knots):
        '''
        Cox-deBoor (Foley, 1990):
        Bk,0 (u) = 1 if u_k <= u <= u_k+1
                 = 0 otherwise
        Bk,d(u) = (((u - u_k) / (u_(k+d) - u_k)) * Bk,d-1(u)) + (((u_(k+d+1) - u) / (u_(k+d+1) - u_(k+1))) * Bk+1,d-1(u))
        '''
        
        if degree == 0:
            if param_u >= knots[point_index] and param_u < knots[point_index + 1]:
                return 1
            else:
                return 0
            
        # treating */0 = 0
        if knots[point_index + degree] != 0:
            first_deboor = self.deboor(param_u, point_index, degree - 1, knots)
            first = ((param_u - knots[point_index]) / (knots[point_index + degree])) * first_deboor
        else:
            first = 0
        if (knots[point_index + degree + 1] - knots[point_index + 1]) != 0:
            second_deboor = self.deboor(param_u, point_index + 1, degree - 1, knots)
            second = ((knots[point_index + degree + 1] - param_u) / (knots[point_index + degree + 1] - knots[point_index + 1])) * second_deboor
        else:
            second = 0
        
        return first + second
    
    def nurbs(self, param_u, knots):
        '''
        NURBS (Mortensen, 2006):
        p(u) = (sum i=0~n (h_i * p_i * Ni,k(u))) / (sum i=0~n (h_i * Ni,k(u)))

        h = weight
        p = control point
        n = num control points
        Ni,k = nonuniform B-spline basis function (cox deboor)
        '''
        first = 0
        second = 0

        for point_index in range(len(self.points)):
            deboor_1 = self.deboor(param_u, point_index, (self.degree + 1), knots)
            first = first + self.points[point_index,-1]*self.points[point_index,:-2]*deboor_1
        for point_index in range(len(self.points)):
            deboor_2 = self.deboor(param_u, point_index, (self.degree + 1), knots)
            second = second + self.points[point_index,-1]*deboor_2

        return first / second
    
    def create_curve(self, num_points):
        n = len(self.points)
        if n == 0:
            return np.empty((0,2))
        degree = self.degree
        knots = self.knot_vector()
        knots = self.normalized_knot(knots)
        
        curve_points = []
        for t in np.linspace(knots[degree], knots[-degree], num_points):
            x, y = self.nurbs(t, knots)
            curve_points.append((x, y))

        return curve_points
    
class Bezier(Curve):

    def bernstein(self, u, n, v):
        '''
        Bernstein basis polynomials (wiki):
        
        n+1 of degree n

        b_v,n(u) := comb(n v) u^v (1 - u)^(n - v), v = 0,...,n
        '''
        c = comb(n, v, exact=False)
        a = u ** v
        b = (1 - u) ** (n - v)
        return c * a * b
    
    def create_curve(self, num_points):
        '''
        De Casteljau's algorithm (wiki):

        Curve B of degree n, control poins B0,...,Bn
        B(t) = sum i=0...n Bi*b_i,n(t)
        '''
        n = len(self.points)
        curve_points = []
        for t in np.linspace(0, 1, num_points):
            points = self.points[:, :-1]
            point = np.array((0.0, 0.0))
            for i in range(len(points)):
                b = self.bernstein(t, n - 1, i)
                point += points[i, :-1] * b
            curve_points.append(point)
        
        return curve_points
    
