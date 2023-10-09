import pygame
import numpy as np

SCREEN_SIZE = (1000, 600)
POINT_RADIUS = 5
GREY = pygame.color.Color(150, 150, 150, 1)
WHITE = pygame.color.Color(255, 255, 255, 255)
RED = pygame.color.Color(255, 0, 0, 255)
BLACK = pygame.color.Color(0, 0, 0, 0)
K = 4 

class Enviroment:
    def __init__(self):
        if not pygame.get_init():
            pygame.init()

        self.screen = pygame.display.set_mode(SCREEN_SIZE, pygame.SCALED)
        self.clock = pygame.time.Clock()
        self.running = True
        self.drag_id = None
        self.points = np.empty((0,4))
        self.knots = np.empty((0,1))
        self.create = True
        self.weight_mode = False
    
    def knot_vector(self):
        '''
        For an open nonuniform curve that interpolated the end points, the t_j (knot value) are calculated using K: (Mortensen, 2006)
        t_j = 0 if j < K
        t_j = i - K + 1 if K <= j <= n -> i = j?
        t_j = n - K + 2 if j > n
        '''
        knot_vector = np.empty((0,1))

        while len(knot_vector) < K:
            knot_vector = np.insert(knot_vector, len(knot_vector), 0)       
        while len(knot_vector) <= len(self.points):
            knot_vector = np.insert(knot_vector, len(knot_vector), len(knot_vector) - K + 1)        
        while len(knot_vector) <= len(self.points) + K:
            knot_vector = np.insert(knot_vector, len(knot_vector), len(self.points) - K + 2)
        return knot_vector

    def normalized_knot(self):
        # knot_vector [0,1]
        min_val = np.min(self.knots)
        max_val = np.max(self.knots)
        
        normalized_knots = (self.knots - min_val) / (max_val - min_val)
        return normalized_knots

    def deboor(self, param_u, point_index, degree):
        '''
        Cox-deBoor (Foley, 1990):
        Bk,0 (u) = 1 if u_k <= u <= u_k+1
                 = 0 otherwise
        Bk,d(u) = (((u - u_k) / (u_(k+d) - u_k)) * Bk,d-1(u)) + (((u_(k+d+1) - u) / (u_(k+d+1) - u_(k+1))) * Bk+1,d-1(u))
        '''
        
        if degree == 0:
            if param_u >= self.knots[point_index] and param_u < self.knots[point_index + 1]:
                return 1
            else:
                return 0
            
        # treating */0 division = 0
        if self.knots[point_index + degree] != 0:
            first_deboor = self.deboor(param_u, point_index, degree - 1)
            first = ((param_u - self.knots[point_index]) / (self.knots[point_index + degree])) * first_deboor
        else:
            first = 0
        if (self.knots[point_index + degree + 1] - self.knots[point_index + 1]) != 0:
            second_deboor = self.deboor(param_u, point_index + 1, degree - 1)
            second = ((self.knots[point_index + degree + 1] - param_u) / (self.knots[point_index + degree + 1] - self.knots[point_index + 1])) * second_deboor
        else:
            second = 0
        
        return first + second
    
    def nurbs(self, param_u):
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
            deboor_1 = self.deboor(param_u, point_index, K)
            first = first + self.points[point_index,-1]*self.points[point_index,:-2]*deboor_1
        for point_index in range(len(self.points)):
            deboor_2 = self.deboor(param_u, point_index, K)
            second = second + self.points[point_index,-1]*deboor_2

        return first / second

    def draw(self):
        self.screen.fill(WHITE)

        for point in self.points:
            pygame.draw.circle(self.screen, BLACK, point[:-2], POINT_RADIUS)
        
        for point1, point2 in ((p1, self.points[p1_index + 1]) for p1_index, p1 in enumerate(self.points[:-1])):
            pygame.draw.line(self.screen, GREY, point1[:-2], point2[:-2], 1)
        
        n = len(self.points)
        if n >= K:
            self.knots = self.knot_vector()
            self.knots = self.normalized_knot()
            curve_points = []
            for t in np.arange(0, 1, 0.005):  #step size = 0.01
                x, y = self.nurbs(t)
                curve_points.append((x, y))
            pygame.draw.lines(self.screen, RED, False, curve_points, 2)

        font = pygame.font.Font(None, 24)
        if not self.weight_mode:
            for (x, y, z, w) in self.points:
                text = font.render(f"P({x}, {y}, {z})", True, BLACK)
                self.screen.blit(text, (x, y))
        else:
            for (x, y, z, w) in self.points:
                text = font.render(f"P({w})", True, BLACK)
                self.screen.blit(text, (x, y))
            
        pygame.display.flip()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        pos = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONUP:
            self.handle_event_mouse_up(pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.handle_event_mouse_down(pos)
        if event.type == pygame.KEYUP:
            self.handle_event_keyboard(event)

    def click_on_point(self, pos):
        if not self.points.any():
            return None
        
        np_pos = np.array([pos,])
        distances = np.array(np.sqrt(np.sum(np.square(np.subtract(np_pos, self.points[:, :-2])), axis=1)))

        if len(indexes := np.argwhere(distances < 2 * POINT_RADIUS)) > 0:
            rindex = indexes[0,0]
            return rindex + (0 if self.drag_id is None else 0 if self.drag_id > rindex else 1)
        
        return None

    def handle_event_mouse_up(self, pos):
        point = np.array((*pos, 1, 1))
        if not self.weight_mode:
            if self.create:
                if (index := self.click_on_point(pos)) is None:
                    if not np.isin(self.points[:, :-2], point).any():
                        self.points = np.concatenate((self.points, [point]), axis=0)
            else:
                if (index := self.click_on_point(pos)) is not None:
                    self.points = np.delete(self.points, (index-1), axis = 0)
        elif self.weight_mode:
            if (index := self.click_on_point(pos)) is not None:
                if self.create:
                    self.points[index-1,-1] = self.points[index-1,-1] + 1
                else:
                    self.points[index-1, -1] = self.points[index-1, -1] - 1 if self.points[index-1, -1] > 1 else 1

        self.drag_id = None

    def handle_event_mouse_down(self, pos):
        if (index := self.click_on_point(pos)) is not None:
            self.drag_id = index

    def handle_event_keyboard(self, event):
        match event.key:
            case pygame.K_d:
                self.create = not self.create
            case pygame.K_w:
                self.weight_mode = not self.weight_mode
                

    def main_loop(self):
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)

            if self.drag_id is not None:
                pos = pygame.mouse.get_pos()
                self.points[self.drag_id] = np.array((*pos, 1, self.points[self.drag_id,-1]))
            
            self.draw()
            self.clock.tick(60)

    def quit(self):
        if pygame.get_init():
            pygame.quit()