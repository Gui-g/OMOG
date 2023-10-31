import numpy as np
import pygame
from curve import *

SCREEN_SIZE = (1200, 800)
POINT_RADIUS = 5
GREY = pygame.color.Color(150, 150, 150, 1)
WHITE = pygame.color.Color(255, 255, 255, 255)
RED = pygame.color.Color(255, 0, 0, 255)
BLACK = pygame.color.Color(0, 0, 0, 0)

class Environment():
    def __init__(self):
        if not pygame.get_init():
            pygame.init()
        
        self.font = pygame.font.Font(None, 24)
        self.screen = pygame.display.set_mode(SCREEN_SIZE, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.is_running = True
        self.curves: list[Curve] = [Nurb()]
        self.drag_id = None
        self.active_curve_index = 0
        self.create = True
        self.weight_mode = False
        self.curve_mode = True #Nurbs
        self.show_points = True

        self.max_curves = 2
        self.num_curves = 1
        self.nurbs_max = 4
        self.bezier_max = 5

    def add_curve(self, curve):

        if self.num_curves >= self.max_curves:
            print(str(self.max_curves) + " curves already created")
            return
           
        self.curves.append(curve)
        self.num_curves = len(self.curves)

    def active_curve(self):
        if isinstance(self.curves[self.active_curve_index], Nurb):
            self.curve_mode = True
        elif isinstance(self.curves[self.active_curve_index], Bezier):
            self.curve_mode = False
        return self.curves[self.active_curve_index]
    
    def click_collision(self, click_position):
        active_curve = self.active_curve()
        if not active_curve.points.any():
            return None
        
        click_pos_np = np.array([click_position,])
        point_distances = np.sqrt(np.sum(np.square(active_curve.points[:, :-2] - click_pos_np), axis=1))
        close_point_indices = np.argwhere(point_distances < 2 * POINT_RADIUS)

        if close_point_indices.size > 0:
            closest_index = close_point_indices[0, 0]
            return closest_index + (0 if self.drag_id is None else (0 if self.drag_id > closest_index else 1))
        
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.is_running = False

        pos = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONUP:
            self.handle_event_mouse_up(pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.handle_event_mouse_down(pos)
        if event.type == pygame.KEYUP:
            self.handle_event_keyboard(event)
        if event.type == pygame.VIDEORESIZE:
            SCREEN_SIZE = event.size
            self.screen = pygame.display.set_mode(SCREEN_SIZE, pygame.RESIZABLE)
    
    def handle_event_mouse_up(self, pos):

        point = np.array((*pos, 1, 1))
        index = self.click_collision(pos)

        if not self.weight_mode:
            if self.create and index is None and not np.any(np.all(self.active_curve().points[:, :-2] == point[:-2], axis=1)):
                if isinstance(self.active_curve(), Nurb) and len(self.active_curve().points) >= self.nurbs_max:
                    print("NURBS can have at most " + str(len(self.active_curve().points)) + " control points.")
                elif isinstance(self.active_curve(), Bezier) and len(self.active_curve().points) >= self.bezier_max:
                    print("Bezier can have at most " + str(len(self.active_curve().points)) + " control points.")
                else:
                    self.active_curve().points = np.vstack((self.active_curve().points, point))
            elif not self.create and index is not None:
                self.active_curve().points = np.delete(self.active_curve().points, index - 1, axis=0)
        else:
            if index is not None:
                if self.create:
                    self.active_curve().points[index-1, -1] += 1
                else:
                    self.active_curve().points[index-1, -1] = max(1, self.active_curve().points[index-1, -1] -1)
                
        self.drag_id = None

    def handle_event_mouse_down(self, pos):
        if (index := self.click_collision(pos)) is not None:
            self.drag_id = index

    def handle_event_keyboard(self, event):
        match event.key:
            case pygame.K_d:
                self.create = not self.create
            case pygame.K_w:
                self.weight_mode = not self.weight_mode
            case pygame.K_r:
                self.curves = [Nurb()]
                self.active_curve_index = 0
                self.num_curves = len(self.curves)
            case pygame.K_t:
                self.show_points = not self.show_points
            case pygame.K_q:
                self.active_curve_index -= 1
                if self.active_curve_index < 0:
                    self.active_curve_index = 0
            case pygame.K_e:
                self.active_curve_index += 1
                if self.active_curve_index >= len(self.curves):
                    self.active_curve_index = len(self.curves) - 1
            case pygame.K_1:
                self.add_curve(Nurb())
                self.active_curve_index = len(self.curves) - 1
            case pygame.K_2:
                self.add_curve(Bezier())
                self.active_curve_index = len(self.curves) - 1
            case pygame.K_c:
                self.c0()
            case pygame.K_v:
                self.g1()
            case pygame.K_b:
                self.g2()
            

    def c0(self):
        for curva_1, curva_2 in zip(self.curves, self.curves[1:]):
            end_point = curva_1.points[-1]
            start_point = curva_2.points[0]

            translation = start_point[:-2] - end_point[:-2]
            translation = np.append(translation, 0)

            curva_2.points[:, :-1] -= translation

    def g1(self):
        self.c0()

        for curve_1, curve_2, curve_2_index in zip(self.curves[:-1], self.curves[1:], range(1, len(self.curves))):
            if len(curve_1.points) == 0 or len(curve_2.points) == 0:
                continue

            curve_1_last, curve_1_second_last = curve_1.points[-1, :-2], curve_1.points[-2, :-2]
            curve_2_first, curve_2_second = curve_2.points[0, :-2], curve_2.points[1, :-2]

            direction = curve_1_last - curve_1_second_last
            direction_mod = np.linalg.norm(direction)
            direction /= direction_mod

            position_dir = curve_2_second - curve_2_first
            position_mag = np.linalg.norm(position_dir)

            position = np.append(curve_2_first + (direction * position_mag), [1, 1])

            self.curves[curve_2_index].points[1] = position
    
    def g2(self):
        self.g1()

        for curve_1, curve_2, curve_2_index in zip(self.curves[:-1], self.curves[1:], range(1, len(self.curves))):
            if len(curve_1.points) == 0 or len(curve_2.points) == 0:
                continue

            curve_1_last, curve_1_second_last, curve_1_third_last = curve_1.points[-1, :-2], curve_1.points[-2, :-2], curve_1.points[-3, :-2]
            curve_2_first, curve_2_second, curve_2_third = curve_2.points[0, :-2], curve_2.points[1, :-2], curve_2.points[2, :-2]

            vector_curve_2_BA = curve_2_first - curve_2_second
            vector_curve_2_BA /= np.linalg.norm(vector_curve_2_BA)
            vector_curve_2_BC = curve_2_third - curve_2_second
            vector_curve_2_BC_magnitude = np.linalg.norm(vector_curve_2_BC)

            vector_curve_1_CD = curve_1_last - curve_1_second_last
            vector_curve_1_CD /= np.linalg.norm(vector_curve_1_CD)
            vector_curva_1_CB = curve_1_third_last - curve_1_second_last
            vector_curva_1_CB /= np.linalg.norm(vector_curva_1_CB)

            angle = np.arccos(np.dot(vector_curva_1_CB, vector_curve_1_CD))
            if all((vector_curve_1_CD[1] >= 0, vector_curva_1_CB[0] <= 0)) or all ((vector_curve_1_CD[1] <= 0, vector_curva_1_CB[0] >= 0)):
                angle += np.pi
            
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            new_vector_curve_2_BC = np.dot(rotation_matrix, vector_curve_2_BA)
            new_vector_curve_2_BC *= vector_curve_2_BC_magnitude
            new_point = curve_2_second + new_vector_curve_2_BC
            self.curves[curve_2_index].points[2] = np.append(new_point, [1, 1])
            

    def draw(self):
        self.screen.fill("white")

        if self.show_points:
            for curve in self.curves:
                x_coords, y_coords = curve.points[:, 0], curve.points[:, 1]
                for point in curve.points:
                    pygame.draw.circle(self.screen, BLACK, point[:-2], POINT_RADIUS)
                for i in range(len(curve.points) - 1):
                    pygame.draw.line(self.screen, GREY, (int(x_coords[i]), int(y_coords[i])), (int(x_coords[i + 1]), int(y_coords[i + 1])), 1)

        for curve in self.curves:
            n = len(curve.points)
            curve_points = curve.create_curve(100 * n)
            for point1, point2 in ((p1, curve_points[p1_index + 1])
                                for p1_index, p1 in enumerate(curve_points[:-2])):
                pygame.draw.line(self.screen, BLACK, point1, point2, 1)

        font = pygame.font.Font(None, 24)
        if self.show_points:
            for point in self.active_curve().points:
                if not self.weight_mode:
                    continue
                else:
                    w = point[-1]
                    text = font.render(f"P({w})", True, BLACK)
                self.screen.blit(text, (int(point[0]), int(point[1])))
        
        weight_mode_text = font.render(f"Weight Mode: {self.weight_mode}", True, "dark green" if self.weight_mode else "crimson")
        self.screen.blit(weight_mode_text, (20, 20))
        if self.weight_mode:
            weight_mode_control_text = font.render(f"Weight Up {self.create}", True, "dark green" if self.create else "crimson")
            self.screen.blit(weight_mode_control_text, (20, 40))
        else:
            create_text = font.render(f"Create Point: {self.create}", True, "dark green" if self.create else "crimson")
            self.screen.blit(create_text, (20, 40))
        nurbs_text = font.render(f"Curve: Nurbs", True, "dark green" if self.curve_mode else "crimson")
        self.screen.blit(nurbs_text, (20, 60))
        bezier_text = font.render(f"Curve: Bezier", True, "crimson" if self.curve_mode else "dark green")
        self.screen.blit(bezier_text, (20, 80))
        curve_text = font.render(f"Active Curve: {self.active_curve_index + 1}", True, "black")
        self.screen.blit(curve_text, (20, 100))
        point_text_on = font.render(f"Show Support Data: {self.show_points}",  True, "dark green" if self.show_points else "crimson")
        self.screen.blit(point_text_on, (20, 120))
        
        pygame.display.flip()

    def main_loop(self):
        while self.is_running:
            for event in pygame.event.get():
                self.handle_event(event)

            if self.drag_id is not None:
                pos = pygame.mouse.get_pos()
                self.active_curve().points[self.drag_id] = np.array((*pos, 1, self.active_curve().points[self.drag_id,-1]))
            
            self.draw()
            self.clock.tick(60)
    
    def quit(self):
        if pygame.get_init():
            pygame.quit()