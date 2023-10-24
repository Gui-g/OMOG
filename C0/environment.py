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
        self.del_curve = False
        self.create = True
        self.weight_mode = False
        self.curve_mode = True #Nurbs

        self.max_curves = 2
        self.num_curves = 1

    def add_curve(self, curve):

        if self.num_curves >= self.max_curves:
            print("2 curves already created")
            return

        if isinstance(curve, Nurb):
            if len(curve.points) > 6:
                print("NURBS can have at most 6 control points.")
                return
        elif isinstance(curve, Bezier):
            if len(curve.points) > 5:
                print("Bezier can have at most 5 control points.")
                return
        self.curves.append(curve)
        self.num_curves += 1

    def delete_curve(self, index):
        if 0 <= index < len(self.curves):
            del self.curves[index]
            self.num_curves -= 1
            if self.active_curve_index >= self.num_curves:
                self.active_curve_index = self.num_curves - 1

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
                if isinstance(self.active_curve(), Nurb) and len(self.active_curve().points) >= 6:
                    print("NURBS can have at most 6 control points.")
                elif isinstance(self.active_curve(), Bezier) and len(self.active_curve().points) >= 5:
                    print("Bezier can have at most 5 control points.")
                else:
                    self.active_curve().points = np.vstack((self.active_curve().points, point))
            elif not self.create and index is not None:
                self.active_curve().points = np.delete(self.active_curve().points, index - 1, axis=0)
            elif index is not None and self.del_curve:
                self.delete_curve(self.active_curve_index)
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
                self.del_curve = not self.del_curve
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
                self.curve_mode = False
                self.add_curve(Bezier())
                self.active_curve_index = len(self.curves) - 1
            case pygame.K_c:
                self.c0()

    def c0(self):
        for curva_1_index, curva_1 in enumerate(self.curves[:-1]):
            curva_2_index = curva_1_index + 1
            curva_2 = self.curves[curva_2_index]

            end_point = curva_1.points[-1]
            start_point = curva_2.points[0]
            translation = np.append(start_point[:-2] - end_point[:-2], 0)

            for i in range(curva_2.points.shape[0]):
                curva_2.points[i, :-1] -= translation

    def draw(self):
        self.screen.fill("white")

        x_coords, y_coords = self.active_curve().points[:, 0], self.active_curve().points[:, 1]

        for point in self.active_curve().points:
            pygame.draw.circle(self.screen, BLACK, point[:-2], POINT_RADIUS)
        for i in range(len(self.active_curve().points) - 1):
            pygame.draw.line(self.screen, GREY, (int(x_coords[i]), int(y_coords[i])), (int(x_coords[i + 1]), int(y_coords[i + 1])), 1)

        for curve in self.curves:
            n = len(curve.points)
            if n >= K:
                curve_points = curve.create_curve(50 * n)
                for point1, point2 in ((p1, curve_points[p1_index + 1])
                                    for p1_index, p1 in enumerate(curve_points[:-2])):
                    pygame.draw.line(self.screen, BLACK, point1, point2, 1)

        font = pygame.font.Font(None, 24)
        for point in self.active_curve().points:
            if not self.weight_mode:
                x, y, z, w = point
                text = font.render(f"P({x}, {y}, {z})", True, BLACK)
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
        delet_curve = font.render(f"Delete Curve: {self.del_curve}", True, "dark green" if self.del_curve else "crimson")
        self.screen.blit(delet_curve, (20, 100))
        curve_text = font.render(f"Active Curve: {self.active_curve_index + 1}", True, "black")
        self.screen.blit(curve_text, (20, 120))
        
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