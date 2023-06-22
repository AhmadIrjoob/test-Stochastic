# Imports:
from PIL import Image, ImageTk   # For adding images into the canvas widget
import tkinter as tk   # To build GUI
import numpy as np   # To deal with data in form of matrices
import random
import time


def fix_ord(spaces):
    for space in range(len(spaces)):
        spaces[space] = (spaces[space][0] * pixels, spaces[space][1] * pixels)
    return spaces


# Setting the sizes for the environment:
DIM = 10
pixels = 16   # pixels
shortest_path_length = 21   # only put value if under compare mode or else put 0 as value
env_height, env_width = DIM, DIM   # grid dimentions

# Global variable for dictionary with coordinates for the final route
a = {}
flag = 1

# Making static obstacles:
spaces = []
for i in range(2, 8, 4):
    for j in range(2, 8, 4):
        spaces += [(i, j), (i, j+1),
                    (i+1, j), (i+1, j+1)]
# print(spaces)

# Goal 1:
goal_1 = (4*pixels, 2*pixels)
goal1_visit = False
# Goal 2 (Final one at 9,9)

# Setting up dynamic obstacles:
grid1 = []
grid2 = []
cat_collision = {}
for i in range(4, 6):
    for j in range(3, 6):
        grid1.append((i, j))
        cat_collision[(i, j)] = 0
for i in range(14, 20):
    for j in range(4, 16):
        grid2.append((i, j))
        cat_collision[(i, j)] = 0

cat_avail_space1 = [ item for item in grid1 if item not in spaces ]
cat_avail_space2 = [ item for item in grid2 if item not in spaces ]
# print(cat_avail_space1)

obj_spaces = fix_ord(spaces)

dy1, dx1 = 0, 0
dy2, dx2 = 0, 0

cat1_pos = {}
cat2_pos = {}
cat_pos = {}
for i in cat_avail_space1:
    cat1_pos[tuple(i)] = 0
    cat_pos[tuple(i)] = 0
for i in cat_avail_space2:
    cat2_pos[tuple(i)] = 0
    cat_pos[tuple(i)] = 0


# Creating class for the environment
class Environment(tk.Tk, object):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.title('RL Q-learning')
        self.geometry('{0}x{1}'.format(env_height * pixels, env_height * pixels))
        self.build_environment()

        # Dictionaries to draw the final route
        self.d = {}
        self.f = {}

        # Key for the dictionaries
        self.i = 0

        # Writing the final dictionary first time
        self.c = True

        # Showing the steps for longest found route
        self.longest = 0

        # Showing the steps for the shortest route
        self.shortest = 0

        self.images = []


    # Function to build the environment
    def build_environment(self):
        global dy1, dx1, dy2, dx2, cat1_pos, cat2_pos

        self.canvas_widget = tk.Canvas(self,  bg='light grey', height=env_height * pixels, width=env_width * pixels)

        # Creating grid lines
        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='black')
        for row in range(0, env_height * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_height * pixels, row
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='black')

        # Creating objects of Obstacles:
        img_obstacle00 = Image.open("images/flag.png")
        self.obstacle00_object = ImageTk.PhotoImage(img_obstacle00)

        img_obstacle0 = Image.open("images/start.png")
        self.obstacle0_object = ImageTk.PhotoImage(img_obstacle0)


        img_obstacle6 = Image.open("images/bot.png")
        self.obstacle6_object = ImageTk.PhotoImage(img_obstacle6)
        
        img_obstacle7 = Image.open("images/bot.png")
        self.obstacle7_object = ImageTk.PhotoImage(img_obstacle7)


        # Creating obstacles themselves
        self.obstacle0 = self.canvas_widget.create_image(pixels * 0, pixels * 0, anchor='nw', image=self.obstacle0_object)
        
        # Creating static obstacles:
        for space in obj_spaces:
            self.canvas_widget.create_rectangle(space[0], space[1], space[0]+pixels, space[1]+pixels, fill='black')

        # Creating goal 1:
        self.canvas_widget.create_rectangle(goal_1[0], goal_1[1], goal_1[0]+pixels, goal_1[1]+pixels, fill='yellow')
        # self.canvas_widget.create_rectangle(goal_2[0], goal_2[1], goal_2[0]+pixels, goal_2[1]+pixels, fill='yellow')


        # Obstacle 6 (dyn_obs 1)
        dyn_obj_loc = random.choice(cat_avail_space1)
        cat1_pos[dyn_obj_loc] += 1
        dy1 = dyn_obj_loc[0]
        dx1 = dyn_obj_loc[1]
        self.obstacle6 = self.canvas_widget.create_image(dy1*pixels, dx1*pixels, anchor='nw', image=self.obstacle6_object)

       
        # Final Point (goal 2):
        self.flag = self.canvas_widget.create_image(pixels * (DIM-1), pixels * (DIM-1), anchor='nw', image=self.obstacle00_object)

        # Uploading the image of Mobile Robot
        img_robot = Image.open("images/agent.png")
        self.robot = ImageTk.PhotoImage(img_robot)

        # Creating an agent with photo of Mobile Robot
        self.agent = self.canvas_widget.create_image(0, 0*pixels, anchor='nw', image=self.robot)

        # Packing everything
        self.canvas_widget.pack()


    def createRectangle(self, x1, y1, x2, y2, **kwargs):
        if 'alpha' in kwargs:
            alpha = int(kwargs.pop('alpha') * 255)
            fill = kwargs.pop('fill')
            fill = self.winfo_rgb(fill) + (alpha,)
            image = Image.new('RGBA', (x2-x1, y2-y1), fill)
            self.images.append(ImageTk.PhotoImage(image))
            self.canvas_widget.create_image(x1, y1, image=self.images[-1], anchor='nw')
        self.canvas_widget.create_rectangle(x1, y1, x2, y2, **kwargs)


    # Function for cat movements 1:
    def cat1_move(self):
        global dy1, dx1, cat1_pos, cat_pos

        while flag:
            check1 = []
            time.sleep(1.5)
            if dy1+1 <= env_width-1 and (dy1+1, dx1) in cat_avail_space1:
                check1.append((dy1+1, dx1))

            if (dy1-1, dx1) in cat_avail_space1:
                check1.append((dy1-1, dx1))

            if (dy1, dx1+1) in cat_avail_space1:
                check1.append((dy1, dx1+1))

            if (dy1, dx1-1) in cat_avail_space1:
                check1.append((dy1, dx1-1))

            if (dy1-1, dx1-1) in cat_avail_space1:
                check1.append((dy1-1, dx1-1))

            if (dy1+1, dx1+1) in cat_avail_space1:
                check1.append((dy1+1, dx1+1))

            if (dy1-1, dx1+1) in cat_avail_space1:
                check1.append((dy1-1, dx1+1))

            if (dy1+1, dx1-1) in cat_avail_space1:
                check1.append((dy1+1, dx1-1))

            if len(check1) == 0:
                dyn_obj_loc = random.choice(cat_avail_space1)
                cat1_pos[dyn_obj_loc] += 1
                cat_pos[dyn_obj_loc] += 1
                dy1 = dyn_obj_loc[0]
                dx1 = dyn_obj_loc[1]
            else:
                random.shuffle(check1)
                dyn_obj_mov = random.choice(check1)
                cat1_pos[dyn_obj_mov] += 1
                cat_pos[dyn_obj_mov] += 1
                dy1 = dyn_obj_mov[0]
                dx1 = dyn_obj_mov[1]

                if cat_pos[dyn_obj_mov] == 1:
                    self.createRectangle((pixels * dy1), (pixels * dx1), (pixels * dy1) + pixels, (pixels * dx1) + pixels, fill='blue', alpha=0.1)
                if cat_pos[dyn_obj_mov] == 9:
                    self.createRectangle((pixels * dy1), (pixels * dx1), (pixels * dy1) + pixels, (pixels * dx1) + pixels, fill='blue', alpha=0.2)

            self.canvas_widget.moveto(self.obstacle6, pixels * dy1, pixels * dx1)


    # Function for cat movements 2:
    def cat2_move(self):
        global dy2, dx2, cat2_pos, cat_pos

        while flag:
            check2 = []
            time.sleep(0.5)
            if (dy2+1, dx2) in cat_avail_space2:    
                check2.append((dy2+1, dx2))

            if (dy2-1, dx2) in cat_avail_space2:
                check2.append((dy2-1, dx2))

            if (dy2, dx2+1) in cat_avail_space2:
                check2.append((dy2, dx2+1))

            if (dy2, dx2-1) in cat_avail_space2:
                check2.append((dy2, dx2-1))

            if (dy2-1, dx2-1) in cat_avail_space2:
                check2.append((dy2-1, dx2-1))

            if (dy2+1, dx2+1) in cat_avail_space2:
                check2.append((dy2+1, dx2+1))

            if (dy2-1, dx2+1) in cat_avail_space2:
                check2.append((dy2-1, dx2+1))

            if (dy2+1, dx2-1) in cat_avail_space2:
                check2.append((dy2+1, dx2-1))

            if len(check2) == 0:
                dyn_obj_loc = random.choice(cat_avail_space2)
                cat2_pos[dyn_obj_loc] += 1
                cat_pos[dyn_obj_loc] += 1
                dy2 = dyn_obj_loc[0]
                dx2 = dyn_obj_loc[1]
            else:
                random.shuffle(check2)
                dyn_obj_mov = random.choice(check2)
                cat2_pos[dyn_obj_mov] += 1
                cat_pos[dyn_obj_mov] += 1
                dy2 = dyn_obj_mov[0]
                dx2 = dyn_obj_mov[1]

                if cat_pos[dyn_obj_mov] == 1:
                    self.createRectangle((pixels * dy2), (pixels * dx2), (pixels * dy2) + pixels, (pixels * dx2) + pixels, fill='blue', alpha=0.1)
                if cat_pos[dyn_obj_mov] == 9:
                    self.createRectangle((pixels * dy2), (pixels * dx2), (pixels * dy2) + pixels, (pixels * dx2) + pixels, fill='blue', alpha=0.2)

            self.canvas_widget.moveto(self.obstacle7, pixels * dy2, pixels * dx2)


    # Function to reset the environment and start new Episode
    def reset(self):
        global goal1_visit
        self.update()

        # Updating agent
        self.canvas_widget.delete(self.agent)
        self.agent = self.canvas_widget.create_image(0, 0*pixels, anchor='nw', image=self.robot)

        # # Clearing the dictionary and the i
        self.d = {}
        self.i = 0
        goal1_visit = False

        # Return observation
        return self.canvas_widget.coords(self.agent)


    # Function to get the next observation and reward by doing next step
    def step(self, action):
        global goal1_visit

        reward = 0

        # Current state of the agent
        state = self.canvas_widget.coords(self.agent)
        base_action = np.array([0, 0])

        # Updating next state according to the action
        # Action 'up'
        if action == 0:
            if state[1] >= pixels:
                base_action[1] -= pixels

        elif action == 1:
            if state[1] < (env_height - 1) * pixels:
                base_action[1] += pixels
 
        elif action == 2:
            if state[0] < (env_width - 1) * pixels:
                base_action[0] += pixels

        elif action == 3:
            if state[0] >= pixels:
                base_action[0] -= pixels
  
        self.canvas_widget.move(self.agent, base_action[0], base_action[1])

        # Writing in the dictionary coordinates of found route
        self.d[self.i] = self.canvas_widget.coords(self.agent)

        # Updating next state
        next_state = self.d[self.i]
        next_pos = self.d[self.i]

        # Updating key for the dictionary
        self.i += 1

        # Calculating the reward for the agent
        if next_state == self.canvas_widget.coords(self.flag):
            if goal1_visit:
                reward = 500 - len(self.d)
                done = True
            else:
                done = False
            
            next_state = 'goal'

            # Filling the dictionary first time
            if self.c == True:
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
                self.c = False
                self.longest = len(self.d)
                self.shortest = len(self.d)

            # Checking if the currently found route is shorter
            if len(self.d) <= len(self.f):
                # Saving the number of steps for the shortest route
                self.shortest = len(self.d)
                # Clearing the dictionary for the final route
                self.f = {}
                # Reassigning the dictionary
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]

            # Saving the number of steps for the longest route
            if len(self.d) > self.longest:
                self.longest = len(self.d)

            if shortest_path_length:
                if len(self.d) <= shortest_path_length and goal1_visit:
                    # Saving the number of steps for the shortest route
                    self.shortest = len(self.d)
                    # Clearing the dictionary for the final route
                    self.f = {}
                    # Reassigning the dictionary
                    for j in range(len(self.d)):
                        self.f[j] = self.d[j]

                    for k in cat1_pos.keys():
                        if cat1_pos[k] != 0:
                            if [k[0] * pixels, k[1] * pixels] in self.f.values():
                                return next_state, reward, done, next_pos, False

                    for k in cat2_pos.keys():
                        if cat2_pos[k] != 0:
                            if [k[0] * pixels, k[1] * pixels] in self.f.values():
                                return next_state, reward, done, next_pos, False

                    return next_state, reward, done, next_pos, True


        elif tuple(next_state) in obj_spaces:
            reward = -100
            done = True
            next_state = 'obstacle'

            # Clearing the dictionary and the i
            self.d = {}
            self.i = 0
            goal1_visit = False

        elif tuple(next_state) in [goal_1]:
            done = False
            goal1_visit = True


        elif next_state in [self.canvas_widget.coords(self.obstacle6)]:
            
            done = True
            next_state = 'dyn_obs'
            reward = -1000

            # Clearing the dictionary and the i
            self.d = {}
            self.i = 0
            goal1_visit = False

        else:
            # reward = -1
            done = False

        return next_state, reward, done, next_pos, False


    # Function to refresh the environment
    def render(self):
        self.update()


    # Function to show the found route
    def final(self):
        global flag
        flag = 0
        # Deleting the agent at the end
        self.canvas_widget.delete(self.agent)

        # Showing the number of steps
        print('The shortest route:', self.shortest)
        print('The longest route:', self.longest)

        # Creating initial point
        origin = np.array([int(pixels/2), int(pixels/2)])
        self.initial_point = self.canvas_widget.create_oval(
            0 + origin[0] - ((10/64)*pixels), (19 * pixels) + origin[1] - ((10/64)*pixels),
            0 + origin[0] + ((10/64)*pixels), (19 * pixels) + origin[1] + ((10/64)*pixels),
            fill='red', outline='red')

        # Filling the route
        print('Final Route')
        for j in range(len(self.f)):
            # Showing the coordinates of the final route
            print(self.f[j])
            self.track = self.canvas_widget.create_oval(
                self.f[j][0] + origin[0] - ((10/64)*pixels), self.f[j][1] + origin[0] - ((10/64)*pixels),
                self.f[j][0] + origin[0] + ((10/64)*pixels), self.f[j][1] + origin[0] + ((10/64)*pixels),
                fill='red', outline='red')
            # Writing the final route in the global variable a
            a[j] = self.f[j]


    def cat1_curr_pos(self):
        return self.canvas_widget.coords(self.obstacle6)

    def cat2_curr_pos(self):
        return self.canvas_widget.coords(self.obstacle7)
    
    def check_pos(self, act):
        state = self.canvas_widget.coords(self.agent)
        base_action = np.array([0, 0])

        if act == 0:
            if state[1] >= pixels:
                base_action[1] -= pixels

        elif act == 1:
            if state[1] < (env_height - 1) * pixels:
                base_action[1] += pixels
 
        elif act == 2:
            if state[0] < (env_width - 1) * pixels:
                base_action[0] += pixels

        elif act == 3:
            if state[0] >= pixels:
                base_action[0] -= pixels

        state = [state[0] + base_action[0], state[1] + base_action[1]]

        if state == self.canvas_widget.coords(self.obstacle6):
            self.d = {}
            self.i = 0
            goal1_visit = False

        return (state, state == self.canvas_widget.coords(self.obstacle6))


# Returning the final dictionary with route coordinates
def final_states():
    return a


# This we need to debug the environment
# If we want to run and see the environment without running full algorithm
if __name__ == '__main__':
    env = Environment()
    env.mainloop()
