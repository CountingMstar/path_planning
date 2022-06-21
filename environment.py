import time
from turtle import distance
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import math

PhotoImage = ImageTk.PhotoImage
HEIGHT = 5  # 그리드 세로
WIDTH = 5  # 그리드 가로
UNIT = int(500/HEIGHT)  # 픽셀 수
SHAPESIZE = int(300/HEIGHT)  # 픽셀 수

global wall

np.random.seed(1)

class Env():
    def __init__(self):
        super(Env, self).__init__()
        self.map = np.zeros([HEIGHT,WIDTH])
        self.cleaned = -1
        self.uncleaned = 0
        self.action_size = 8
        self.cur_position = [0,0,2] # x,y,th
        self.pre_position = self.cur_position
        self.counter = 0
        self.render = False
        self.render_time = 0.1
        self.go_to_wall = False
        self.clean = False
        self.combo = 0

        # render
        self.tk = tk.Tk()
        self.tk.title('Path_planning')
        self.tk.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = [
            [PhotoImage(Image.open("./img/rectangle"+str(i)+".png").resize((30, 30))) for i in range(8)
            ],
            PhotoImage(Image.open("./img/wall.png").resize((30, 30))),
            PhotoImage(Image.open("./img/should_clean.png").resize((30, 30))),
            PhotoImage(Image.open("./img/cleaned.png").resize((30, 30))),
            PhotoImage(Image.open("./img/wall.png").resize((30, 30)))
        ]
        
        self.canvas = tk.Canvas(self.tk, bg='white', height=HEIGHT * UNIT, width=WIDTH * UNIT)
        self.canvas.pack()
        self.cuc = [0,0]

    def render_grid(self):
        if self.render:
            time.sleep(self.render_time)
            self.canvas.delete("all")

            # grid
            for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
                x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
                self.canvas.create_line(x0, y0, x1, y1)
            for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
                x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
                self.canvas.create_line(x0, y0, x1, y1)

            # shapes
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    if self.cur_position[0:2] == [i,j]:
                        img = self.shapes[0][self.cur_position[2]]
                    elif self.map[i,j] == self.uncleaned:
                        img = self.shapes[2]
                    elif self.map[i,j] == self.cleaned:
                        img = self.shapes[3]
                    x, y = (UNIT * i) + UNIT / 2, (UNIT * j) + UNIT / 2,
                    self.canvas.create_image(x, y, image=img)

            # state
            x, y = (self.cur_position[0]-1)*UNIT, (self.cur_position[1]-1)*UNIT
            self.canvas.create_line(x, y, x+UNIT*3, y, width=2, fill='red')
            self.canvas.create_line(x, y, x, y+UNIT*3, width=2, fill='red')
            self.canvas.create_line(x+UNIT*3, y, x+UNIT*3, y+UNIT*3, width=2, fill='red')
            self.canvas.create_line(x, y+UNIT*3, x+UNIT*3, y+UNIT*3, width=2, fill='red')

            x, y = self.cuc[0]*UNIT, self.cuc[1]*UNIT
            self.canvas.create_line(x, y, x+UNIT, y, width=2, fill='red')
            self.canvas.create_line(x, y, x, y+UNIT, width=2, fill='red')
            self.canvas.create_line(x+UNIT, y, x+UNIT, y+UNIT, width=2, fill='red')
            self.canvas.create_line(x, y+UNIT, x+UNIT, y+UNIT, width=2, fill='red')

            self.tk.update()

    def reset(self):
        self.map = np.zeros([HEIGHT,WIDTH])
        self.cur_position = [0,0,2]
        self.map[self.cur_position[0],self.cur_position[1]] = self.cleaned
        self.counter = 0
        self.combo = 0

        return self.get_state()

    def get_state(self):
        state = []

        # 현재 자신의 x좌표(정규화)
        state.append(round(self.cur_position[0]/WIDTH, 2))
        # 현재 자신의 y좌표(정규화)
        state.append(round(self.cur_position[1]/HEIGHT, 2))
        # 현재 자신의 각도(정규화)
        state.append(round(self.cur_position[2]/8, 2))

        # 로봇중심 3x3
        # [(x0,y0~y2), (x1,y0~y2), (x2,y0~y2)]
        # 벽: -2, cleaned = -1, uncleaned = 0
        difs = [-1,0,1]
        for i in difs:
            x = self.cur_position[0] + i
            for j in difs:
                y = self.cur_position[1] + j
                if i == j == 0:# 내위치
                    continue
                if not(0 <= x < HEIGHT) or not(0 <= y < WIDTH):# 바깥이면 -2
                    state.append(-2)
                else:
                    state.append(self.map[x][y])

        #cuc(제일 가까운 uncleaned타일 위치) 찾는과정
        dists = []
        for i in range(HEIGHT):
            dists.append([])
            for j in range(WIDTH):
                dist = 30
                if self.map[i][j] == self.uncleaned:
                    dx, dy = i-self.cur_position[0], j-self.cur_position[1]
                    dist = math.sqrt(dx*dx+dy*dy)
                dists[i].append(dist)
        dists = np.array(dists)
        dist_min = dists.min()
        mins = np.where(dists == dist_min)
        x_c, y_c = mins[0][0], mins[1][0]
        self.cuc = [x_c, y_c]

        # 현재 자신의 위치와 cuc의 상대위치(정규화)
        state.append(round((x_c - self.cur_position[0])/WIDTH, 2))
        state.append(round((y_c - self.cur_position[1])/HEIGHT, 2))

        return state

    def get_reward(self, turning_reward):
        x, y = self.cur_position[:2]
        reward = self.map[x,y]
        if self.map[x,y] == self.cleaned:
            if self.cur_position[:2] == self.pre_position[:2]:
                reward = -0.5
                if self.go_to_wall:
                    reward = -2
                    self.go_to_wall = False

        if self.clean:
            reward = 0
            self.clean = False

        # turning reward를 더해줌
        reward += turning_reward

        # combo
        # combo수의 지수함수배 만큼 추가 리워드를줌, combo가 끊어지면 리셋
        if reward >= 0:
            self.combo += 1
            reward += 1.5**(self.combo)
        if reward < 0:
            self.combo = 0

        return reward

    def get_done(self):
        if self.map.sum() == -HEIGHT*WIDTH:
            return True
        if self.counter >= 500:
            return True
        return False

    def step(self, action):
        self.counter += 1

        turning_reward = self.move(action)

        next_state = self.get_state()
        reward = self.get_reward(turning_reward)

        done = self.get_done()

        self.pre_position = self.cur_position
        
        self.render_grid()
        return next_state, reward, done

    def move(self, action):
        x, y = self.cur_position[:2]
        di = self.cur_position[2]

        # 0 ~ 7까지 8개의 action
        if action == 0:
            x = x
            y = y-1
            new_di = action
        if action == 1:
            x = x+1
            y = y-1
            new_di = action
        if action == 2:
            x = x+1
            y = y
            new_di = action
        if action == 3:
            x = x+1
            y = y+1
            new_di = action
        if action == 4:
            x = x
            y = y+1
            new_di = action
        if action == 5:
            x = x-1
            y = y+1
            new_di = action
        if action == 6:
            x = x-1
            y = y
            new_di = action
        if action == 7:
            x = x-1
            y = y-1
            new_di = action

        # 만약 grid밖으로 나가면 그 전위치 그대로
        if not(0 <= x < HEIGHT) or not(0 <= y < WIDTH):
            self.go_to_wall = True
            x, y = self.pre_position[:2]

        if self.map[x,y] == self.uncleaned:
            self.clean = True
            self.map[x,y] = self.cleaned
        self.cur_position = [x, y, di]

        # 목표위치로 가기위해 회전하는 만큼 리워드
        turning_reward = abs(di - new_di)
        k = 1/2
        if turning_reward > 3:
            turning_reward = -abs(8 - turning_reward)*k
        else:
            turning_reward = -turning_reward*k

        di = new_di
        self.cur_position = [x, y, di]

        return turning_reward