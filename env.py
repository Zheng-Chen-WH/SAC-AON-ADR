import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
import math
from Orbit_Dynamics.CW_Prop import CW_Prop

'''每次都会忘的vsc快捷键：
    打开设置脚本：ctrl+shift+P
    多行注释：ctrl+/
    关闭多行注释：选中之后再来一次ctrl+/
    多行缩进：tab
    关闭多行缩进：选中多行shift+tab'''

class env:
    def __init__(self):
         #状态空间设置为无限连续状态空间，虽然不知道相比设成离散空间有什么影响
        self.attack_action_space=spaces.Box(low=np.array([-2.0,-2.0,-2.0,600]), high=np.array([2.0,2.0,2.0,1200]), shape=(4,), dtype=np.float32)  #降低收敛难度，设限
        self.defense_action_space=spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) 
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(13,),dtype=np.float32)
        self.blue_pos=np.array([np.random.uniform(250, 400)*1000,
                                np.random.uniform(-50, 50)*1000,
                                np.random.uniform(-100, 100)*1000])
        # self.red_pos=np.array([np.random.uniform(-50, 50)*1000,
        #                         np.random.uniform(-50, 50)*1000,
        #                         np.random.uniform(-20, 20)*1000])
        self.red_pos=np.array([0,0,0])
        self.distance=np.linalg.norm(self.blue_pos-self.red_pos)
        self.blue_vel= np.random.uniform(-10, 10, size=(3,))
        #self.red_vel= np.random.uniform(-5, 5, size=(3,))
        self.red_vel=np.array([0,0,0])
        self.target_distance=5000
        self.fuel=np.array([20])
        self.omega=2*math.pi/(24*3600)
        self.i=0
        self.info=0
        
    def three_point(self,blue_pos,blue_vel,red_pos,red_vel,time_interval): 
        state=[] 
        for i in range(3):
            blue_pos,blue_vel=CW_Prop(blue_pos,blue_vel,self.omega,time_interval*i)
            red_pos,red_vel=CW_Prop(red_pos,red_vel,self.omega,time_interval*i)
            x_vector=blue_vel/np.linalg.norm(blue_vel)
            y_vector=np.cross(blue_pos,blue_vel)/np.linalg.norm(np.cross(blue_pos,blue_vel))
            z_vector=np.cross(x_vector,y_vector)/np.linalg.norm(np.cross(x_vector,y_vector)) #采用VNC坐标系定义角度
            relative_pos=red_pos-blue_pos
            coordinate_trans_matrix=np.concatenate([[x_vector.T],[y_vector.T],[z_vector.T]],axis=0) #坐标转换矩阵，用不上但是先放在这里
            x_angle=math.degrees(math.acos(np.dot(relative_pos,x_vector)/np.linalg.norm(relative_pos))) #这里的三行就是把姿态转换矩阵拆了
            y_angle=math.degrees(math.acos(np.dot(relative_pos,y_vector)/np.linalg.norm(relative_pos)))
            #z_angle=np.dot(relative_pos,z_vector)/np.linalg.norm(relative_pos)
            state.append(x_angle)
            state.append(y_angle)
            state.append(blue_pos[0]/1e5)
            state.append(blue_pos[1]/1e5)
            state.append(blue_pos[2]/1e5)
        return state
    
    def angle_observe(self,blue_pos,blue_vel,red_pos,red_vel,time_interval): 
        blue_pos,blue_vel=CW_Prop(blue_pos,blue_vel,self.omega,time_interval)
        red_pos,red_vel=CW_Prop(red_pos,red_vel,self.omega,time_interval)
        relative_pos=red_pos-blue_pos
        x_angle=math.degrees(math.acos(relative_pos[0]/np.linalg.norm(relative_pos))) #这里的三行就是把姿态转换矩阵拆了
        y_angle=math.degrees(math.acos(relative_pos[1]/np.linalg.norm(relative_pos)))
        z_angle=math.degrees(math.acos(relative_pos[2]/np.linalg.norm(relative_pos)))
        return x_angle,z_angle

    def reset(self,seed=None): #,prop_t
        if seed is None:
            np.random.seed(self.i)
        else:
            np.random.seed(seed)
        super().__init__()
        self.blue_pos_list=[]
        #self.red_pos_list=[]
        self.info=0
        self.blue_pos=np.array([np.random.uniform(-105, -95)*1000,
                                 np.random.uniform(-5, 5)*1000,
                                 np.random.uniform(-20, -10)*1000])
        # self.red_pos=np.array([np.random.uniform(-5, 5)*1000,
        #                         np.random.uniform(0, 0)*1000,
        #                         np.random.uniform(-5, 5)*1000])
        self.blue_vel=np.array([np.random.uniform(1, 3),
                                 np.random.uniform(0, 0),
                                 np.random.uniform(1, 3)])
        # self.red_vel= np.random.uniform(-1, 1, size=(3,))
        # self.blue_pos=np.array([-300000,0,-15000])
        self.red_pos=np.array([0,0,0])
        self.red_vel=np.array([0,0,0])
        # self.blue_vel= np.array([10,0,10])
        self.blue_pos_list.append(self.blue_pos)
        # self.red_pos_list.append(self.red_pos)
        self.done=False
        self.fuel=np.array([50])
        self.init_distance=np.linalg.norm((self.blue_pos-self.red_pos))
        state=[]
        o1,o2=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,-300)
        state.append(o1)
        state.append(o2)
        o3,o4=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,-150)
        state.append(o3)
        state.append(o4)
        o5,o6=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,0)
        state.append(o5)
        state.append(o6)
        state.append(0)
        state.append(0)
        state.append(0)
        o7,o8=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,150)
        state.append(o7)
        state.append(o8)
        o9,o10=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,300)
        state.append(o9)
        state.append(o10)
        state=np.array(state)
        return state

    def step(self,blue_action):#,red_action
        state=[]
        o1,o2=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,-300)
        state.append(o1)
        state.append(o2)
        o3,o4=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,-150)
        state.append(o3)
        state.append(o4)
        o5,o6=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,0)
        state.append(o5)
        state.append(o6)
        state.append(blue_action[0])
        state.append(blue_action[1])
        state.append(blue_action[2])
        natural_blue_pos,_=CW_Prop(self.blue_pos,self.blue_vel,self.omega,blue_action[3])
        natural_red_pos,_=CW_Prop(self.red_pos,self.red_vel,self.omega,blue_action[3])
        self.blue_vel=self.blue_vel+blue_action[0:3]
        manueuver_blue_pos,_=CW_Prop(self.blue_pos,self.blue_vel,self.omega,300)
        manueuver_red_pos,_=CW_Prop(self.red_pos,self.red_vel,self.omega,300)
        o7,o8=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,150)
        state.append(o7)
        state.append(o8)
        o9,o10=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,300)
        state.append(o9)
        state.append(o10)
        state=np.array(state)
        next_blue_pos,self.blue_vel=CW_Prop(self.blue_pos,self.blue_vel,self.omega,blue_action[3])
        next_red_pos,self.red_vel=CW_Prop(self.red_pos,self.red_vel,self.omega,blue_action[3])
        self.blue_pos_list.append(next_blue_pos)
        self.blue_pos=next_blue_pos
        self.red_pos=next_red_pos
        state=np.array(state)
        self.fuel=np.array([self.fuel[0]-np.linalg.norm(blue_action[0:3])])
        #self.distance=np.linalg.norm(self.blue_pos-self.red_pos)

        if self.fuel[0]<0 or np.linalg.norm(next_blue_pos-next_red_pos)<=self.target_distance:
            self.done=True
        
        if not self.done:
            # natural_relative=natural_red_pos-natural_blue_pos
            # manueuver_relative=manueuver_red_pos-manueuver_blue_pos
            # obs_buff=np.dot(natural_relative,manueuver_relative)/(np.linalg.norm(natural_relative)*np.linalg.norm(manueuver_relative))
            # if obs_buff>0:
            #     obs_buff=5-obs_buff*5
            # else:
            #     obs_buff=-10
            reward=(np.linalg.norm(natural_blue_pos-natural_red_pos)-np.linalg.norm(next_blue_pos-next_red_pos))/1000 #+obs_buff

        if self.done:
            if self.fuel[0]<0:
                self.info=0 #用来区分越界和到达目标
                reward=0
            elif np.linalg.norm(next_blue_pos-next_red_pos)<=self.target_distance and self.fuel[0]>=0:
                self.info=1
                reward=400
            self.i+=1

        return state, reward, self.done, self.info

    def plot(self, args, data_x, data_y, data_z=None):
        font = {'family': 'serif',
         'serif': 'Times New Roman',
         'weight': 'normal',
         'size': 15,
         }
        plt.rc('font', **font)
        plt.style.use('seaborn-whitegrid')
        if data_z!=None and args['plot_type']=="3D-1line":
            fig = plt.figure()
            ax = fig.gca(projection='3d') #Axes对象是图形的绘图区域，可以在其上进行各种绘图操作。
                                        #通过gca()函数可以获取当前图形的Axes对象，如果该对象不存在，则会创建一个新的。
                                        #projection='3d' 参数指定了Axes对象的投影方式为3D，即创建一个三维坐标系。
            plt.plot(0,0,0,'r*') #画一个位于原点的星形
            plt.plot(data_x,data_y,data_z,'b',linewidth=1) #画三维图
            ax.set_xlabel('x/km', fontsize=15)
            ax.set_ylabel('y/km', fontsize=15)
            ax.set_zlabel('z/km', fontsize=15)
            ax.set_xlim(np.min(data_x),np.max(data_x))
            ax.set_ylim(np.min(data_y),np.max(data_y))
            ax.set_zlim(np.min(data_z),np.max(data_z))
            # ax.set_xlim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
            # ax.set_ylim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
            # ax.set_zlim(np.min([np.min(data_x),np.min(data_y),np.min(data_z)]),np.max([np.max(data_x),np.max(data_y),np.max(data_z)]))
            plt.tight_layout()# 调整布局使得图像不溢出
            plt.savefig('svg.svg', format='svg', bbox_inches='tight')# 'logs/{}epoch-{}steps.png'.format(epoch,steps))
            plt.show()
        elif data_z!=None and args['plot_type']=="2D-2line":
            fig = plt.figure()
            ax = fig.gca() #Axes对象是图形的绘图区域，可以在其上进行各种绘图操作。通过gca()函数可以获取当前图形的Axes对象，如果该对象不存在，则会创建一个新的。
            plt.plot(data_x,data_y,'b',linewidth=0.5)
            plt.plot(data_x,data_z,'g',linewidth=1)
            ax.set_xlabel('x', fontsize=15)
            ax.set_ylabel('y', fontsize=15)
            ax.set_xlim(np.min(data_x),np.max(data_x))
            ax.set_ylim(np.min([np.min(data_y),np.min(data_z)]),np.max([np.max(data_y),np.max(data_z)]))
            plt.tight_layout()# 调整布局使得图像不溢出
            plt.savefig(args['plot_title'], format='svg', bbox_inches='tight')# 'logs/{}epoch-{}steps.png'.format(epoch,steps))
    
    def plotstep(self,blue_action):
        state=[]
        o1,o2=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,-300)
        state.append(o1)
        state.append(o2)
        o3,o4=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,-150)
        state.append(o3)
        state.append(o4)
        o5,o6=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,0)
        state.append(o5)
        state.append(o6)
        state.append(blue_action[0])
        state.append(blue_action[1])
        state.append(blue_action[2])
        natural_blue_pos,_=CW_Prop(self.blue_pos,self.blue_vel,self.omega,blue_action[3])
        natural_red_pos,_=CW_Prop(self.red_pos,self.red_vel,self.omega,blue_action[3])
        self.blue_vel=self.blue_vel+blue_action[0:3]
        o7,o8=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,150)
        state.append(o7)
        state.append(o8)
        o9,o10=self.angle_observe(self.blue_pos,self.blue_vel,self.red_pos,self.red_vel,300)
        state.append(o9)
        state.append(o10)
        state=np.array(state)
        for i in range(20):
            plot_pos,_=CW_Prop(self.blue_pos,self.blue_vel,self.omega,blue_action[3]/20*i)
            self.blue_pos_list.append(plot_pos)
        next_blue_pos,self.blue_vel=CW_Prop(self.blue_pos,self.blue_vel,self.omega,blue_action[3])
        next_red_pos,self.red_vel=CW_Prop(self.red_pos,self.red_vel,self.omega,blue_action[3])
        self.blue_pos_list.append(next_blue_pos)
        self.blue_pos=next_blue_pos
        self.red_pos=next_red_pos
        state=np.array(state)
        self.fuel=np.array([self.fuel[0]-np.linalg.norm(blue_action[0:3])])
        #self.distance=np.linalg.norm(self.blue_pos-self.red_pos)

        if self.fuel[0]<0 or np.linalg.norm(next_blue_pos-next_red_pos)<=self.target_distance:
            self.done=True

        if self.done:
            if self.fuel[0]<0:
                self.info=0 #用来区分越界和到达目标
            elif np.linalg.norm(next_blue_pos-next_red_pos)<=self.target_distance and self.fuel[0]>=0:
                self.info=1
            self.i+=1

        return state, self.done, self.info, self.blue_pos_list