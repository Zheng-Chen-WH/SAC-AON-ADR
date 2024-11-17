SAC v3.0  
**state:** 3-point observation, in this version means 2 angles and the impulse at 5 time points.
>2 angles represent the angle between relative vector and x and y axis in VVLH (rather than VNC) coordinates

**initialstate:** self.blue_pos=np.array([np.random.uniform(-105, -95)*1000,np.random.uniform(-5, 5)*1000,np.random.uniform(-20, -10)*1000]),self.blue_vel=np.array([np.random.uniform(1, 3),np.random.uniform(0, 0),np.random.uniform(1, 3)]),self.red_pos=np.array([0,0,0]),self.red_vel=np.array([0,0,0])  
**action：**[ax,ay,az,time]  
>time:time interval until next impulse  

**training target:** reach 5km within target   
**done:** reach target (success) or run out of fuel(fail)  
**reward：** (np.linalg.norm(natural_blue_pos-natural_red_pos)-np.linalg.norm(next_blue_pos-next_red_pos))/1000, only use shaping reward and. The shaping reward is the distance between maneuvered motion and free drift.  
result as follow, training basically succeed.
![Alt text](reward-steps-1.png)
**note:** after simplifying the initial state, the training succeeded, and shows good generalization ability. And now the training process is documented via Tensorboard, the log file is in the ./runs folder.      
**analysis:** Finally succeeded. Possible future work includes examining the influencing factor of generalization ability and learning efficiency, transfer learning, including more bonds to reward function, comparison between RL-based AON and other algorithms, and add measurement error to the state.