"""
Here we define some variables used for the game and rendering later
"""
#last frame time keeps track of which frame we are at
#last_frame_time = 0# not necessary
#translate the actions to human readable words 不需要，最后打印的时候直接调用结果即可。
#translate_action = ["Left","Stay","Right","Create Ball","End Test"]

# network
state_size = 10  #这里的statesize就是任务数量！！！表示最大接收的任务数量。
num_actions = state_size + 1  # [1000个任务+结束这一轨]
hidden_size = 100 # Size of the hidden layers
state_dim = 5 #输入数据的维度
model_save_path = "DQN.h5"
trainDQN = True
testDQN = True

# learning
discount = .9 # gamma
epsilon = .2  # exploration
max_memory = 10000 # Maximum number of experiences we are storing
batch_size = 100 # Number of experiences we use for training per batch
epoch = 20 # Number of games played in training, I found the model needs about 4,000 games till it plays well
scenes = 15 # Number of different scenes

# path
path_cc4_2 = "cc4_2.exe"
path_orbit = "CC4_2_orbit.dat"
path_task = "tasklist.txt"
path_timewindow = "outputtimewindow.txt"
path_outputattitude = "outputattitude.txt"
path_temp = "CC4_PARAM.DAT"
path_access = "CC4_2_ACCESS.TXT"
path_angle = "CC4_2_ANGLE.TXT"
path_shadow = "SHADOW.TXT"
path_workingdirectory = "C:\\Users\\Administrator\\Desktop\\yingyong\\Method1_DQN_Keras\\preperation\\"
path_dataset = "C:\\Users\\Administrator\\Desktop\\yingyong\\test"

# scheduling
algorithm = 4
verbose = 0 #开关，是否详细打印出结果

# environment
max_orbit = 16

low_lat = 3#-65#
high_lat = 53#65#
low_lon = 73#-180#
high_lon = 133#180#

lowest_profit = 1
highest_profit = 10
lowest_duration = 5
highest_duration = 10

#test
task_set_num = [400]
test_size = 10 # Test data sets
