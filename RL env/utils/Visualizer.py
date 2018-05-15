import numpy as np
import os
import ntpath
import time
import visdom






class Visualizer():
    def __init__(self, actor_num,vis):

            self.vis = vis
            self.name= 'Actor num '+ str(actor_num)
            self.plot_data = {'X': [], 'Y': [], 'legend': 'Actor '+self.name+' Reward'}






    # errors: dictionary of error labels and values
    def plot_current_errors(self, global_step, avg_reward):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': 'Actor '+self.name+' Reward'}
        #print ("We are in")
        self.plot_data['X'].append(global_step)
    #    print ("List X from visualizer")
    #    print ('[%s]' % ', '.join(map(str, self.plot_data['X'])))
        self.plot_data['Y'].append(avg_reward)
    #    print (np.array(self.plot_data['X']))
        if global_step%5 == 0:
            self.vis.line(
                X=np.array(self.plot_data['X']),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' Reward avg over steps',
                    #'legend': self.plot_data['legend'],
                    'xlabel': 'Step',
                    'ylabel': 'Running AVG'},
                win=self.name)
