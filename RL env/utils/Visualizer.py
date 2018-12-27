import numpy as np
import os
import ntpath
import time
import visdom





class Visualizer():
    def __init__(self, actor_num,vis,num_actions):

            self.vis = vis
            self.num_actions = num_actions
            self.name= 'Actor num '+ str(actor_num)
            self.plot_data = {'X': [], 'Y': [], 'legend': 'Actor '+self.name+' Reward'}
            self.total_ae_plot ={}
            self.minimized_actions_counter = {value:[] for value in range(num_actions)}
            self.global_step = []
            self.q_values_lower = {value:[] for value in range(num_actions)}
            self.q_values_upper = {value:[] for value in range(num_actions)}
            self.Vlow = []



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


    def plot_total_ae_counter(self, global_step, minimized_actions_counter, action_meanings):

        self.global_step.append(global_step)


        for  i, action in enumerate(action_meanings):
            self.minimized_actions_counter[i].append(minimized_actions_counter[action])

            if global_step % 5 == 0:

                self.vis.line(
                X = np.array(self.global_step),
                Y = np.array(self.minimized_actions_counter[i]),
                win = "actions eliminated",
                name = action,
                update='append',
                )



    def plot_q_values(self,global_step, q_values_lower, q_values_upper,action_meanings):

        self.global_step.append(global_step)
        self.Vlow.append(max(q_values_lower))
        for  i in range(self.num_actions):
            self.q_values_upper[i].append(q_values_upper[i])


            if self.global_step[-1] % 5 == 0:

                self.vis.line(
                X = np.array(self.global_step),
                Y = np.array(self.q_values_upper[i]),
                win = "q_values",
                name = str(action_meanings[i]) + " upper",
                update='append',
                )

        if self.global_step[-1] % 5 == 0:

            self.vis.line(
            X = np.array(self.global_step),
            #Y = np.array(self.q_values_lower[i]),
            Y = np.array(self.Vlow),
            win = "q_values",
            name = "Vlow",
            update='append',
            )

        # if global_step%5 == 0:
        #     self.vis.line(
        #         X=np.array(self.plot_data['X']),
        #         Y=np.array(self.plot_data['Y']),
        #         opts={
        #             'title': self.name + ' actions eliminated',
        #             #'legend': self.plot_data['legend'],
        #             'xlabel': 'Step',
        #             'ylabel': 'Running AVG'},
        #         win= " Accumulated eliminations by action")
