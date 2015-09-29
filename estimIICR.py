#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import ConfigParser

def generate_MS_t2(ms_command):
    # Simulate T2 values using MS.
    # The input is a string containing the MS-command
    # The output is a list of float containing independent values of T2
    o = os.popen(ms_command).read()
    o = o.split('\n')
    t_obs = []
    for l in o:
        if l[:6] == 'time:\t':
            temp = l.split('\t')
            t_obs.append(float(temp[1]))
    return t_obs

def compute_real_history_from_ms_command(ms_command, N0):
    # Returns a function depending on the scenario found in the ms_command
    # First we compute the value of N0
    msc = ms_command.split(' ')

    # Case of instantaneous changes
    if ms_command.__contains__('-eN'):
        size_changes = ms_command.split(' -eN ')
        (t_k, alpha_k) = ([i.split(' ')[0] for i in size_changes[1:]], 
                             [j.split(' ')[1] for j in size_changes[1:]])
        t_k = [0]+[4*N0*float(t) for t in t_k]
        N_k = [N0]+[N0*float(alpha) for alpha in alpha_k]
        return ('-eN', t_k, N_k)
        # print 'case 1'
    # Case of exponential grow
    elif ms_command.__contains__('G'):
        alpha = float(msc[msc.index('-G') + 1])
        T = float(msc[msc.index('-G') + 3])
        return ('ExponGrow', [alpha, T, N0])
        # print 'exponnential grow'
    # StSI case
    elif ms_command.__contains__('-I'):
        n = int(msc[msc.index('-I') + 1])
        M = float(msc[msc.index('-I') + n+2])
        if msc[msc.index('-I') + 2] == '2':
            return ('StSI same_island', [n, M, N0])
        else:
            return ('StSI disctint_island', [n, M, N0])
    else:
        return ('-eN', [[0], [N0]])

def compute_empirical_dist(obs, x_vector='', dx=0):
    # This method computes the empirical distribution given the
    # observations.
    # The functions are evaluated in the x_vector parameter
    # by default x_vector is computed as a function of the data
    # by default the differences 'dx' are a vector 

    if x_vector == '':
        actual_x_vector = np.arange(0, max(obs)+0.1, 0.1)

    elif x_vector[-1]<=max(obs):
        actual_x_vector = list(x_vector)
        actual_x_vector.append(max(obs))
        actual_x_vector = np.array(x_vector)
    else:
        actual_x_vector = np.array(x_vector)
        
    if (dx == 0):
        dx = actual_x_vector[1:]-actual_x_vector[:-1]
        # Computes the cumulative distribution and the distribution
        x_vector_left = actual_x_vector[1:] - np.true_divide(dx,2)
        x_vector_right = actual_x_vector[1:] + np.true_divide(dx,2)
        x_vector_left = np.array([0,0] + list(x_vector_left))
        x_vector_right = np.array([0, actual_x_vector[0]+dx[0]] + list(x_vector_right))
        actual_dx = np.array([dx[0]]+list(dx))
    else:
        actual_dx = dx
        half_dx = np.true_divide(dx,2)
    
    counts, ignored_values = np.histogram(obs, bins = actual_x_vector)
    counts_left, ignored_values = np.histogram(obs, bins = x_vector_left)
    counts_right, ignored_values = np.histogram(obs, bins = x_vector_right)
    
    cdf_x = counts.cumsum()
    cdf_x = np.array([0]+list(cdf_x))
    cdf_left = counts_left.cumsum()
    cdf_right = counts_right.cumsum()
    
    """
    # Normalizing
    cdf_obs_x = np.true_divide(cdf_x,len(obs))
    cdf_left = np.true_divide(cdf_left, len(obs))
    cdf_right = np.true_divide(cdf_right, len(obs))
    """

    # now we compute the pdf (the derivative of the cdf)
    dy = cdf_right - cdf_left
    pdf_obs_x = np.true_divide(dy, actual_dx)

    return (cdf_x, pdf_obs_x)
    
class ParamsLoader():
    # We use it for loading the parameters from a file
    def __init__(self, path2params='./parameters.txt'):
        self.path2params = path2params
        [self.path2ms, self.ms_command, self.dx, self.original_time_interval, 
         self.pattern, self.N0, self.g_time, self.plot_real, 
         self.plot_limits, self.n_rep] = self.load_parameters()
        self.times_vector = self.group_t(self.original_time_interval, 
                                          self.pattern)

    def group_t(self, time_interval, pattern):
        # Groupes the time following the pattern as specifyed in the psmc
        # documentation
        constant_blocks = pattern.split('+')
        t = list(time_interval)
        t = t[:]+t[-1:]
        temp = [t[0]]
        current_pos = 0
        for b in constant_blocks:
            if b.__contains__('*'):
                n_of_blocks = int(b.split('*')[0])
                size_of_blocks = int(b.split('*')[1])
                for i in xrange(n_of_blocks):
                    temp.append(t[current_pos+size_of_blocks])
                    current_pos+=size_of_blocks
            else:
                size_of_blocks = int(b)
                temp.append(t[current_pos+size_of_blocks])
                current_pos+=size_of_blocks
        return np.array(temp)
    
    def load_parameters(self):
        parser = ConfigParser.ConfigParser()
        parser.read(self.path2params)
        
        path2ms = parser.get('ms_parameters', 'path2ms')
        ms_command = parser.get('ms_parameters', 'ms_command')
        dx = float(parser.get('computation_parameters', 'dx'))
        if parser.get('custom_x_vector', 'set_custom_xvector') == 'False':
            start = float(parser.get('computation_parameters', 'start'))
            end = float(parser.get('computation_parameters', 'end'))
            number_of_values = int(parser.get('computation_parameters', 
                                              'number_of_values'))
            vector_type = parser.get('computation_parameters', 'x_vector_type')
            if vector_type == 'linear':
                x_vector = np.linspace(start, end, number_of_values)
            elif vector_type == 'log':
                n = number_of_values
                x_vector = [0.1*(np.exp(i * np.log(1+10*end)/n)-1)
                            for i in range(n+1)]
                x_vector[0] = x_vector[0]+start
            else:
                # For the moment, the default output is a linspace distribution
                x_vector = np.linspace(start, end, number_of_values)
        x_vector = np.array(x_vector)
        pattern = parser.get('computation_parameters', 'pattern')
        N0 = float(parser.get('scale_params', 'N0'))
        g_time = int(parser.get('scale_params', 'generation_time'))
        plot_real = int(parser.get('plot_params', 'plot_real_ms_history'))==1
        limits = parser.get('plot_params', 'plot_limits')
        plot_limits = [float(i) for i in limits.split(',')]
        n_rep = int(parser.get('number_of_repetitions', 'n_rep'))
        return [path2ms, ms_command, dx, x_vector, pattern, N0, g_time, 
                plot_real, plot_limits, n_rep]

if __name__ == "__main__":
    p = ParamsLoader()
    ms_full_cmd = os.path.join(p.path2ms, p.ms_command)

    empirical_histories = []
    # Do n independent simulations     
    for i in range(p.n_rep):
        obs = generate_MS_t2(ms_full_cmd)
        obs = 2*np.array(obs) # Given that in ms time is scaled to 4N0 and 
        # our model scales times to 2N0, we multiply the output of MS by 2.
        (F_x, f_x) = compute_empirical_dist(obs, p.times_vector, p.dx)
        F_x = np.array(F_x)
        x = np.array(p.times_vector)
        empirical_lambda = np.true_divide(len(obs)-F_x, f_x)
        empirical_histories.append((x, empirical_lambda))
    # empirical_lambda = np.true_divide((1-F_x[:-1])*(x[1:]-x[:-1]), 
    #                                  F_x[1:]-F_x[:-1])

    # Do the plot    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    for (x, empirical_lambda) in empirical_histories:
        x[0] = float(x[1])/5 # this is for avoiding to have x[0]=0 in a logscale
        ax.step(2 * p.N0 * p.g_time*x, p.N0 * empirical_lambda, '-r', 
            where='post', label='empirical lambda from MS')
    
    # Plot the real history (if commanded)
    if p.plot_real:
        [case, x, y] = compute_real_history_from_ms_command(p.ms_command, p.N0)
        print(case)
        print(x)
        print(y)
        x[0] = min(float(x[1])/5, p.plot_limits[2]) # this is for avoiding 
        # to have x[0]=0 in a logscale
        x.append(1e7) # adding the last value 
        y.append(y[-1])
        ax.step(x, y, '-b', where='post', label='Real history')
    
    ax.set_xlabel('Time (in years)')
    ax.set_ylabel(r'Coalescence rates $\lambda(t)$')
    ax.set_xscale('log')
    
    plt.legend(loc='best')
    plt.xlim(p.plot_limits[0], p.plot_limits[1])
    plt.ylim(p.plot_limits[2], p.plot_limits[3])
    plt.show()
    
    #fig.savefig('./plot.png', dpi=300)
