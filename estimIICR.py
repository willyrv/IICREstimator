#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from scipy import misc
import argparse

def generate_MS_tk(ms_command):
    # Simulate T2 values using MS.
    # The input is a string containing the MS-command
    # The output is a list of float containing independent values of Tk
    # where Tk is the first coalescent event of the sample
    o = os.popen(ms_command).read()
    newick_re = "\([(0-9.,:)]+\)" # Find the tree line
    newick_pattern = re.compile(newick_re)
    single_coal_re = "\([0-9.,:]+\)"
    single_coal_pattern = re.compile(single_coal_re)
    t_obs = []
    for newick_line in newick_pattern.finditer(o):
        newick_text = newick_line.group()
        coal_times = []
        for single_coal_event in single_coal_pattern.finditer(newick_text):
            matched_text = single_coal_event.group()
            coal_time = float(matched_text.split(':')[1].split(',')[0])
            coal_times.append(coal_time)
        t_obs.append(min(coal_times))
    return t_obs

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

def compute_empirical_dist(obs, x_vector=''):
    # This method computes the empirical distribution given the
    # observations.
    # The functions are evaluated in the x_vector parameter
    # by default x_vector is computed as a function of the data
    # by default the differences 'dx' are a vector 

    if x_vector == '':
        actual_x_vector = np.arange(0, max(obs)+0.1, 0.1) 

    elif x_vector[-1]<=max(obs): # extend the vector to cover all the data
        actual_x_vector = list(x_vector)
        actual_x_vector.append(max(obs))
        actual_x_vector = np.array(x_vector)
    else:
        actual_x_vector = np.array(x_vector)
        
    actual_x_vector[0] = 0 # The first element of actual_x_vector should be 0
    
    half_dx = np.true_divide(actual_x_vector[1:]-actual_x_vector[:-1], 2)
    # Computes the cumulative distribution and the distribution
    x_vector_shift = actual_x_vector[:-1] + half_dx
    x_vector_shift = np.array([0] + list(x_vector_shift) + 
                                [actual_x_vector[-1]+half_dx[-1]])
    
    counts = np.histogram(obs, bins = actual_x_vector)[0]
    counts_shift = np.histogram(obs, bins = x_vector_shift)[0]
    
    cdf_x = counts.cumsum()
    cdf_x = np.array([0]+list(cdf_x))
    
    # now we compute the pdf (the derivative of the cdf)
    dy_shift = counts_shift
    dx_shift = x_vector_shift[1:] - x_vector_shift[:-1]
    pdf_obs_x = np.true_divide(dy_shift, dx_shift)

    return (cdf_x, pdf_obs_x)

def compute_t_vector(start, end, number_of_values, vector_type):
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
    return np.array(x_vector)

def group_t(time_interval, pattern):
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
            for i in range(n_of_blocks):
                temp.append(t[current_pos+size_of_blocks])
                current_pos+=size_of_blocks
        else:
            size_of_blocks = int(b)
            temp.append(t[current_pos+size_of_blocks])
            current_pos+=size_of_blocks
    return np.array(temp)

def compute_IICR_n_islands(n, M, t, s=True):
    # This method evaluates the lambda function in a vector
    # of time values t.
    # If 's' is True we are in the case when two individuals where
    # sampled from the same island. If 's' is false, then the two
    # individuals where sampled from different islands.

    # Computing constants
    gamma = np.true_divide(M, n-1)
    delta = (1+n*gamma)**2 - 4*gamma
    alpha = 0.5*(1+n*gamma + np.sqrt(delta))
    beta =  0.5*(1+n*gamma - np.sqrt(delta))

    # Now we evaluate
    x_vector = t
    if s:
        numerator = (1-beta)*np.exp(-alpha*x_vector) + (alpha-1)*np.exp(-beta*x_vector)
        denominator = (alpha-gamma)*np.exp(-alpha*x_vector) + (gamma-beta)*np.exp(-beta*x_vector)
    else:
        numerator = beta*np.exp(-alpha*(x_vector)) - alpha*np.exp(-beta*(x_vector))
        denominator = gamma * (np.exp(-alpha*(x_vector)) - np.exp(-beta*(x_vector)))

    lambda_t = np.true_divide(numerator, denominator)

    return lambda_t

def plotJson(jsonFilename, ax):
    # Do the corresponding plots in a json parameters file
    #
    # Returns
    #   a plot object
    with open(jsonFilename) as json_params:
        p = json.load(json_params)
    
    times_vector = []
    if p["custom_x_vector"]["set_custom_xvector"] == 0:
        start = p["computation_parameters"]["start"]
        end = p["computation_parameters"]["end"]
        number_of_values = p["computation_parameters"]["number_of_values"]
        vector_type = p["computation_parameters"]["x_vector_type"]
        t_vector = compute_t_vector(start, end, number_of_values, vector_type)
        pattern = p["computation_parameters"]["pattern"]
        times_vector = group_t(t_vector, pattern)
    else:
        times_vector = np.array(p["custom_x_vector"]["x_vector"])

    empirical_densities = []
    empirical_histories = []
    # Do n independent simulations     
    for i in range(len(p["scenarios"])):
        ms_full_cmd = os.path.join(p["path2ms"], p["scenarios"][i]["ms_command"])
        obs = generate_MS_tk(ms_full_cmd)
        obs = 2*np.array(obs) # Given that in ms time is scaled to 4N0 and 
        # our model scales times to 2N0, we multiply the output of MS by 2.
        (F_x, f_x) = compute_empirical_dist(obs, times_vector)
        empirical_densities.append(np.true_divide(np.array(f_x), sum(np.array(f_x))))
        F_x = np.array(F_x)
        x = times_vector
        # If the sample size on the ms command is greater than 2
        # the IICR that we obtain when the sample size is 2
        # must be multiplied by a factor
        
        # Parsing the ms command for getting the sample size
        ms_command = p["scenarios"][i]["ms_command"]
        sample_size = int(ms_command.split("ms ")[1].split(" ")[0])
        factor = misc.comb(sample_size, 2)
        
        empirical_lambda = factor * np.true_divide(len(obs)-F_x, f_x)
        empirical_histories.append((x, empirical_lambda))

    # Do the plot    
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    
    N0 = p["scale_params"]["N0"]
    g_time = p["scale_params"]["generation_time"]
    for i in range(len(empirical_histories)):
        (x, empirical_lambda) = empirical_histories[i]
        
        # Avoiding to have x[0]=0 in a logscale
        if x[0] == 0:
            x[0] = float(x[1])/100

        linecolor = p["scenarios"][i]["color"]
        line_style = p["scenarios"][i]["linestyle"]
        linewidth = p["scenarios"][i]["linewidth"]
        alpha = p["scenarios"][i]["alpha"]
        plot_label = p["scenarios"][i]["label"]
        ax.plot(2 * N0 * g_time*x, N0 * empirical_lambda, color = linecolor,
                ls=line_style, linewidth=linewidth, drawstyle='steps-post', alpha=alpha, label=plot_label)
    
    # Draw the vertical lines (if specifyed)
    for vl in p["vertical_lines"]:
      ax.axvline(vl, color='k', ls='--')
      
    # Plot the real history (if commanded)
    if p["plot_params"]["plot_real_ms_history"]:
        [case, x, y] = compute_real_history_from_ms_command(p.ms_command, p.N0)
        print(case)
        print(x)
        print(y)
        x[0] = min(float(x[1])/5, p.plot_limits[2]) # this is for avoiding 
        # to have x[0]=0 in a logscale
        x.append(1e7) # adding the last value 
        y.append(y[-1])
        ax.step(x, y, '-b', where='post', label='Real history')
        
    if p["plot_params"]["plot_theor_IICR"]:
        theoretical_IICR_list = []
        T_max = np.log10(p["plot_params"]["plot_limits"][1])
        t_k = np.logspace(1, T_max, 1000)
        t_k = np.true_divide(t_k, 2 * N0 * g_time)
        for i in range(len(p["theoretical_IICR_nisland"])):
            n = p["theoretical_IICR_nisland"][i]["n"]
            M = p["theoretical_IICR_nisland"][i]["M"]
            same_island = p["theoretical_IICR_nisland"][i]["sampling_same_island"]
            theoretical_IICR_list.append(compute_IICR_n_islands(n, M, t_k, 
                                                                same_island))
            
        # Plotting the theoretical IICR
        for i in range(len(p["theoretical_IICR_nisland"])):
            linecolor = p["theoretical_IICR_nisland"][i]["color"]
            line_style = p["theoretical_IICR_nisland"][i]["linestyle"]
            linewidth = p["theoretical_IICR_nisland"][i]["linewidth"]
            alpha = p["theoretical_IICR_nisland"][i]["alpha"]        
            plot_label = p["theoretical_IICR_nisland"][i]["label"]
            ax.plot(2 * N0 * g_time * t_k, N0 * theoretical_IICR_list[i],
                color=linecolor, ls=line_style, alpha=alpha, label=plot_label)

    # Plotting constant piecewise functions (if any)
    if "peicewise_constant_functions" in p:
        for f in p["peicewise_constant_functions"]:
            x = f["x"]
            y = f["y"]
            plot_label = f["label"]
            linecolor = f["color"]
            line_style = f["linestyle"]
            line_width = f["linewidth"]
            line_alpha = f["alpha"]
            ax.step(x, y, where='post', color=linecolor, ls=line_style, linewidth=line_width,
                     alpha=line_alpha, label=plot_label)
    ax.set_xlabel(p["plot_params"]["plot_xlabel"])
    ax.set_ylabel(p["plot_params"]["plot_ylabel"])
    if "y_scale" in p["plot_params"]:
        if p["plot_params"]["y_scale"] == "log":
            ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend(loc='best')
    [x_a, x_b, y_a, y_b] = p["plot_params"]["plot_limits"]
    plt.xlim(x_a, x_b)
    plt.ylim(y_a, y_b)
    if "plot_title" in p["plot_params"]:
      ax.set_title(p["plot_params"]["plot_title"])
    return ax
    # plt.show()
    
def get_PSMC_IICR(filename):
    a = open(filename, 'r')
    result = a.read()
    a.close()

    # getting the time windows and the lambda values
    last_block = result.split('//\n')[-2]
    last_block = last_block.split('\n')
    time_windows = []
    estimated_lambdas = []
    for line in last_block:
        if line[:2]=='RS':
            time_windows.append(float(line.split('\t')[2]))
            estimated_lambdas.append(float(line.split('\t')[3]))


    # getting the estimations of theta and N0
    result = result.split('PA\t') # The 'PA' lines contain the estimated lambda values
    result = result[-1].split('\n')[0]
    result = result.split(' ')
    #theta = float(result[1])
    #N0 = theta/(4*args.mutation_rate)/args.bin_size
    return(time_windows, estimated_lambdas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate T2 values with ms then plot the IICR')
    parser.add_argument('params_file', type=str,
                    help='the filename of the parameters')
    args = parser.parse_args()
    with open(args.params_file) as json_params:
        p = json.load(json_params)
    
    times_vector = []
    if p["custom_x_vector"]["set_custom_xvector"] == 0:
        start = p["computation_parameters"]["start"]
        end = p["computation_parameters"]["end"]
        number_of_values = p["computation_parameters"]["number_of_values"]
        vector_type = p["computation_parameters"]["x_vector_type"]
        t_vector = compute_t_vector(start, end, number_of_values, vector_type)
        pattern = p["computation_parameters"]["pattern"]
        times_vector = group_t(t_vector, pattern)
    else:
        times_vector = np.array(p["custom_x_vector"]["x_vector"])

    empirical_densities = []
    empirical_histories = []
    # Do n independent simulations     
    for i in range(len(p["scenarios"])):
        ms_full_cmd = os.path.join(p["path2ms"], p["scenarios"][i]["ms_command"])
        obs = generate_MS_tk(ms_full_cmd)
        obs = 2*np.array(obs) # Given that in ms time is scaled to 4N0 and 
        # our model scales times to 2N0, we multiply the output of MS by 2.
        (F_x, f_x) = compute_empirical_dist(obs, times_vector)
        empirical_densities.append(np.true_divide(np.array(f_x), sum(np.array(f_x))))
        F_x = np.array(F_x)
        x = times_vector
        # If the sample size on the ms command is greater than 2
        # the IICR that we obtain when the sample size is 2
        # must be multiplied by a factor
        
        # Parsing the ms command for getting the sample size
        ms_command = p["scenarios"][i]["ms_command"]
        sample_size = int(ms_command.split("ms ")[1].split(" ")[0])
        factor = misc.comb(sample_size, 2)
        
        empirical_lambda = factor * np.true_divide(len(obs)-F_x, f_x)
        empirical_histories.append((x, empirical_lambda))

    # Do the plot    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    N0 = p["scale_params"]["N0"]
    g_time = p["scale_params"]["generation_time"]
    if "use_real_data" in p:
        for d in p["use_real_data"]:
            (t_real_data, IICR_real_data) = get_PSMC_IICR(d["psmc_results_file"])
            t_real_data = np.array(t_real_data)
            IICR_real_data = np.array(IICR_real_data)
            plot_label = d["label"]
            linecolor = d["color"]
            line_style = d["linestyle"]
            linewidth = d["linewidth"]
            alpha = d["alpha"]
            ax.plot(2 * N0 * g_time * t_real_data, N0 * IICR_real_data, color = linecolor,
                ls=line_style, linewidth=linewidth, drawstyle='steps-post', alpha=alpha, label=plot_label)
                
    for i in range(len(empirical_histories)):
        (x, empirical_lambda) = empirical_histories[i]
        
        # Avoiding to have x[0]=0 in a logscale
        if x[0] == 0:
            x[0] = float(x[1])/100

        linecolor = p["scenarios"][i]["color"]
        line_style = p["scenarios"][i]["linestyle"]
        linewidth = p["scenarios"][i]["linewidth"]
        alpha = p["scenarios"][i]["alpha"]
        plot_label = p["scenarios"][i]["label"]
        ax.plot(2 * N0 * g_time*x, N0 * empirical_lambda, color = linecolor,
                ls=line_style, linewidth=linewidth, drawstyle='steps-post', alpha=alpha, label=plot_label)
    
    # Draw the vertical lines (if specifyed)
    for vl in p["vertical_lines"]:
      ax.axvline(vl, color='k', ls='--')
      
    # Plot the real history (if commanded)
    if p["plot_params"]["plot_real_ms_history"]:
        [case, x, y] = compute_real_history_from_ms_command(p.ms_command, p.N0)
        print(case)
        print(x)
        print(y)
        x[0] = min(float(x[1])/5, p.plot_limits[2]) # this is for avoiding 
        # to have x[0]=0 in a logscale
        x.append(1e7) # adding the last value 
        y.append(y[-1])
        ax.step(x, y, '-b', where='post', label='Real history')
        
    if p["plot_params"]["plot_theor_IICR"]:
        theoretical_IICR_list = []
        T_max = np.log10(p["plot_params"]["plot_limits"][1])
        t_k = np.logspace(1, T_max, 1000)
        t_k = np.true_divide(t_k, 2 * N0 * g_time)
        for i in range(len(p["theoretical_IICR_nisland"])):
            n = p["theoretical_IICR_nisland"][i]["n"]
            M = p["theoretical_IICR_nisland"][i]["M"]
            same_island = p["theoretical_IICR_nisland"][i]["sampling_same_island"]
            theoretical_IICR_list.append(compute_IICR_n_islands(n, M, t_k, 
                                                                same_island))
            
        # Plotting the theoretical IICR
        for i in range(len(p["theoretical_IICR_nisland"])):
            linecolor = p["theoretical_IICR_nisland"][i]["color"]
            line_style = p["theoretical_IICR_nisland"][i]["linestyle"]
            linewidth = p["theoretical_IICR_nisland"][i]["linewidth"]
            alpha = p["theoretical_IICR_nisland"][i]["alpha"]        
            plot_label = p["theoretical_IICR_nisland"][i]["label"]
            ax.plot(2 * N0 * g_time * t_k, N0 * theoretical_IICR_list[i],
                color=linecolor, ls=line_style, alpha=alpha, label=plot_label)

    # Plotting constant piecewise functions (if any)
    if "peicewise_constant_functions" in p:
        for f in p["peicewise_constant_functions"]:
            x = f["x"]
            y = f["y"]
            plot_label = f["label"]
            linecolor = f["color"]
            line_style = f["linestyle"]
            line_width = f["linewidth"]
            line_alpha = f["alpha"]
            ax.step(x, y, where='post', color=linecolor, ls=line_style, linewidth=line_width,
                    alpha=line_alpha, label=plot_label)

    ax.set_xlabel(p["plot_params"]["plot_xlabel"])
    ax.set_ylabel(p["plot_params"]["plot_ylabel"])
    if "y_scale" in p["plot_params"]:
        if p["plot_params"]["y_scale"] == "log":
            ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend(loc='best')
    [x_a, x_b, y_a, y_b] = p["plot_params"]["plot_limits"]
    plt.xlim(x_a, x_b)
    plt.ylim(y_a, y_b)
    if "plot_title" in p["plot_params"]:
      ax.set_title(p["plot_params"]["plot_title"])
    plt.show()
    # Plotting the densities
    if "plot_densities" in p:
        if len(p["plot_densities"]["densities_to_plot"])>0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in p["plot_densities"]["densities_to_plot"]:
                l = p["scenarios"][i]["label"]
                c = p["scenarios"][i]["color"]
                s = p["scenarios"][i]["linestyle"]
                a = p["scenarios"][i]["alpha"]
                ax.step(times_vector, empirical_densities[i], color=c, ls=s, 
                        alpha = a, label = l)
            plt.title("Density of T2")
            plt.xlim(p["plot_densities"]["x_lim"][0], p["plot_densities"]["x_lim"][1])
            plt.ylim(p["plot_densities"]["y_lim"][0], p["plot_densities"]["y_lim"][1])
            plt.legend(loc='best')
            plt.show()