def stationary_nisland_iicr(m, n):
    from math import sqrt, exp

    alpha = m * n + n - 1
    delta = sqrt(alpha * alpha - 4 * m * (n - 1))
    coef_e1 = (delta - alpha) / (2 * (n - 1))
    coef_e2 = (-delta - alpha) / (2 * (n - 1))
    coef_f1 = (delta - alpha + 2 * m) / (2 * delta)
    coef_f2 = m * (n - 1) / delta 

    def iicr(t):
        e1 = exp(t * coef_e1)
        e2 = exp(t * coef_e2)
        f1 = coef_f1 * e1 + (1 - coef_f1) * e2
        f2 = coef_f2 * (e1 - e2)
        
        d_e1 = coef_e1 * e1
        d_e2 = coef_e2 * e2
        d_f1 = coef_f1 * d_e1 + (1 - coef_f1) * d_e2
        d_f2 = coef_f2 * (d_e1 - d_e2)

        return -(f1 + f2) / (d_f1 + d_f2)
    
    return iicr


def generate_panmictic_ms(migration_rate, islands):
	import os
	import subprocess
	import numpy as np
	from math import exp
	
	iicr = stationary_nisland_iicr(migration_rate, islands)

	time = [10**t for t in np.linspace(-2, 2, 100)]
	iicr = [iicr(t) for t in time]

	ms_command = ['./scrm', '2', str(ms_seq_simulations), 
		'-t', str(ms_theta), '-r', str(ms_recombination),
		str(ms_sites), '-p', '8']
	
	for i in range(len(time)):
		ms_command.extend(['-eN', str(0.5 * time[i]), str(iicr[i])])

	# run scrm command
	msout_filename = './psmc tests/results 01/' + '2019-01-21_19-12-06_PSMC_c01_t001_r01.pan.scrm'
	msout = open(msout_filename, 'w')
	proc = subprocess.Popen(ms_command, stdout = msout)
	proc.wait()
	msout.close()