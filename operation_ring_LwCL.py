import os
import time
import shutil
t0=time.strftime("%Y_%m_%d_%H_%M_%S")
args='modelring_LwCL'
actlist=('relu')
# methodlist=('WGAN','prox','LC','SN','GAN')
# methodlist=('prox_rhg','WGAN_rhg','LC_rhg','RHG')
# methodlist=('prox_cg','WGAN_cg','LC_cg','CG')
methodlist=('prox_fixed_point','WGAN_fixed_point','LC_fixed_point','CG')
act='tanh'


folder=os.getcwd()
folder=os.path.join(folder,args,t0)
print(folder)
if not os.path.exists(folder):
    os.makedirs(folder)
batfolder=os.path.join(folder,'set.bat')
ganfolder=os.path.join(folder,'unrolled_gan_rhg_ring_LwCL.py')
gaussfolder=os.path.join(folder,'mixture_gaussian_ring.py')
shutil.copyfile('unrolled_gan_rhg_ring_LwCL.py',ganfolder)
shutil.copyfile('mixture_gaussian_ring.py',gaussfolder)
with open(batfolder,'w') as f:
    for m in methodlist:

        for q in range(1):
            for glr in range(1):
                for dlr in range(1):
                    if m=='prox_rhg':
                        f.write('python unrolled_gan_rhg_ring_LwCL.py --method {} --us {} --glr {} --dlr {} --act {} --IV --r 8 --num_iter 10000 \n'.format(m, 1* (q + 1),0.1**(glr+3),0.1**(dlr+4),act))
                    else:
                        f.write(
                            'python unrolled_gan_rhg_ring_LwCL.py --method {} --us {} --glr {} --dlr {} --act {} --IV --r 8 --num_iter 10000 \n'.format(
                                m, 5 * (q + 2), 0.1 ** (glr + 3), 0.1 ** (dlr + 4), act))
os.chdir(folder)
os.system(batfolder)

