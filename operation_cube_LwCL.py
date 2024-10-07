import os
import time
import shutil
t0=time.strftime("%Y_%m_%d_%H_%M_%S")
args='cube_us_DF1_LwCL'
actlist=('relu','tanh')
# methodlist=('WGAN','prox','LC','SN','GAN')
# m='WGAN'
methodlist=('prox_rhg','LC_rhg','RHG')
# methodlist=('LC_cg','prox_cg','WGAN_cg','CG')

folder=os.getcwd()
folder=os.path.join(folder,args,t0)
print(folder)
if not os.path.exists(folder):
    os.makedirs(folder)
batfolder=os.path.join(folder,'set.bat')
ganfolder=os.path.join(folder,'unrolled_gan_rhg_cube_LwCL.py')
gaussfolder=os.path.join(folder,'mixture_gaussian_cube.py')
shutil.copyfile('unrolled_gan_rhg_cube_LwCL.py',ganfolder)
shutil.copyfile('mixture_gaussian_cube.py',gaussfolder)

with open(batfolder,'w') as f:
    for m in methodlist:

        # for GN in range(1):
        for q in range(1):
            for glr in range(1):
                for dlr in range(1):
                    if m=='prox_rhg' :
                        f.write('python unrolled_gan_rhg_cube_LwCL.py --method {} --us {} --glr {} --dlr {} --GP --num_iter 10000\n'.format(m, 1* (q + 1),0.1**(glr+3),0.1**(dlr+4)))
                    elif  m =='WGAN_rhg':
                        f.write('python unrolled_gan_rhg_cube_LwCL.py --method {} --us {} --glr {} --dlr {} --GP --num_iter 10000\n'.format(m, 1* (q + 1),0.1**(glr+4),0.1**(dlr+4)))
                    else:
                        f.write(
                            'python unrolled_gan_rhg_cube_LwCL.py --method {} --us {} --glr {} --dlr {} --num_iter 10000\n'.format(
                                m, 5 * (q + 1), 0.1 ** (glr + 3), 0.1 ** (dlr + 4)))
os.chdir(folder)
os.system(batfolder)
