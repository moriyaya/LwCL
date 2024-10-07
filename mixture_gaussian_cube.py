import numpy as np
import matplotlib.pyplot as plt
import seaborn
import ipdb
import scipy.stats
"""
    generate 2d gaussian around a circle
"""
class data_generator(object):
    def __init__(self,dis='normal',n=3,std=0.02,dim=3):

        # n = 8      #8
        radius = 2
        # std = 0.02 #0.02
        delta_theta = 4*np.pi / n

        centers_x = []
        centers_y = []
        centers_z = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    centers_x.append(radius*(i-n/2)*delta_theta)
                    centers_y.append(radius*(j-n/2)*delta_theta)
                    centers_z.append(radius*(k-n/2)*delta_theta)

        centers_x = np.expand_dims(np.array(centers_x), 1)
        centers_y = np.expand_dims(np.array(centers_y), 1)
        centers_z = np.expand_dims(np.array(centers_z), 1)

        p = [1./(n*n*n) for _ in range(n*n*n)]

        self.p = p
        self.size = dim
        self.n = n
        self.dis=dis

        self.centers = np.concatenate([centers_x, centers_y, centers_z], 1)
        self.std = std

    # switch to random distribution (harder)
    def random_distribution(self, p=None):
        if p is None:
            p = [np.random.uniform() for _ in range(self.n)]
            p = p / np.sum(p)
        self.p = p

    # switch to uniform distribution
    def uniform_distribution(self):
        p = [1./(self.n*self.n*self.n) for _ in range(self.n*self.n*self.n)]
        self.p = p

    def sample(self, N,center_require=False):
        n = self.n*self.n*self.n
        std = self.std
        centers = self.centers

        ith_center = np.random.choice(n, N,p=self.p)
        sample_centers = centers[ith_center, :]
        if self.dis=='normal':
            sample_points = np.random.normal(loc=sample_centers, scale=std)
        elif self.dis=='dirichlet':
            alpha=std*np.ones_like(sample_centers)+sample_centers
            sample_points = scipy.stats.dirichlet.rvs(alpha=alpha,size=512)
        if center_require:
            return sample_points.astype('float32'),ith_center
        else:
            return sample_points.astype('float32')


def plot(points):
    plt.scatter(points[:, 0], points[:, 1], c=[0.3 for i in range(1000)], alpha=0.5)
    plt.show()
    plt.close()

def main():
    gen = data_generator()
    sample_points = gen.sample(1000)
    plot(sample_points)

if __name__ == '__main__':
    main()
