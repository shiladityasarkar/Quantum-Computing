from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as pt
import math
from math import pi
from qiskit import Aer, IBMQ, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.tools.visualization import plot_histogram

backend = Aer.get_backend('qasm_simulator')

def get_theta(d):
    x = d[0]
    y = d[1]
    theta = 2*math.acos((x+y)/2.0)
    return theta

def initialize_centers(points,k):
    return points[np.random.randint(points.shape[0],size=k),:]

def find_nearest_neighbour(points,centroids):
    n = len(points)
    k = centroids.shape[0]
    centers = np.zeros(n)
    for i in range(n):
        min_dis = 10000
        ind = 0
        for j in range(k):
            temp_dis = get_Distance(points[i,:],centroids[j,:])
            if temp_dis < min_dis:
                min_dis = temp_dis
                ind = j
        centers[i] = ind
    return centers

def find_centroids(points,centers):
    n = len(points)
    k = int(np.max(centers))+1
    centroids = np.zeros([k,2])
    for i in range(k):
        #print(points[centers==i])
        centroids[i,:] = np.average(points[centers==i])
    return centroids

def preprocess(points):
    n = len(points)
    x = 30.0*np.sqrt(2)
    for i in range(n):
        points[i,:]+=15
        points[i,:]/=x
    return points

def get_Distance(x,y):
    theta_1 = get_theta(x)
    theta_2 = get_theta(y)
    qr = QuantumRegister(3, name="qr")
    cr = ClassicalRegister(3, name="cr")
    qc = QuantumCircuit(qr, cr, name="k_means")
    qc.h(qr[0])
    qc.h(qr[1])
    qc.h(qr[2])
    qc.u(theta_1, pi, pi, qr[1])
    qc.u(theta_2, pi, pi, qr[2])
    qc.cswap(qr[0], qr[1], qr[2])
    qc.h(qr[0])
    qc.measure(qr[0], cr[0])
    qc.reset(qr)
    job = execute(qc,backend=backend, shots=1024)
    result = job.result()
    data = result.get_counts()
    if len(data)==1:
        return 0.0
    else:
        return data['001']/1024.0
    
def generate_data(n,k,std):
    data = make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=std, random_state=100)
    points = data[0]
    centers = data[1]
    return points, centers

def draw_plot(points,centers,label=True):
    if label==False:
        pt.scatter(points[:,0], points[:,1])
    else:
        pt.scatter(points[:,0], points[:,1], c=centers, cmap='viridis')
    pt.xlim(0,1)
    pt.ylim(0,1)
    pt.show()

def plot_centroids(centers):
    pt.scatter(centers[:,0], centers[:,1])
    pt.xlim(0,1)
    pt.ylim(0,1)
    pt.show()


# n = 100
# k = 4
# std = 2

# points,o_centers = get_data(n,k,std)

# points = preprocess(points)
# pt.figure()                   
# draw_plot(points,o_centers,label=False)
# centroids = initialize_centers(points,k)

# for i in range(5):    
#     centers = find_nearest_neighbour(points,centroids)
#     pt.figure()
#     draw_plot(points,centers)
#     #plot_centroids(centroids)
#     centroids = find_centroids(points,centers)
