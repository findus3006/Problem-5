import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def leapFrog(r,p,R,dt,coulomb):
    rnorm = np.zeros([np.size(r[0,:])]);
    for i in range(np.size(r[0,:])):
        rnorm[i] = np.linalg.norm(r[:,i]-R)**2;
    r = r + p*dt;
    if coulomb == 1:
        p = p + (-rnorm*(r-R)+F(r))*dt;
    else:
        p = p - rnorm*(r-R)*dt
    return r,p

def F(r):
    alpha = 1/np.size(r[0,:]);
    F = np.zeros(r.shape);
    for j in range(np.size(F[0,:])):
        for i in range(np.size(F[0,:])):
            if i != j:
                F = alpha*(r[:,j]-r[:,i])/(np.linalg.norm(r[:,j]-r[:,i])**3)
    return F

# General parameters
N = 30;
dt = 0.01;

t = np.arange(0,50,dt);
R = np.zeros([3,1]);
rcenter = np.zeros([3,len(t)])
r = np.zeros([3,N,len(t)])
p = np.zeros([3,N,len(t)])
r[:,:,0] = np.random.uniform(-0.5,0.5,(3, N)) #0: x,y,z 1:Number of particles, 2: at time t
p[:,:,0] = np.random.uniform(-0.5,0.5,(3, N))

for i in range(len(t)-1):
    if t[i] == 25:
        R = np.zeros([3,1]);
        R[0,0] = 1;
    r[:,:,i+1],p[:,:,i+1] = leapFrog(r[:,:,i],p[:,:,i],R,dt,0);
    print(round(t[i],2))
    rcenter[:,i] = np.transpose(np.sum(r[:,:,i],1)/N);

# fig = plt.figure();
# ax = fig.add_subplot(projection='3d')
# Just pass the z corodiante to scatter as well and adjust the limits accordingly

fig = plt.figure();
ax = fig.add_subplot()

def update(i):
    ax.clear();
    ax.scatter(r[0,:,i],r[1,:,i]);
    ax.set_xlim(-2, 4)
    ax.set_ylim(-3, 3)
    ax.text(2,2,f't = {round(t[i],2)}')
    ax.set_aspect('equal')
    ax.plot(rcenter[0,i],rcenter[1,i],'r+',markersize=10)

ani = animation.FuncAnimation(fig, update,frames = np.arange(0, len(t), 10), interval = 33.3)
plt.show()

#f = r"C:\Users\Stefan\Desktop\Uni\Master\2.Semester\Computational Physics\Python_scripts\Problem-5\animation.gif"
#writergif = animation.PillowWriter(fps=1/dt) 
#ani.save(f, writer=writergif) 