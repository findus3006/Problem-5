import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def leapFrog(r,p,R,dt):
    rnorm = np.zeros([np.size(r[0,:])]);
    for i in range(np.size(r[0,:])):
        rnorm[i] = np.linalg.norm(r[:,i]-R)**2;
    r = r + p*dt;
    p = p - rnorm*(r-R)*dt;
    return r,p

# General parameters
N = 300;
dt = 0.001;
t = np.arange(0,50,dt);
global r 
global p

R = np.zeros([3,1]);
r = np.zeros([3,N,len(t)])
p = np.zeros([3,N,len(t)])
r[:,:,0] = np.random.uniform(-0.5,0.5,(3, N)) #0: x,y,z 1:Number of particles, 2: at time t
p[:,:,0] = np.random.uniform(-0.5,0.5,(3, N))
for i in range(len(t)-1):
    if t[i] == 24:
        R = np.zeros([3,1]);
        R[0,0] = 1;
    r[:,:,i+1],p[:,:,i+1] = leapFrog(r[:,:,i],p[:,:,i],R,dt);

#plt.plot(r[0,:,0],r[1,:,0],'.')
#plt.plot(r[0,:,-1],r[1,:,-1],'.')

fig, ax = plt.subplots();

def update(i):
    ax.clear();
    ax.scatter(r[0,:,i],r[1,:,i]);
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 1)
    ax.text(0.7,0.8,f't = {round(t[i],2)}')
    

ani = animation.FuncAnimation(fig, update,frames = np.arange(0, len(t), 100), interval = 33.3)
plt.show()


#f = r"C:\Users\Stefan\Desktop\animation.gif"
#ani = animation.FuncAnimation(fig, update,frames = np.arange(0, len(t), 1), interval = 33.3)
#writergif = animation.PillowWriter(fps=1/dt) 
#ani.save(f, writer=writergif) 