import numpy as np
import numpy.random as rand


class model:
    def __init__(self, N=1000, fp=0.5, fm=0.5, x=0.5, n=5):
        self.N = N
        self.fp = fp
        self.fm = fm
        self.x = x
        self.n = n
        self.p1 = fm*x**np.linspace(1,n,n)/fp/(1-x)
        self.p0 = fp*x**np.linspace(1,n,n)/fm/(1-x)
        self.p1[n-1]=0
        self.p0[n-1]=0
        self.q = x**np.linspace(0,n-1,n)
        self.q[n-1] = self.q[n-1]/(1-x)
        self.state_0 = np.empty((2,self.N),dtype=int)
        self.pot = np.zeros(self.N)
        
    def init(self):
        self.state_0 = np.array([rand.randint(2, size=self.N),rand.randint(self.n, size=self.N)+1])
        return self.state_0
    
    def memory(self, state_0):
        state = np.empty(state_0.shape, dtype=int)
        state[:] = state_0[:]
        self.pot = rand.randint(2, size=self.N)
        #print(pot)
        for i in range(self.N):
            
            if self.pot[i]==1:
                if state[0,i]==1:
                    
                    if rand.uniform(0,1)<self.p1[state[1,i]-1]:
                        state[1,i] += 1
                else:
                    if rand.uniform(0,1)<self.q[state[1,i]-1]:
                        state[0,i] = 1
                        state[1,i] = 1
                        #print('h')
            else:
                if state[0,i]==1:
                    if rand.uniform(0,1)<self.q[state[1,i]-1]:
                        state[0,i] = 0
                        state[1,i] = 1
                        #print('h')
                else:
                    if rand.uniform(0,1)<self.p0[state[1,i]-1]:
                        state[1,i] += 1
        return state
    
    def update(self, state):
        
        for i in range(self.N):
            if rand.uniform(0,1)<self.fp:
                if state[0,i]==1:
                    if rand.uniform(0,1)<self.p1[state[1,i]-1]:
                        state[1,i] += 1
                else:
                    if rand.uniform(0,1)<self.q[state[1,i]-1]:
                        state[0,i] = 1
                        state[1,i] = 1
            else:
                if state[0,i]==1:
                    if rand.uniform(0,1)<self.q[state[1,i]-1]:
                        state[0,i] = 0
                        state[1,i] = 1
                else:
                    if rand.uniform(0,1)<self.p0[state[1,i]-1]:
                        state[1,i] += 1
        return state
    
    def Sig(self, state):
        S=0
        self.pot = self.pot.astype(bool)
        dep = (1-self.pot).astype(bool)
        
        S+=np.sum(state[0][self.pot])-np.sum(self.state_0[0][self.pot])
        S+=np.sum(1-state[0][dep])-np.sum(1-self.state_0[0][dep])
        return S