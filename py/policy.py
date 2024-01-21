'''
the control policy and related functions
'''
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import linprog #LP
from qpsolvers import solve_qp #QP
import faiss # for similarity search
from glob import glob

def normalize(v):
    '''
    normalize vector v
    '''
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

class Dataset():
    def __init__(self, data, h):
        self.data=data
        self.obsdata=np.array([i[0,-5:] for i in data])
        
        self.index=faiss.IndexFlatL2(5)
        self.index.add(self.obsdata)
        
    def similar_examples(self, query, k):
        '''
        return the k most similar examples from the
        dataset with their similarities scored.
        
        query is a 5D observation vector.

        returns D and trajectories
        D: similarity score, lower is more similar
        '''
        D,I=self.index.search(query,k)
        
        return D, self.data[I].squeeze()

def load_trajectories(h, stride=1):
    '''
    load all recordings and create staggered views of horizon
    length h from the dataset. stride determines window stride.

    returns a Dataset object which can be queried for similar vectors.
    '''
    # load all files
    recordings=[]
    for filename in glob("data/*.npy"):
        recording=np.load(filename)
        recordings.append(recording)
    # split into sliding views
    splits=[sliding_window_view(i, window_shape=(h,7))[::stride] for i in recordings]
    splits=[i.squeeze() for i in splits] #get rid of unused dims
    splits=np.concatenate(splits) #merge all recordings
    return Dataset(splits, h)

def policy_quadratic(current_obs, dataset, k, qpsolver, alt=False):
    '''
    generate output using quadratic approx.

    current_obs: 5D vector
    dataset: dataset object from load_trajectories()
    k: similar example count

    returns hx2 trajectory vector
    '''
    sims, datas=dataset.similar_examples(np.expand_dims(current_obs,0),k)
    #extract the actions from the examples
    trajs=[i[:,:2].flatten() for i in datas]
    # invert similarity score and normalize
    normsims=normalize(1/(sims+np.finfo(float).eps))

    # add a counterexamples to avoid
    diffsims, diffdatas=dataset.similar_examples(np.expand_dims(current_obs,0),100)
    diffdata=diffdatas[-k:]
    difftrajs=[i[:,:2].flatten() for i in diffdata]
    normsims=np.concatenate((normsims, -np.ones((1,len(difftrajs)))*0.1),1)
    trajs+=difftrajs

    if not alt:
        params=fit_quadratic(normsims, trajs, optimmethod="highs", paramsum=1, use_ineqs=True, diagnorm=True)
    else:
        params=alternate_quadratic(normsims,trajs)

    # construct quadratic function
    hh=trajs[0].shape[0] #get dimension (2h)
    P=params[:hh**2].reshape((hh,hh))
    q=params[-hh:]
    print(P,q)
    print(np.linalg.det(P))

    # inequality constraint to prevent discontinuities
    # TODO
    
    lb=np.zeros(hh)

    x=solve_qp(P, q, lb=lb,
               solver=qpsolver, verbose=False)
    
    return x.reshape((hh//2,2))
 

def diag_ineq_vec(dim, i, j):
    '''
    construct vector used for symmetry constraints
    '''
    mat=np.zeros(dim)
    mat[i,j]=1
    mat[j,i]=-1
    return mat.flatten()

def fit_quadratic(similarities, points, paramsum=1, sumall=True, cneg=False, optimmethod="highs", diagnorm=False, use_ineqs=False, debug=False):
    '''
    express quadratic fitting as an LP problem

    similarities: list of similarity scores, higher means more similar
    points: list of (2h)-dim vectors
    '''
    hh=points[0].shape[0] #get dimension (2h)
    k=len(points) #get point count
    # A will have (2h)^2 params, c will have (2h) params.

    # the number of parameters needed to describe the quadratic func is:
    p = hh**2 + hh
    
    # writing out the quadratic eq as a single matrix mul.
    # calculate the "variable" terms of the equation
    # eg. for 2D, [x^2, xy, xy, y^2] and [x, y]
    termvecs=[]
    for point in points:
        # xx'
        # to get all the terms for x'Px
        Pterms=(np.expand_dims(point,0).T*point).flatten()
        # the terms for c'x are just x itself
        # construct vector
        # multiplying this vec with [P.flatten(), q'] should give us the
        # same result as the quadratic eq.
        pointterms=np.concatenate((Pterms,point))
        termvecs.append(pointterms)
        
    # construct c vector for the LP minimization score func.
    c=np.array(termvecs)
    # scale each one by their similarity score and flatten into a vector
    # should be 2hk dimensional.
    c=(similarities.T*c).flatten()
    
    # use the equality constraint to make sure all incidences of the params stay the same
    extra=2 if diagnorm else 1
    A=np.zeros((p*(k-1) + ((hh*(hh-1))//2) + extra, p*k))
    for i in range(k-1):
        # put I matrix in first p cols
        A[p*i:p*(i+1), 0:p]=np.eye(p)
        # put -I matrix in i+1th p cols, ith p rows
        A[p*i:p*(i+1), p*(i+1):p*(i+2)]=-np.eye(p)
    # the matrix params should also be symmetric.
    symconstr=[diag_ineq_vec((hh,hh),i,j) for i in range(hh) for j in range(hh) if i>j]
    A[-((hh*(hh-1))//2)-extra:-extra,:hh**2]=np.array(symconstr)
    
    # use the final equality constraints to make sure params are nonzero
    if sumall:
        A[-extra,:p]=1 # all of the parameters should sum to paramsum
    else:
        A[-extra,:hh**2]=1 # all parameters of A should sum to paramsum
        
    if diagnorm:
        A[-extra+1,:hh**2]=np.eye(hh).flatten() #diagonals of A should sum to paramsum/2.
    
    # these should all be equal to zero
    b=np.zeros(A.shape[0])
    # excepting the last constraints
    b[-extra]=paramsum
    if diagnorm:
        b[-extra+1]=paramsum/2

    if use_ineqs:
        # use inequalities to try to make matrix determinant nonzero
        A_ub=np.zeros((hh-2, p*k))
        for i in range(hh-2):
            A_ub[i,i*hh+i+1:(i+1)*hh]=1*(-1)**i
            A_ub[i,(i+1)*hh+i+2:(i+2)*hh]=-1*(-1)**i
        b_ub=np.zeros(A_ub.shape[0])
    else:
        A_ub=None
        b_ub=None

    if debug:
        return A,b,c

    if cneg: # allow c' to take negative values?
        bnd=[]
        for i in range(hh**2):
            bnd.append((0,None))
        for i in range(hh):
            bnd.append((-600,None))
        bnd=bnd*k
    else:
        bnd=(0.001,None)
    
    # solve the LP problem
    res=linprog(c, A_eq=A, b_eq=b,
                A_ub=A_ub, b_ub=b_ub,
                method=optimmethod,
                options={"disp":True, "presolve":True}, bounds=bnd)

    print(f"Vals: {res.x}\nScore: {res.fun}\nResiduals: {res.con}")
    
    if not res.success:
        print(f"FAILED! {res.message}")
        raise Exception("Optimization failed")

    # return first set of params
    return res.x[0:p]

def alternate_quadratic(similarities, points):
    '''
    an alternate that just places a cvx quadratic on
    the weighted avg of the positive examples
    '''
    hh=points[0].shape[0] #get dimension (2h)
    k=len(points) #get point count
    # A will have (2h)^2 params, c will have (2h) params.

    # the number of parameters needed to describe the quadratic func is:
    p = hh**2 + hh

    pospoints=points[:len(points)//2]
    possimilarities=similarities[0,:similarities.shape[1]//2]
    center=np.average(np.array(pospoints), axis=0, weights=possimilarities)
    A=2*np.ones((hh,hh))+2*np.eye(hh)
    c=-A@center
    params=np.concatenate((A.flatten(), c))
    return params

def test():
    '''
    lines of code that aren't executed during normal runs,
    just here for testing
    '''
    d=load_trajectories(2,2)
    sims, datas=d.similar_examples(np.array([[200,300,300,200,0]]),2)
    sims, datas=d.similar_examples(np.expand_dims(d.obsdata[0],0),2)
    #extract the points from the thingy
    trajs=[i[:,:2].flatten() for i in datas]
    # invert similarity score and normalize
    normsims=normalize(1/(sims+np.finfo(float).eps))

    # add a different one to avoid
    diffsims, diffdatas=d.similar_examples(np.expand_dims(d.obsdata[0],0),100)
    diffdata=diffdatas[-1]
    normsims=np.concatenate((normsims, -np.ones((1,1))),1)
    trajs=np.concatenate((trajs, np.expand_dims(diffdata[:,:2].flatten(),0)))

    #alt
    diffdata=diffdatas[-2:]
    difftrajs=[i[:,:2].flatten() for i in diffdata]
    normsims=np.concatenate((normsims, -np.ones((1,len(difftrajs)))),1)
    trajs+=difftrajs
    
    params=fit_quadratic(normsims, trajs, optimmethod="highs", paramsum=10, debug=False, sumall=True)

    x=policy_quadratic(np.array([200,300,300,200,0]),d,2, "clarabel")
    x
    
