using LinearAlgebra
using RL
using IntervalSets
using StableRNGs
using SparseArrays
using Conda
using Flux
using Random
using PyCall
using FFTW
using PlotlyJS
using FileIO, JLD2
using Statistics
#using Blink

sensors = (8,48)
actuators = 12
dt = 1.5
nx  = 64
ny  = 96


scriptname = "RB_AC_$(dt)_$(sensors[2])"

include(pwd() * "/src/fluid_rk4_NEW.jl")

#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    println(io, "saves/*")
end


py"""
from shenfun import *
import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import symbols, pi, exp
from random import randrange

x, y, tt = sympy.symbols('x,y,t', real=True)

comm = MPI.COMM_WORLD

class KMM:
    def __init__(self,
                 N=(32, 32),
                 domain=((-1, 1), (0, 2*np.pi)),
                 nu=0.01,
                 dt=0.1,
                 conv=0,
                 dpdy=1,
                 filename='KMM',
                 family='C',
                 padding_factor=(1, 1.5),
                 modplot=100,
                 modsave=1e8,
                 moderror=100,
                 checkpoint=1000,
                 timestepper='IMEXRK3'):
        self.N = N
        self.nu = nu
        self.dt = dt
        self.conv = conv
        self.modplot = modplot
        self.modsave = modsave
        self.moderror = moderror
        self.filename = filename
        self.padding_factor = padding_factor
        self.dpdy = dpdy
        self.PDE = PDE = globals().get(timestepper)
        self.im1 = None

        # Regular spaces
        self.B0 = FunctionSpace(N[0], family, bc=(0, 0, 0, 0), domain=domain[0])
        self.D0 = FunctionSpace(N[0], family, bc=(0, 0), domain=domain[0])
        self.C0 = FunctionSpace(N[0], family, domain=domain[0])
        self.F1 = FunctionSpace(N[1], 'F', dtype='d', domain=domain[1])
        self.D00 = FunctionSpace(N[0], family, bc=(0, 0), domain=domain[0])  # Streamwise velocity, not to be in tensorproductspace
        self.C00 = self.D00.get_orthogonal()

        # Regular tensor product spaces
        self.TB = TensorProductSpace(comm, (self.B0, self.F1), modify_spaces_inplace=True) # Wall-normal velocity
        self.TD = TensorProductSpace(comm, (self.D0, self.F1), modify_spaces_inplace=True) # Streamwise velocity
        self.TC = TensorProductSpace(comm, (self.C0, self.F1), modify_spaces_inplace=True) # No bc
        self.BD = VectorSpace([self.TB, self.TD])  # Velocity vector space
        self.CD = VectorSpace(self.TD)             # Dirichlet vector space

        # Padded space for dealiasing
        self.TDp = self.TD.get_dealiased(padding_factor)

        self.u_ = Function(self.BD)      # Velocity vector solution
        self.H_ = Function(self.CD)      # convection
        self.ub = Array(self.BD)

        self.v00 = Function(self.D00)   # For solving 1D problem for Fourier wavenumber 0, 0
        self.w00 = Function(self.D00)

        self.work = CachedArrayDict()
        self.mask = self.TB.get_mask_nyquist() # Used to set the Nyquist frequency to zero
        self.X = self.TD.local_mesh(bcast=True)
        self.K = self.TD.local_wavenumbers(scaled=True)

        # Classes for fast projections. All are not used except if self.conv=0
        self.dudx = Project(Dx(self.u_[0], 0, 1), self.TD)
        if self.conv == 0:
            self.dudy = Project(Dx(self.u_[0], 1, 1), self.TB)
            self.dvdx = Project(Dx(self.u_[1], 0, 1), self.TC)
            self.dvdy = Project(Dx(self.u_[1], 1, 1), self.TD)

        self.curl = Project(curl(self.u_), self.TC)
        self.divu = Project(div(self.u_), self.TC)
        self.solP = None # For computing pressure

        # File for storing the results
        # self.file_u = ShenfunFile('_'.join((filename, 'U')), self.BD, backend='hdf5', mode='w', mesh='uniform')

        # Create a checkpoint file used to restart simulations
        self.checkpoint = Checkpoint(filename,
                                     checkevery=checkpoint,
                                     data={'0': {'U': [self.u_]}})

        # set up equations
        v = TestFunction(self.TB)

        # Chebyshev matrices are not sparse, so need a tailored solver. Legendre has simply 5 nonzero diagonals and can use generic solvers.
        sol1 = chebyshev.la.Biharmonic if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND

        self.pdes = {

            'u': PDE(v,                                   # test function
                     div(grad(self.u_[0])),               # u
                     lambda f: self.nu*div(grad(f)),      # linear operator on u
                     Dx(Dx(self.H_[1], 0, 1), 1, 1)-Dx(self.H_[0], 1, 2),
                     dt=self.dt,
                     solver=sol1,
                     latex=r"\frac{\partial \nabla^2 u}{\partial t} = \nu \nabla^4 u + \frac{\partial^2 N_y}{\partial x \partial y} - \frac{\partial^2 N_x}{\partial y^2}"),
        }

        # v. Momentum equation for Fourier wavenumber 0
        if comm.Get_rank() == 0:
            v0 = TestFunction(self.D00)
            self.h1 = Function(self.D00)  # Copy from H_[1, :, 0, 0] (cannot use view since not contiguous)
            source = Array(self.C00)
            source[:] = -self.dpdy        # dpdy set by subclass
            sol = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.Solver
            self.pdes1d = {
                'v0': PDE(v0,
                          self.v00,
                          lambda f: self.nu*div(grad(f)),
                          [-Expr(self.h1), source],
                          dt=self.dt,
                          solver=sol,
                          latex=r"\frac{\partial v}{\partial t} = \nu \frac{\partial^2 v}{\partial x^2} - N_y - \frac{\partial p}{\partial y}"),
            }

    def convection(self):
        H = self.H_.v
        self.up = self.u_.backward(padding_factor=self.padding_factor)
        up = self.up.v
        if self.conv == 0:
            dudxp = self.dudx().backward(padding_factor=self.padding_factor).v
            dudyp = self.dudy().backward(padding_factor=self.padding_factor).v
            dvdxp = self.dvdx().backward(padding_factor=self.padding_factor).v
            dvdyp = self.dvdy().backward(padding_factor=self.padding_factor).v
            H[0] = self.TDp.forward(up[0]*dudxp+up[1]*dudyp, H[0])
            H[1] = self.TDp.forward(up[0]*dvdxp+up[1]*dvdyp, H[1])

        elif self.conv == 1:
            curl = self.curl().backward(padding_factor=self.padding_factor)
            H[0] = self.TDp.forward(-curl*up[1])
            H[1] = self.TDp.forward(curl*up[0])
        self.H_.mask_nyquist(self.mask)

    def compute_v(self, rk):
        u = self.u_.v
        if comm.Get_rank() == 0:
            self.v00[:] = u[1, :, 0].real
            self.h1[:] = self.H_[1, :, 0].real

        # Find velocity components v from div. constraint
        # print(f"self.dudx() : {self.dudx()} \t self.K[1]   : {self.K[1]}");
        self.K[1][0, 0] = 1

        u[1] = 1j*self.dudx()/self.K[1]

        # Still have to compute for wavenumber = 0, 0
        if comm.Get_rank() == 0:
            # v component
            self.pdes1d['v0'].compute_rhs(rk)
            u[1, :, 0] = self.pdes1d['v0'].solve_step(rk)

        return u

    def compute_pressure(self):
        if self.solP is None:
            self.d2udx2 = Project(self.nu*Dx(self.u_[0], 0, 2), self.TC)
            N0 = self.N0 = FunctionSpace(self.N[0], self.B0.family(), bc={'left': {'N': self.d2udx2()}, 'right': {'N': self.d2udx2()}})
            TN = self.TN = TensorProductSpace(comm, (N0, self.F1), modify_spaces_inplace=True)
            sol = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND
            self.divH = Inner(TestFunction(TN), -div(self.H_))
            self.solP = sol(inner(TestFunction(TN), div(grad(TrialFunction(TN)))))
            self.p_ = Function(TN)

        self.d2udx2()
        self.N0.bc.set_tensor_bcs(self.N0, self.TN)
        p_ = self.solP(self.divH(), self.p_, constraints=((0, 0),))
        return p_

    def print_energy_and_divergence(self, t, tstep):
        if tstep % self.moderror == 0 and self.moderror > 0:
            ub = self.u_.backward(self.ub)
            e0 = inner(1, ub[0]*ub[0])
            e1 = inner(1, ub[1]*ub[1])
            divu = self.divu().backward()
            e3 = np.sqrt(inner(1, divu*divu))
            if comm.Get_rank() == 0:
                print("Time %2.5f Energy %2.6e %2.6e div %2.6e" %(t, e0, e1, e3))

    def init_from_checkpoint(self):
        self.checkpoint.read(self.u_, 'U', step=0)
        self.checkpoint.open()
        tstep = self.checkpoint.f.attrs['tstep']
        t = self.checkpoint.f.attrs['t']
        self.checkpoint.close()
        return t, tstep

    def initialize(self, from_checkpoint=False):
        if from_checkpoint:
            return self.init_from_checkpoint()
        raise RuntimeError('Initialize solver in subclass')

    def plot(self, t, tstep):
        pass

    def update(self, t, tstep):
        self.plot(t, tstep)
        self.print_energy_and_divergence(t, tstep)

    def tofile(self, tstep):
        self.file_u.write(tstep, {'u': [self.u_.backward(mesh='uniform')]}, as_scalar=True)

    def prepare_step(self, rk):
        self.convection()

    def assemble(self):
        for pde in self.pdes.values():
            pde.assemble()
        if comm.Get_rank() == 0:
            for pde in self.pdes1d.values():
                pde.assemble()

    def solve(self, t=0, tstep=0, end_time=1000):
        self.assemble()
        while t < end_time-1e-8:
            for rk in range(self.PDE.steps()):
                self.prepare_step(rk)
                for eq in self.pdes.values():
                    eq.compute_rhs(rk)
                for eq in self.pdes.values():
                    eq.solve_step(rk)
                self.compute_v(rk)
            t += self.dt
            tstep += 1
            self.update(t, tstep)
            self.checkpoint.update(t, tstep)
            if tstep % self.modsave == 0:
                self.tofile(tstep)






class RayleighBenard(KMM):

    def __init__(self,
                    seed = 369,
                    N=(32, 32),
                    domain=((-1, 1), (0, 2*np.pi)),
                    Ra=10000,
                    Pr=0.7,
                    dt=0.1,
                    bcT=(2, 1),
                    conv=0,
                    filename='RB',
                    family='C',
                    padding_factor=(1, 1.5),
                    modplot=100,
                    modsave=10,
                    moderror=100,
                    checkpoint=10,
                    timestepper='IMEXRK3'):
    
        KMM.__init__(self, N=N, domain=domain, nu=np.sqrt(Pr/Ra), dt=dt, conv=conv,
                        filename=filename, family=family, padding_factor=padding_factor,
                        modplot=modplot, modsave=modsave, moderror=moderror,
                        checkpoint=checkpoint, timestepper=timestepper, dpdy=0)
        kappa = self.kappa = 1./np.sqrt(Pr*Ra)
        self.bcT = bcT

        # Additional spaces and functions for Temperature equation
        self.T0 = FunctionSpace(N[0], family, bc=bcT, domain=domain[0])
        self.TT = TensorProductSpace(comm, (self.T0, self.F1), modify_spaces_inplace=True) # Temperature
        self.uT_ = Function(self.BD)     # Velocity vector times T
        self.T_ = Function(self.TT)      # Temperature solution
        self.Tb = Array(self.TT)

        # self.file_T = ShenfunFile('_'.join((filename, 'T')), self.TT, backend='hdf5', mode='w', mesh='uniform')

        # Modify checkpoint file
        self.checkpoint.data['0']['T'] = [self.T_]

        self.dt = dt
        self.domain = domain

        # Chebyshev matrices are not sparse, so need a tailored solver. Legendre has simply 5 nonzero diagonals
        sol2 = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND

        # Addition to u equation.
        self.pdes['u'].N = [self.pdes['u'].N, Dx(self.T_, 1, 2)]
        self.pdes['u'].latex += r'\frac{\partial^2 T}{\partial y^2}'

        # Remove constant pressure gradient from v0 equation
        self.pdes1d['v0'].N = self.pdes1d['v0'].N[0]

        # Add T equations
        q = TestFunction(self.TT)
        self.pdes['T'] = self.PDE(q,
                                    self.T_,
                                    lambda f: kappa*div(grad(f)),
                                    -div(self.uT_),
                                    dt=self.dt,
                                    solver=sol2,
                                    latex=r"\frac{\partial T}{\partial t} = \kappa \nabla^2 T - \nabla \cdot \vec{u}T")

        self.im1 = None
        self.im2 = None
        self._np_random = np.random.default_rng(seed)

    def update_bc(self, t, action = True):
        if action:
            self.T0.bc.bc['left']['D'] = self.bcT[0]
            self.T0.bc.update()
            self.T0.bc.set_tensor_bcs(self.T0, self.T0.tensorproductspace)

            self.T_.get_dealiased_space(self.padding_factor).bases[0].bc.bc['left']['D'] = self.bcT[0]
            self.T_.get_dealiased_space(self.padding_factor).bases[0].bc.update()
            self.T_.get_dealiased_space(self.padding_factor).bases[0].bc.set_tensor_bcs(self.T_.get_dealiased_space(self.padding_factor).bases[0], self.T_.get_dealiased_space(self.padding_factor).bases[0].tensorproductspace)
        else:# Update time-dependent bcs.
            self.T0.bc.update(t)
            self.T_.get_dealiased_space(self.padding_factor).bases[0].bc.update(t)

    def prepare_step(self, rk):
        self.convection()
        Tp = self.T_.backward(padding_factor=self.padding_factor)
        self.uT_ = self.up.function_space().forward(self.up*Tp, self.uT_)

    def tofile(self, tstep):
        self.file_u.write(tstep, {'u': [self.u_.backward(mesh='uniform')]}, as_scalar=True)
        self.file_T.write(tstep, {'T': [self.T_.backward(mesh='uniform')]})

    def init_from_checkpoint(self):
        self.checkpoint.read(self.u_, 'U', step=0)
        self.checkpoint.read(self.T_, 'T', step=0)
        self.checkpoint.open()
        tstep = self.checkpoint.f.attrs['tstep']
        t = self.checkpoint.f.attrs['t']
        self.checkpoint.close()
        return t, tstep

    def print_energy_and_divergence(self, t, tstep):
        # if tstep % self.moderror == 0 and self.moderror > 0:
        ub = self.u_.backward(self.ub)
        Tb = self.T_.backward(self.Tb)
        # Mass Matrices
        e0 = inner(1, ub[0]*ub[0])
        e1 = inner(1, ub[1]*ub[1])
        d0 = inner(1, Tb*Tb)

        divu = self.divu().backward()
        e3 = np.sqrt(inner(1, divu*divu))
        if comm.Get_rank() == 0:
            if tstep % (10*self.moderror) == 0 or tstep == 0:
                print(f"{'Time':^11}{'uu':^11}{'vv':^11}{'T*T':^11}{'div':^11}")
            # print(f"{t:2.4e} {e0:2.4e} {e1:2.4e} {d0:2.4e} {e3:2.4e}")

        return {'t': t, 'uu': e0, 'vv': e1, 'T*T': d0, 'div': e3}

    def initialize(self, rand=0.001, from_checkpoint=False, shift_ic=False):
        if from_checkpoint:
            # if taking from checkpoint go from step 2000 / t= 100
            self.checkpoint.read(self.u_, 'U', step=0)
            self.checkpoint.read(self.T_, 'T', step=0)
            self.checkpoint.open()
            tstep = self.checkpoint.f.attrs['tstep']
            t = self.checkpoint.f.attrs['t']

            self.Tb = self.T_.backward(self.Tb)
            self.ub = self.u_.backward(self.ub)

            if shift_ic:
                # circshift/roll randomly
                temp_rand = randrange(self.N[1])
                self.Tb = np.roll(self.Tb, temp_rand, axis=1)
                self.ub[0,:,:] = np.roll(self.ub[0,:,:], temp_rand, axis=1)
                self.ub[1,:,:] = np.roll(self.ub[1,:,:], temp_rand, axis=1)

            self.T_ = self.Tb.forward(self.T_)
            self.u_ = self.ub.forward(self.u_)
            self.T_.mask_nyquist(self.mask)
            self.u_.mask_nyquist(self.mask)

            self.checkpoint.close()
            self.update_bc(t)
            return t, tstep

        X = self.X

        # funT = 4 if self.bcT[0] == 2 else 2
        funT=4

        # changing according to the boundary conditions
        fun = {1: 1.0,
                2: (0.9+0.1*np.sin(X[1])),
                3: 0.6, 
                4: 2.0}[funT]
        
        # understand this initial state
        self.Tb[:] = 0.5*0.5*(1-X[0] + 0.25*np.sin(np.pi*X[0])) * fun + 1 + rand*self._np_random.standard_normal(size=self.Tb.shape)*(1-X[0])*(1+X[0])
        # self.Tb[:] = 0.25*(1-X[0] + 0.125*np.sin(np.pi*X[0])) * fun + rand*np.random.randn(*self.Tb.shape)*(1-X[0])*(1+X[0])

        self.T_ = self.Tb.forward(self.T_)
        self.T_.mask_nyquist(self.mask)
        return 0, 0

    def init_plots(self):
        self.ub = ub = self.u_.backward()
        Tb = self.T_.backward(self.Tb)
        if comm.Get_rank() == 0:
            plt.figure(1, figsize=(6, 3))
            #self.im1 = plt.quiver(self.X[1][::4, ::4], self.X[0][::4, ::4], ub[1, ::4, ::4], ub[0, ::4, ::4], pivot='mid', scale=0.01)
            self.im1 = plt.quiver(self.X[1][:, :], self.X[0][:, :], ub[1, :, :], ub[0, :, :], pivot='mid', scale=0.01)
            plt.draw()
            plt.figure(2, figsize=(6, 3))
            self.im2 = plt.contourf(self.X[1][:, :], self.X[0][:, :], Tb[:, :], 100)
            plt.draw()
            plt.pause(1e-6)

    def plot(self, t, tstep):
        if self.im1 is None and self.modplot > 0:
            self.init_plots()

        if tstep % self.modplot == 0 and self.modplot > 0:
            ub = self.u_.backward(self.ub)
            self.Tb = self.T_.backward(self.Tb)
            if comm.Get_rank() == 0:
                plt.figure(1)
                #self.im1.set_UVC(ub[1, ::4, ::4], ub[0, ::4, ::4])
                self.im1.set_UVC(ub[1, :, :], ub[0, :, :])
                self.im1.scale = np.linalg.norm(ub[1])
                plt.pause(1e-6)
                plt.figure(2)
                self.im2.axes.clear()
                self.im2.axes.contourf(self.X[1][:, :], self.X[0][:, :], self.Tb[:, :], 100)
                self.im2.autoscale()
                plt.pause(1e-6)
                plt.savefig('test.png')

    def get_Nusselt(self):
        # ub[0] is the y component of the velocity

        H = self.domain[0][1] - self.domain[0][0]
        delta_T = 1.0 # T_top - T_bottom = 1.0 because it remains same throughout
        
        den = self.kappa * delta_T / H

        ub = self.u_.backward(self.ub)
        Tb = self.T_.backward(self.Tb)

        q_1 = np.multiply(ub[0], Tb)
        # print(self.ub)
        q_1_mean = np.mean(q_1) #same as <q_1>_{x,y}

        # gradient of Tb in y direction
        Tx = np.mean(Tb, axis = 1) #average over x
        q_2 = self.kappa * np.mean(np.gradient(Tx)) #average over y
        
        Nu_inst = (q_1_mean - q_2) / den

        # eq 13 [Negative instantaneous Nusselt 
        # number fluctuations in turbulent Rayleigh-Bénard convection]
        # to be included in reward function
        return Nu_inst
        

    def step(self, action = False):
        c = self.pdes['u'].stages()[2]
        self.assemble()
        t, tstep = self.t, self.tstep
        for rk in range(self.PDE.steps()):
            self.prepare_step(rk)
            for eq in ['u', 'T']:
                self.pdes[eq].compute_rhs(rk)
            for eq in ['u']:
                self.pdes[eq].solve_step(rk)
            self.compute_v(rk)
            self.update_bc(t+self.dt*c[rk+1], action) # modify time-dep boundary condition
            self.pdes['T'].solve_step(rk)

        # print(f"t = {t:2.4e}")
            # update time
        self.t += self.dt
        self.tstep += 1
        t, tstep = self.t, self.tstep

        # self.checkpoint.update(t, tstep)
        # print(tstep)
        # if tstep % self.modsave == 0:
        #     print("saved")
            # self.tofile(tstep)

        # Tb = self.T_.backward(self.Tb)
        # ub = self.u_.backward(self.ub)
        return 0, 0


    def solve(self, t=0, tstep=0, end_time=10, apply_action = False):
        self.t, self.tstep = t, tstep

        while self.t < end_time-1e-8:
            self.step(apply_action)

    def destroy(self):
        self.TT.destroy()
        self.TB.destroy()
        self.TD.destroy()
        self.TC.destroy()
        self.TDp.destroy()
        self.BD.destroy()
        self.CD.destroy()






class BCHandler:
    def __init__(self, num_actions = 10, sigma=0.2, threshold = 0.75, y_domain = 2*np.pi):
        self.y = y
        self.L = num_actions
        self.actions = np.ones(num_actions)
        self.x_domain = y_domain
        self.sigma = sigma
        # self.threshold = threshold
        self.eta = threshold / self.L

    def gaussian(self, x):
        # y, yi = self.y, self.yi
        sigma = self.sigma
        gauss = (
            # 1 / sqrt(2 * pi * sigma**2) * 
            exp(-(x**2) / (2 * sigma**2))
            + 1
            # / sqrt(2 * pi * sigma**2)
            * exp(-((x - self.x_domain) ** 2) / (2 * sigma**2))
            + 1
            # / sqrt(2 * pi * sigma**2)
            * exp(-((x + self.x_domain) ** 2) / (2 * sigma**2))
        )
        return gauss

    # def gaussian_circular_pdf(self, x):
    #     sigma = self.sigma
    #     pdf = (
    #         1 / (2 * pi * sigma**2) * exp(-((x - mu) ** 2 / (2 * sigma**2)))
    #         + 1
    #         / (2 * pi * sigma**2)
    #         * exp(-((x - mu - 2 * pi) ** 2 / (2 * sigma**2)))
    #         + 1
    #         / (2 * pi * sigma**2)
    #         * exp(-((x - mu + 2 * pi) ** 2 / (2 * sigma**2)))
    #     )
    #     return pdf

    def get_denormalizing_constant(self, actions):
        sigma = self.sigma
        eta = self.eta
        # integral = sympy.integrate(action_expr, (y, 0, 2*pi)) #domain = (0, 2*pi)
        # c = 2 - np.sum(self.actions) / self.x_domain
        c = 2 - (eta * sigma * np.sum(actions)) / np.sqrt(2 * np.pi)
        return c

    def collate_actions(self, actions=None):
        # to facilitate different tiem in control and simulation
        if actions is None:
            actions = self.actions
        else:
            self.actions = actions

        domain = self.x_domain 
        C = (1*domain - np.sqrt(2*np.pi)*self.sigma * (np.sum(actions)))/(domain)
        action_expr = 1.0 + C
        
        # eta = self.eta
        for i, action in enumerate(zip(actions)):
            # print(i, action[0])
            # act = 
            action_expr = action_expr + action[0] * self.gaussian(
                self.y - i * self.x_domain / self.L
            )

        # c = self.get_denormalizing_constant(actions)
        return action_expr

    def collate_actions_colin(self, actions=None):
        # to facilitate different tiem in control and simulation
        if actions is None:
            actions = self.actions
        else:
            self.actions = actions

        domain = self.x_domain 

        # Amplitude of variation of T
        self.ampl = 0.75  

        # half-length of the interval on which we do the smoothing
        self.dx = 0.03  

        values = self.ampl*actions
        Mean = values.mean()
        K2 = max(1, np.abs(values-np.array([Mean]*self.L)).max()/self.ampl)

        # Position:
        xmax = domain
        ind = sympy.floor(self.L*y//xmax)

        seq=[]
        count = 0
        while count<self.L-1:  # Temperatures will vary between: 2 +- 0.75

            x0 = count*xmax/self.L
            x1 = (count+1)*xmax/self.L

            T1 = 2+(self.ampl*actions[count]-Mean)/K2
            T2 = 2+(self.ampl*actions[count+1]-Mean)/K2
            if count == 0:
                T0 = 2+(self.ampl*actions[self.L-1]-Mean)/K2
            else:
                T0 = 2+(self.ampl*actions[count-1]-Mean)/K2
                
            seq.append((T0+((T0-T1)/(4*self.dx**3))*(y-x0-2*self.dx)*(y-x0+self.dx)**2, y<x0+self.dx))  # cubic smoothing		
            seq.append((T1, y<x1-self.dx))
            seq.append((T1+((T1-T2)/(4*self.dx**3))*(y-x1-2*self.dx)*(y-x1+self.dx)**2, y<x1))  # cubic smoothing

            count += 1

            if count == self.L-1:
                x0 = count*xmax/self.L
                x1 = (count+1)*xmax/self.L
                T0 = 2+(self.ampl*actions[count-1]-Mean)/K2
                T1 = 2+(self.ampl*actions[count]-Mean)/K2
                T2 = 2+(self.ampl*actions[0]-Mean)/K2

                seq.append((T0+((T0-T1)/(4*self.dx**3))*(y-x0-2*self.dx)*(y-x0+self.dx)**2, y<x0+self.dx))
                seq.append((T1, y<x1-self.dx))
                seq.append((T1+((T1-T2)/(4*self.dx**3))*(y-x1-2*self.dx)*(y-x1+self.dx)**2, True))
                
        return sympy.Piecewise(*seq)








from time import time
N = ($nx, $ny)
Ra = 1e4
d = {
    "N": N,
    "dt": 0.01,
    "domain":((-1,1), (0, 2*np.pi)),
    "Ra": Ra,
    "Pr": 0.71,
    "modsave": 100,
    "filename": f"RB_{N[0]}_{N[1]}_bl_Ra={Ra}_local",
    }



c = RayleighBenard(**d)
t, tstep = c.initialize(rand=0.003, from_checkpoint=True)

bc_handler = BCHandler(num_actions=$actuators, 
                                    sigma=0.138, 
                                    y_domain=d["domain"][1][1]
                                    )

"""


# env parameters

seed = Int(floor(rand()*1000))

#seed = 591


gpu_env = false

te = 300.0
# te = 15.0
t0 = 0.0
min_best_episode = 1

check_max_value = "nothing"
max_value = 30.0

Lx  = 2;            
Ly  = 2*pi;
dx  = Lx/nx;        
dy  = Ly/ny;
sim_space = Space(fill(0..1, (nx, ny)))




y0 = zeros(3,nx,ny)

y0[1,:,:] = py"c.T_.backward()"
y0[2,:,:] = py"c.u_.backward()[0,:,:]"
y0[3,:,:] = py"c.u_.backward()[1,:,:]"

y0 = Float32.(y0)



# sensor positions - 
variance = 0.001

sensor_positions = [collect(1:Int(nx/sensors[1]):nx), collect(1:Int(ny/sensors[2]):ny)]

actuator_positions = collect(1:Int(ny/actuators):ny)

actuators_to_sensors = [findfirst(x->x==i, sensor_positions[2]) for i in actuator_positions]


# agent tuning parameters
memory_size = 0
nna_scale = 7.0
nna_scale_critic = 10.0
drop_middle_layer = false
drop_middle_layer_critic = false
fun = leakyrelu
temporal_steps = 1
action_punish = 0#0.002#0.2
delta_action_punish = 0#0.002#0.5
window_size = 23
use_gpu = false
actionspace = Space(fill(-1..1, (1 + memory_size, length(actuator_positions))))
nu = 0.2
agent_power = 7.5

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99997f0
p = 0.9995f0

start_steps = -1
start_policy = ZeroPolicy(actionspace)

update_freq = 40


learning_rate = 1e-4
n_epochs = 4
n_microbatches = 8
logσ_is_network = false
max_σ = 0.2f0
entropy_loss_weight = 0.01


## Grid generation
x1 = range(0,Lx,length=nx+1); 
y1 = range(0,Ly,length=ny+1); 

x1 = x1[1:nx]
y1 = y1[1:ny]

xx,yy = meshgrid(x1,y1);


function prepare_gaussians(;variance = 0.04, norm_mode = 1)
    temp = []

    for (i, position) in enumerate(sensor_positions)
        
        p = real(ifft(taylorvtx(position[1]*dx - dx, position[2]*dy - dy, variance, 1.0)))
        p[p.<0.1] .= 0.0
        if norm_mode == 1
            p ./= sum(p)
        else
            p ./= maximum(p)
        end
        p = sparse(p)

        push!(temp, gpu_env ? CuArray(p) : p)
    end

    return temp    
end

#gaussians = prepare_gaussians(variance = variance)
#gaussians_actuators = prepare_gaussians(variance = variance, norm_mode = 2)
#gaussians_actuators = gaussians_actuators[actuators_to_sensors]


function do_step(env)
    global control = env.p[:]

    # global control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]


    py"""
    act = np.array($control)
    # c.bcT = (bc_handler.collate_actions(act), 1)
    c.bcT = (bc_handler.collate_actions_colin(act), 1)
    """

    py"c.solve(t=0, tstep=tstep, end_time=$dt, apply_action=True)"

    result = zeros(3,nx,ny)

    result[1,:,:] = py"c.T_.backward()"
    result[2,:,:] = py"c.u_.backward()[0,:,:]"
    result[3,:,:] = py"c.u_.backward()[1,:,:]"

    result
end


function array_gradient(a)
    result = zeros(length(a))

    for i in 1:length(a)
        if i == 1
            result[i] = a[i+1] - a[i]
        elseif i == length(a)
            result[i] = a[i] - a[i-1]
        else
            result[i] = (a[i+1] - a[i-1]) / 2
        end
    end

    result
end

function reward_function(env)
    H = py"c.domain[0][1] - c.domain[0][0]"

    delta_T = 1.0

    kappa = py"c.kappa"

    den = kappa * delta_T / H

    q_1_mean = mean(env.y[1,:,:] .* env.y[2,:,:])

    Tx = mean(env.y[1,:,:], dims = 2)

    q_2 = kappa * mean(array_gradient(Tx))

    globalNu = (q_1_mean - q_2) / den

    rewards = zeros(actuators)

    for i in 1:actuators
        tempstate = env.state[:,i]

        tempT = tempstate[1:3:length(tempstate)]
        tempU1 = tempstate[2:3:length(tempstate)]

        q_1_mean = mean(tempT .* tempU1)

        #Tx = mean(env.y[1,:,:], dims = 2)
        Tx = zeros(sensors[1])

        for j in 1:window_size
            Tx .+= tempT[ 8*(j-1)+1 : j*8 ]
        end

        q_2 = kappa * mean(array_gradient(Tx))

        localNu = (q_1_mean - q_2) / den

        # rewards[1,i] = 2.89 - (0.995 * globalNu + 0.005 * localNu)
        rewards[i] = - (globalNu/2.89)^2
    end
 
    return rewards
end



function featurize(y0 = nothing, t0 = nothing; env = nothing)
    if isnothing(env)
        y = y0
    else
        y = env.y
    end

    #convolution is delta
    sensordata = y[:,sensor_positions[1],sensor_positions[2]]

    # New Positional Encoding
    P_Temp = zeros(sensors[1], sensors[2])

    for j in 1:sensors[2]
        i_rad = (2*pi/sensors[2])*j
        P_Temp[:,j] .= sin(i_rad)
    end

    sensordata[1,:,:] += P_Temp

    window_half_size = Int(floor(window_size/2))

    result = Vector{Vector{Float64}}()

    for i in actuators_to_sensors
        temp_indexes = [(i + j + sensors[2] - 1) % sensors[2] + 1 for j in 0-window_half_size:0+window_half_size]

        tempresult = sensordata[:,:,temp_indexes]


        # Positional Encoding

        # P_Temp = zeros(sensors[1], window_size)
        # for j in 1:window_size
        #     i_rad = 2*pi/window_size*j
        #     P_Temp[:,j] .= sin(i_rad)
        # end


        # # NOT SURE
        # #tempresult[1,:,:] += circshift(P_Temp, (0,i))
        # tempresult[1,:,:] += P_Temp

        push!(result, tempresult[:])
    end

    result = reduce(hcat,result)


    if temporal_steps > 1
        if isnothing(env)
            resulttemp = result
            for i in 1:temporal_steps-1
                result = vcat(result, resulttemp)
            end
        else
            result = vcat(result, env.state[1:end-size(result)[1]-memory_size,:])
        end
    end

    if memory_size > 0
        if isnothing(env)
            result = vcat(result, zeros(memory_size, length(actuator_positions)))
        else
            result = vcat(result, env.action[end-(memory_size-1):end,:])
        end
    end

    return Float32.(result)
end

function prepare_action(action0 = nothing, t0 = nothing; env = nothing) 
    if isnothing(env)
        action =  action0
        p = action0
    else
        action = env.action
        p = env.p
    end

    # action = 0.8 * action + 0.2 * p

    return action
end


# PDEenv can also take a custom y0 as a parameter. Example: PDEenv(y0=y0_sawtooth, ...)
function initialize_setup(;use_random_init = false)

    global env = GeneralEnv(do_step = do_step, 
                reward_function = reward_function,
                featurize = featurize,
                prepare_action = prepare_action,
                y0 = y0,
                te = te, t0 = t0, dt = dt, 
                sim_space = sim_space, 
                action_space = actionspace,
                max_value = max_value,
                check_max_value = check_max_value)

    global agent = create_agent_ppo(n_envs = actuators,
                action_space = actionspace,
                state_space = env.state_space,
                use_gpu = use_gpu, 
                rng = rng,
                y = y, p = p,
                start_steps = start_steps, 
                start_policy = start_policy,
                update_freq = update_freq,
                learning_rate = learning_rate,
                nna_scale = nna_scale,
                nna_scale_critic = nna_scale_critic,
                drop_middle_layer = drop_middle_layer,
                drop_middle_layer_critic = drop_middle_layer_critic,
                fun = fun,
                clip1 = true,
                n_epochs = n_epochs,
                n_microbatches = n_microbatches,
                logσ_is_network = logσ_is_network,
                max_σ = max_σ,
                entropy_loss_weight = entropy_loss_weight)

    global hook = GeneralHook(min_best_episode = min_best_episode,
                collect_NNA = false,
                generate_random_init = generate_random_init,
                collect_history = false,
                collect_rewards_all_timesteps = true,
                early_success_possible = false)
end

function generate_random_init()
    py"""
    d = {
        "N": N,
        "dt": 0.05,
        "domain":((-1,1), (0, 2*np.pi)),
        "Ra": Ra,
        "Pr": 0.71,
        "modsave": 100,
        "filename": f"RB_{N[0]}_{N[1]}_bl_Ra={Ra}_local",
        }



    c = RayleighBenard(**d)
    t, tstep = c.initialize(rand=0.003, from_checkpoint=True, shift_ic=False)
    """

    result = zeros(3,nx,ny)

    result[1,:,:] = py"c.T_.backward()"
    result[2,:,:] = py"c.u_.backward()[0,:,:]"
    result[3,:,:] = py"c.u_.backward()[1,:,:]"

    env.y0 = Float32.(result)
    env.y = deepcopy(env.y0)
    env.state = env.featurize(; env = env)

    Float32.(result)
end

initialize_setup()

# plotrun(use_best = false, plot3D = true)

function train(use_random_init = true; visuals = false, num_steps = 1600, inner_loops = 5, outer_loops = 1)
    rm(dirpath * "/training_frames/", recursive=true, force=true)
    mkdir(dirpath * "/training_frames/")
    frame = 1

    if visuals
        colorscale = [[0, "rgb(34, 74, 168)"], [0.25, "rgb(224, 224, 180)"], [0.5, "rgb(156, 33, 11)"], [1, "rgb(226, 63, 161)"], ]
        ymax = 30
        layout = Layout(
                plot_bgcolor="#f1f3f7",
                coloraxis = attr(cmin = 1, cmid = 2.5, cmax = 3, colorscale = colorscale),
            )
    end


    if use_random_init
        hook.generate_random_init = generate_random_init
    else
        hook.generate_random_init = false
    end
    

    for i = 1:outer_loops
        
        for i = 1:inner_loops
            println("")
            
            stop_condition = StopAfterEpisodeWithMinSteps(num_steps)


            # run start
            hook(PRE_EXPERIMENT_STAGE, agent, env)
            agent(PRE_EXPERIMENT_STAGE, env)
            is_stop = false
            while !is_stop
                reset!(env)
                agent(PRE_EPISODE_STAGE, env)
                hook(PRE_EPISODE_STAGE, agent, env)

                while !is_terminated(env) # one episode
                    action = agent(env)

                    agent(PRE_ACT_STAGE, env, action)
                    hook(PRE_ACT_STAGE, agent, env, action)

                    env(action)

                    agent(POST_ACT_STAGE, env)
                    hook(POST_ACT_STAGE, agent, env)

                    if visuals
                        p = plot(heatmap(z=env.y[1,:,:], coloraxis="coloraxis"), layout)

                        savefig(p, dirpath * "/training_frames//a$(lpad(string(frame), 5, '0')).png"; width=1000, height=800)
                    end

                    frame += 1

                    if stop_condition(agent, env)
                        is_stop = true
                        break
                    end
                end # end of an episode

                if is_terminated(env)
                    agent(POST_EPISODE_STAGE, env)  # let the agent see the last observation
                    hook(POST_EPISODE_STAGE, agent, env)
                end
            end
            hook(POST_EXPERIMENT_STAGE, agent, env)
            # run end


            println(hook.bestreward)
            

            # hook.rewards = clamp.(hook.rewards, -3000, 0)
        end
    end

    if visuals && false
        rm(dirpath * "/training.mp4", force=true)
        run(`ffmpeg -framerate 16 -i $(dirpath * "/training_frames/a%05d.png") -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le $(dirpath * "/training.mp4")`)
    end

    save()
end


#train()
#train(;num_steps = 140)
#train(;visuals = true, num_steps = 70)


function load(number = nothing)
    if isnothing(number)
        global hook = FileIO.load(dirpath * "/saves/hook.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env.jld2","env")
    else
        global hook = FileIO.load(dirpath * "/saves/hook$number.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent$number.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env$number.jld2","env")
    end
end

function save(number = nothing)
    isdir(dirpath * "/saves") || mkdir(dirpath * "/saves")

    if isnothing(number)
        FileIO.save(dirpath * "/saves/hook.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env.jld2","env",env)
    else
        FileIO.save(dirpath * "/saves/hook$number.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent$number.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env$number.jld2","env",env)
    end
end



function render_run()

    copyto!(agent.policy.behavior_actor, hook.bestNNA)

    temp_noise = agent.policy.act_noise
    agent.policy.act_noise = 0.0

    temp_start_steps = agent.policy.start_steps
    agent.policy.start_steps  = -1
    
    temp_update_after = agent.policy.update_after
    agent.policy.update_after = 100000

    agent.policy.update_step = 0
    global rewards = Float64[]
    reward_sum = 0.0

    #w = Window()

    rm("frames/", recursive=true, force=true)
    mkdir("frames")

    colorscale = [[0, "rgb(34, 74, 168)"], [0.5, "rgb(224, 224, 180)"], [1, "rgb(156, 33, 11)"], ]
    ymax = 30
    layout = Layout(
            plot_bgcolor="#f1f3f7",
            coloraxis = attr(cmin = 1, cmid = 2.5, cmax = 3, colorscale = colorscale),
        )


    RLBase.reset!(env)
    generate_random_init()

    for i in 1:1000
        action = agent(env)

        env(action)

        result = env.y[1,:,:]

        p = plot(heatmap(z=result, coloraxis="coloraxis"), layout)

        savefig(p, "frames/a$(lpad(string(i), 4, '0')).png"; width=1000, height=800)
        #body!(w,p)

        temp_reward = py"c.get_Nusselt()"
        println(temp_reward)

        reward_sum += temp_reward
        push!(rewards, temp_reward)

        # println(mean(env.reward))

        # reward_sum += mean(env.reward)
        # push!(rewards, mean(env.reward))
    end

    println(reward_sum)

    copyto!(agent.policy.behavior_actor, hook.currentNNA)

    agent.policy.start_steps = temp_start_steps
    agent.policy.act_noise = temp_noise
    agent.policy.update_after = temp_update_after

    if true
        isdir("video_output") || mkdir("video_output")
        rm("video_output/$scriptname.mp4", force=true)
        #run(`ffmpeg -framerate 16 -i "frames/a%04d.png" -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le "video_output/$scriptname.mp4"`)

        run(`ffmpeg -framerate 16 -i "frames/a%04d.png" -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac "video_output/$scriptname.mp4"`)
    end
end

# t1 = scatter(y=rewards1)
# t2 = scatter(y=rewards2)
# t3 = scatter(y=rewards3)
# plot([t1, t2, t3])