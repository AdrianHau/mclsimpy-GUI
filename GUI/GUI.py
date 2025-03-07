# Import utilities
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LightSource
from matplotlib import cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QMovie

# Import vessel and waves
from mclsimpy.simulator import RVG_DP_6DOF
from mclsimpy.waves.wave_loads import WaveLoad
from mclsimpy.waves.wave_spectra import JONSWAP
from mclsimpy.utils import J

import sys
import time


# Matplotlib configuration
plt.rcParams.update({
    "font.family": "Segoe UI",
    "font.size": 10,
    "axes.titlesize": 11,
})

class GUI(QtWidgets.QMainWindow):
    """Defines the graphical user interface using the PyQt6 framework.
    The class loads the graphical user interface .ui file created in the
    Qt Designer software and assigns the necessary back-end code. 
    
    References
    ----------
    pythonguis.com
    PyQt6 documentation
    Qt documentation
    Qt Designer documentation
    """

    def __init__(self):
        super().__init__()
        uic.loadUi('mclsimpy.ui', self)


        # Create Matplotlib figures and embed them into the desired QWidgets
        """Consider moving into a separate function for less clutter,
        initialize_plots(self) for instance. Also find better names,
        "plot1", "plot2" is too general."""
        self.figure1, self.ax1 = plt.subplots(tight_layout=True)
        self.canvas1 = FigureCanvasQTAgg(self.figure1)
        self.toolbar1 = NavigationToolbar2QT(self.canvas1,self)

        self.plot1 = QtWidgets.QVBoxLayout(self.plot_frame_1_ui)
        self.plot1.setContentsMargins(0,0,0,0)
        self.plot1.setSpacing(0)
        self.plot1.addWidget(self.toolbar1)
        self.plot1.addWidget(self.canvas1)

        self.ax1.axis('equal')
        self.ax1.grid()
        self.ax1.set_xlabel('x [m]')
        self.ax1.set_ylabel('y [m]')
        self.ax1.set_title('XY plot')
        
        self.figure2, self.ax2 = plt.subplots(nrows=2, ncols=1, tight_layout=True)
        self.canvas2 = FigureCanvasQTAgg(self.figure2)
        self.toolbar2 = NavigationToolbar2QT(self.canvas2,self)

        self.plot2 = QtWidgets.QVBoxLayout(self.plot_frame_2_ui)
        self.plot2.setContentsMargins(0,0,0,0)
        self.plot2.setSpacing(0)
        self.plot2.addWidget(self.toolbar2)
        self.plot2.addWidget(self.canvas2)

        self.ax2[0].grid()
        self.ax2[0].set_xlabel('Time [s]')
        self.ax2[0].set_ylabel('Position [m]')
        self.ax2[0].set_title('Position')

        self.ax2[1].grid()
        self.ax2[1].set_xlabel('Time [s]')
        self.ax2[1].set_ylabel('Angle [rad]')
        self.ax2[1].set_title('Angle')

        #self.wave_anim_fig, self.wave_anim_ax = plt.subplots(figsize=(6.51,4.51),subplot_kw={"projection": "3d"})

        # Connect simulation run button to the related method
        self.run_button.clicked.connect(self.run_button_clicked)

        self.show()
    
    def run_button_clicked(self):
        """Executes when the run button is clicked.
        Runs the simulation with the wave, vessel, environment, and simulation
        variables given by the user. The variables are stored in this class,
        with variable names defined in Qt Designer, and can be referenced/used
        through these names. As an example, self.dt.text() fetches the time step
        value given by the user in the user interface. Be wary of types - in this case
        the time step dt is given as a string, and must be cast to an int or float.
        
        During simulation runtime the variables are copied and stored temporarily
        inside this function. This is to prevent a change by the user affecting the
        simulation during runtime.
        
        Todo: better errorhandling for user input
        """

        try:
            # Simulation variables setup
            dt = float(self.dt_ui.text())
            simtime = float(self.simtime_ui.text())
            t = np.arange(0,simtime,dt)

            # Vessel setup
            sim_method = None
            if self.sim_method_ui.currentText() == "Runge-Kutta 4":
                sim_method = "RK4"
            elif self.sim_method_ui.currentText() == "Euler":
                sim_method = "Euler"

            vessel = None
            if self.vessel_ui.currentText() == "RV Gunnerus 6DOF":
                vessel = RVG_DP_6DOF(dt,method=sim_method)

            # Current setup
            U_c = float(self.U_c_ui.text())
            beta_c = float(self.beta_c_ui.text())

            # Ocean wave setup
            hs = float(self.Hs_ui.text())
            tp = float(self.Tp_ui.text())
            wp = 2*np.pi/hs
            gamma = float(self.gamma_ui.text())

            N = 100 # Number of wave components
            wmin = 0.5*wp
            wmax = 3.0*wp
            w = np.linspace(wmin, wmax, N)
            k = w**2/9.81

            jonswap = JONSWAP(w)
            wave_spectrum = None
            if self.wave_spectrum_ui.currentText() == "JONSWAP":
                _, wave_spectrum = jonswap(hs=hs,tp=tp,gamma=gamma)

            # Wave-vessel loads setup
            qtf_method = None
            if self.qtf_method_ui.currentText() == "Geometric mean":
                qtf_method = "geo-mean"
            elif self.qtf_method_ui.currentText() == "Newman":
                qtf_method = "Newman"
            
            dw = (wmax-wmin)/N
            wave_amps = np.sqrt(2*wave_spectrum*dw)
            rand_phase = np.random.uniform(0, 2*np.pi, size=N)
            wave_angles = np.ones(N)*np.pi/4

            waveload = WaveLoad(
                wave_amps,
                freqs=w,
                eps=rand_phase,
                angles=wave_angles,
                config_file=vessel._config_file,
                interpolate=True,
                qtf_method=qtf_method,
                deep_water=True,
            )
            
        except ValueError:
            self.output_display.append("Invalid input. Please check your inputs.") # Should add more specific warnings for specific cases
            return

        # Initialize arrays for eta and nu (for plotting)
        eta = np.zeros((6, len(t)))
        nu = np.zeros((6, len(t)))

        # Set control input
        tau_control = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    
        # Assign initial position and pose if necessary
        eta_init = np.array([0, 0, 0, 0, 0, 0])
        nu_init = np.zeros(6)

        vessel.set_eta(eta_init)
        vessel.set_nu(nu_init)

        animate_wave_field = self.animation_checkbox_ui.isChecked()
        if animate_wave_field == True:
            """Clean up and find a better structure"""
            Nx = 100    # Number of discrete x locations
            Ny = 100    # Number of discrete y locations


            #temporary
            Lpp = 28.9
            B = 9.9
            T = 2.7
            L = Lpp/2
            H = T
            mesh_lim = 30
            x = np.linspace(-mesh_lim, mesh_lim, Nx)
            y = np.linspace(-mesh_lim, mesh_lim, Ny)

            X, Y = np.meshgrid(x, y)

            wave_time_series = np.zeros((len(t), Nx, Ny))

            vessel_points = self.initialize_vessel_visualization()
            vessel_time_series = np.zeros((len(t), vessel_points.shape[0], vessel_points.shape[1]))
            hps = np.ix_([0, 1, 2], [0, 1, 2])

        # Run the simulation
        self.output_display.append("Simulating with the given parameters.")
        QtWidgets.QApplication.processEvents()

        for i in range(len(t)):
            # Calculate loads and integrate vessel
            tau_wave = waveload(t[i], vessel.get_eta())
            tau = tau_control + tau_wave
            eta[:, i] = vessel.get_eta()
            nu[:, i] = vessel.get_nu()
            vessel.integrate(U_c, beta_c, tau)

            # Generate time series for animation if requested
            if animate_wave_field == True:
                wave_time_series[i, :, :] = self.calculate_wave_elevation(t[i], N, X, Y, w, k, wave_angles, wave_amps, rand_phase)
                vessel_time_series[i, :, :] = np.array([eta[:3, i] + (J(eta[:, i])[hps])@point for point in vessel_points])

        self.output_display.append("Simulation successful.")
        QtWidgets.QApplication.processEvents() # Updates the GUI manually (otherwise it updates after function is finished running)

        # Plot results
        self.plot_vessel_pose(eta,t)
        self.plot_vessel_XY_position(eta,t)
        QtWidgets.QApplication.processEvents()

        if animate_wave_field == True:
            self.animate_waves(wave_time_series, X, Y, vessel_time_series, eta, dt, hs)

    def plot_vessel_pose(self,eta,t):
        """Plots vessel (x,y,z) position and yaw angle."""
        # Clear previous plot (self.ax2 technically contains two axis objects, hence the loop)
        for ax in self.ax2:
            ax.clear()

        # Plot xyz
        self.ax2[0].plot(t, eta[0, :], label='x')
        self.ax2[0].plot(t, eta[1, :], label='y')
        self.ax2[0].plot(t, eta[2, :], label='z')
        self.ax2[0].legend()
        self.ax2[0].grid()
        self.ax2[0].set_xlabel('Time [s]')
        self.ax2[0].set_ylabel('Position [m]')
        self.ax2[0].set_title('Position')
        
        # Plot yaw angle
        self.ax2[1].plot(t, eta[5, :], label=r"$\psi$")
        self.ax2[1].legend()
        self.ax2[1].grid()
        self.ax2[1].set_xlabel('Time [s]')
        self.ax2[1].set_ylabel('Angle [rad]')
        self.ax2[1].set_title('Angle')

        self.canvas2.draw()

    def plot_vessel_XY_position(self,eta,t):
        """Plots vessel XY position, with arrows indicating vessel orientation."""
        # Clear previous plot
        self.ax1.clear()

        # Plot XY position
        self.ax1.plot(eta[0, :],eta[1, :])

        # Plot orientation of the vessel described by arrows for every 30th time step
        for i in range(0, len(t), 400):
            self.ax1.arrow(eta[0, i], eta[1, i], 0.1* np.cos(eta[5, i]), 0.1*np.sin(eta[5, i]), head_width=0.05, head_length=0.05)

        # Configure the plot
        self.ax1.axis('equal')
        self.ax1.grid()
        self.ax1.set_xlabel('x [m]')
        self.ax1.set_ylabel('y [m]')
        xlim = max(eta[0,:])*1.1
        ylim = max(eta[1,:])*1.1
        self.ax1.set_xlim(-xlim,xlim)
        self.ax1.set_ylim(-ylim,ylim)
        self.ax1.set_title('XY plot')
        
        self.canvas1.draw()

    def animate_waves(self, wave_time_series, X, Y, vessel_time_series, eta, dt, hs):
        """
        To do: Some dt values cause the animation to play at incorrect speeds due to fps issues (FPS needs to be integer).
        At low dt values the fps value becomes very high, causing the animation to lag? Find a fix for both of these problems.
        """

        wave_anim_fig, wave_anim_ax = plt.subplots(figsize=(6.51,4.51),subplot_kw={"projection": "3d"})

        # Set plot labels and axis limits
        wave_anim_fig.suptitle("Wave Elevation & Vessel Motion")
        wave_anim_ax.set_xlabel("X (m)")
        wave_anim_ax.set_ylabel("Y (m)")
        wave_anim_ax.set_zlabel("Elevation (m)")

        wave_animation = FuncAnimation(
            wave_anim_fig,
            self.update_wave_animation,
            fargs=(wave_time_series, X, Y, vessel_time_series, eta, hs, wave_anim_fig, wave_anim_ax, dt),
            frames=wave_time_series.shape[0],
            interval=1, # Only relevant if integrating directly into GUI, not when generating gif
            repeat=False,
            blit=False
        )
        
        fps = 1/dt

        output_gif = "wave_animation.gif"
        wave_animation.save("wave_animation.gif", writer=PillowWriter(fps=fps), dpi=200)

        print(f"FPS:", fps)
        self.movie = QMovie(output_gif)
        self.movie.setScaledSize(self.gif_label_ui.size())

        self.gif_label_ui.setMovie(self.movie)
        self.movie.start()

        self.output_display.append("Animation generated.")
        QtWidgets.QApplication.processEvents()
    
    def update_wave_animation(self, frame, wave_time_series, X, Y, vessel_time_series, eta, hs, wave_anim_fig, wave_anim_ax, dt):
        # Clear the previous surface
        wave_anim_ax.clear()

        anim_axis_lim = 30
        wave_anim_ax.set_xlim(-anim_axis_lim,anim_axis_lim)
        wave_anim_ax.set_ylim(-anim_axis_lim,anim_axis_lim)
        wave_anim_ax.set_zlim(-anim_axis_lim,anim_axis_lim)

        # Display current time
        wave_anim_ax.set_title(f"t={round(frame*dt,1)}")

        # Plot wave surface mesh at current frame
        #ls = LightSource(azdeg=0, altdeg=60)
        #rgb = ls.shade(wave_time_series[frame], cmap=cm.Blues_r)
        wave_anim_ax.plot_surface(X, Y, wave_time_series[frame],
            cmap=cm.Blues_r, edgecolor="none", rstride=1, cstride=1, zorder=2,
            vmin= -hs*0.5, vmax = hs*1.5
        )
        #facecolors=rgb

        # Plot vessel pose and position at current frame
        wave_anim_ax.plot(vessel_time_series[frame][:, 0],
                               vessel_time_series[frame][:, 1],
                               vessel_time_series[frame][:, 2],
                               color='black', linestyle='-', linewidth=1,
                               zorder=10,alpha=1
        )
        wave_anim_ax.plot(eta[0, frame], eta[1, frame], eta[2, frame], 'o',
                               linewidth=1, zorder=10, color='black', alpha=0.5
        )

        # Print progress to user
        if frame % 10 == 0: 
            progress = frame/len(wave_time_series)*100
            self.output_display.append(f"Generating animation - {progress}%")
            QtWidgets.QApplication.processEvents()
            print(frame)

        return wave_anim_ax,

    def calculate_wave_elevation(self,t, N, X, Y, w, k, wave_angles, wave_amps, rand_phase):
        wave_elevation = 0
        for i in range(N):
            wave_elevation += wave_amps[i]*np.cos(w[i]*t-k[i]*X*np.cos(wave_angles[i])-k[i]*Y*np.sin(wave_angles[i]+rand_phase[i]))
        return wave_elevation
    
    def initialize_vessel_visualization(self):
        """Function to create simplified visualization of the vessel
        based on its main geometric characteristics.
        Function needs to be generalized.
        """

        Lpp = 28.9
        B = 9.9
        T = 2.7
        L = Lpp/2
        H = T
        scale = 3/7
        vessel_points = np.array([
            # Four points in the back
            [-L, -B, -H],
            [-L, -B, H],
            [-L, B, H],
            [-L, B, -H],
            # Four points at 2/3 of lenght
            [L*scale, B, -H],
            [L*scale, B, H],
            [L*scale, -B, H],
            [L*scale, -B, -H],
            # Front of vessel
            [L, 0, -H],
            [L, 0, H],
            # Now go back to complete the figure
            [L*scale, B, H],
            [L*scale, B, -H],
            [L, 0, -H],
            [L*scale, -B, -H],
            [-L, -B, -H],
            [-L, -B, H],
            [L*scale, -B, H],
            [L, 0, H],
            [L*scale, B, H],
            [-L, B, H],

            
            # Add some lines that are missing to complete the shape
            [-L, B, -H],
            [-L, -B, -H],
            [L*scale, -B, -H],
            [L*scale, B, -H]
        ])
        vessel_points[:, 2] = -vessel_points[:, 2]

        return vessel_points

app = QtWidgets.QApplication(sys.argv)
window = GUI()
sys.exit(app.exec())

