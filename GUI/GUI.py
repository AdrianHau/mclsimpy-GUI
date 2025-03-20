"""
GUI.py

Graphical user interface for the ocean wave and vessel simulator mclsimpy.
Based on the PyQt6 framework, designed using the Qt Designer application.

Author: Adrian Haugland
Date: 2025-02-25
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LightSource
from matplotlib import cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QMovie
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
import json

from mclsimpy.simulator import RVG_DP_6DOF, GunnerusManeuvering3DoF, CSAD_DP_6DOF, CSAD_DP_Seakeeping
from mclsimpy.waves.wave_loads import WaveLoad
from mclsimpy.waves.wave_spectra import JONSWAP, ModifiedPiersonMoskowitz, BasePMSpectrum
from mclsimpy.utils import J

import sys

# Matplotlib configuration
plt.rcParams.update(
    {
        "font.family": "Segoe UI",
        "font.size": 10,
        "axes.titlesize": 11,
    }
)


class GUI(QtWidgets.QMainWindow):
    """Defines the graphical user interface.
    The class loads the graphical user interface .ui file created in the
    Qt Designer software and assigns the necessary back-end functionalities.

    References
    ----------
    pythonguis.com
    PyQt6 documentation
    Qt documentation
    Qt Designer documentation
    """

    def __init__(self):
        super().__init__()

        # Load ui file generated in Qt Designer
        uic.loadUi("mclsimpy.ui", self)

        # Create Matplotlib figures and embed them into the desired QWidgets
        self.initialize_plots()

        # Connect simulation run button to the related method
        self.run_button_ui.clicked.connect(self.run_button_clicked)

        self.simulation_thread = None

        self.show()

    def run_button_clicked(self):
        """Executes when the GUI button is clicked.
        Runs the simulation with the wave, vessel, environment, and simulation
        variables given by the user. Utilizes threading through the PyQt framework
        in order to prevent the simulation and animation process from freezing the
        GUI window.

        To do:
        - Better errorhandling for user input?
        """

        # Prevent running multiple simulation threads at once
        if self.simulation_thread is not None:
            self.simulation_thread.stop()
            self.simulation_thread.wait()
            self.simulation_thread.deleteLater()
            self.simulation_thread = None
            return

        # Fetch data given by user
        try:

            # Simulation variables
            dt = float(self.dt_ui.text())
            simtime = float(self.simtime_ui.text())

            sim_method = None
            if self.sim_method_ui.currentText() == "Runge-Kutta 4":
                sim_method = "RK4"
            elif self.sim_method_ui.currentText() == "Euler":
                sim_method = "Euler"

            # Vessel selection
            vessel = None
            if self.vessel_ui.currentText() == "RV Gunnerus 6DOF":
                vessel = RVG_DP_6DOF(dt, method=sim_method)
            elif self.vessel_ui.currentText() == "RV Gunnerus 3DOF":
                # vessel = GunnerusManeuvering3DoF(dt, method=sim_method)
                self.output_display.append("Error in vessel GUI implementation - to be fixed.")
                # Need to figure out config file issue
                return
            elif self.vessel_ui.currentText() == "CSAD 6DOF":
                vessel = CSAD_DP_6DOF(dt, method=sim_method)
            elif self.vessel_ui.currentText() == "CSAD 3DOF":
                vessel = CSAD_DP_Seakeeping(dt, method=sim_method)

            # Environmental load variables
            U_c = float(self.U_c_ui.text())
            beta_c = float(self.beta_c_ui.text())

            # Wave variables and setup
            hs = float(self.Hs_ui.text())
            tp = float(self.Tp_ui.text())
            N = int(self.N_ui.text())  # Number of wave components
            wp = 2 * np.pi / hs
            wmin = 0.5 * wp
            wmax = 3.0 * wp
            w = np.linspace(wmin, wmax, N)

            wave_spectrum = None
            if self.wave_spectrum_ui.currentText() == "JONSWAP":
                gamma = float(self.gamma_ui.text())
                jonswap = JONSWAP(w)
                _, wave_spectrum = jonswap(hs=hs, tp=tp, gamma=gamma)
            elif self.wave_spectrum_ui.currentText() == "Modified Pierson–Moskowitz":
                modified_pm_spectrum = ModifiedPiersonMoskowitz(w)
                _, wave_spectrum = modified_pm_spectrum(hs=hs, tp=tp)
            elif self.wave_spectrum_ui.currentText() == "Pierson–Moskowitz":
                # A = self.PM_A_ui.text()
                # B = self.PM_B_ui.text()
                # pm_spectrum = BasePMSpectrum(w)
                # _, wave_spectrum = pm_spectrum(A=A, B=B)
                # TypeError, need to fix.
                self.output_display.append("Error in wave spectrum GUI implementation - to be fixed.")
                return

            # Wave loads method
            qtf_method = None
            if self.qtf_method_ui.currentText() == "Geometric mean":
                qtf_method = "geo-mean"
            elif self.qtf_method_ui.currentText() == "Newman":
                qtf_method = "Newman"

            # Check if user wants 3D animation
            animation_check = self.animation_checkbox_ui.isChecked()

        except ValueError:
            self.print_to_user("Invalid input. Please check your inputs.")
            return

        # Initialize the simulation thread
        self.simulation_thread = SimulationThread(
            dt=dt,
            simtime=simtime,
            sim_method=sim_method,
            vessel=vessel,
            U_c=U_c,
            beta_c=beta_c,
            hs=hs,
            tp=tp,
            N=N,
            wave_spectrum=wave_spectrum,
            qtf_method=qtf_method,
            animation_check=animation_check,
        )

        # Connect to simulation thread to receive simulation updates and results
        self.simulation_thread.message_signal.connect(self.print_to_user)
        self.simulation_thread.sim_results_signal.connect(self.plot_results)
        self.simulation_thread.animation_finished_signal.connect(self.display_animation)

        # Start the simulation
        self.simulation_thread.start()

    def initialize_plots(self):
        """Initializes figures for plotting the simulation results.

        To do:
        - Consider finding better variable names (plot 1, plot 2, ...
          might be too general)
        """

        self.figure1, self.ax1 = plt.subplots(tight_layout=True)
        self.canvas1 = FigureCanvasQTAgg(self.figure1)
        self.toolbar1 = NavigationToolbar2QT(self.canvas1, self)

        self.plot1 = QtWidgets.QVBoxLayout(self.plot_frame_1_ui)
        self.plot1.setContentsMargins(0, 0, 0, 0)
        self.plot1.setSpacing(0)
        self.plot1.addWidget(self.toolbar1)
        self.plot1.addWidget(self.canvas1)

        self.ax1.axis("equal")
        self.ax1.grid()
        self.ax1.set_xlabel("x [m]")
        self.ax1.set_ylabel("y [m]")
        self.ax1.set_title("XY plot")

        self.figure2, self.ax2 = plt.subplots(nrows=2, ncols=1, tight_layout=True)
        self.canvas2 = FigureCanvasQTAgg(self.figure2)
        self.toolbar2 = NavigationToolbar2QT(self.canvas2, self)

        self.plot2 = QtWidgets.QVBoxLayout(self.plot_frame_2_ui)
        self.plot2.setContentsMargins(0, 0, 0, 0)
        self.plot2.setSpacing(0)
        self.plot2.addWidget(self.toolbar2)
        self.plot2.addWidget(self.canvas2)

        self.ax2[0].grid()
        self.ax2[0].set_xlabel("Time [s]")
        self.ax2[0].set_ylabel("Position [m]")
        self.ax2[0].set_title("Position")

        self.ax2[1].grid()
        self.ax2[1].set_xlabel("Time [s]")
        self.ax2[1].set_ylabel("Angle [rad]")
        self.ax2[1].set_title("Angle")

    @pyqtSlot(str)
    def print_to_user(self, message):
        """Printing function to update the user on simulation status/events.
        Connected to the simulation thread by the pyqtSignal/pyqtSlot functionality, but
        can also be used purely within the GUI class as a general printing function.

        Parameters
        ----------
        message : string
            String to be printed to the user.
        """

        self.output_display.append(message)
        QtWidgets.QApplication.processEvents()

    @pyqtSlot(str)
    def display_animation(self, gif_path):
        self.movie = QMovie(gif_path)
        self.movie.setScaledSize(self.gif_label_ui.size())
        self.gif_label_ui.setMovie(self.movie)
        self.movie.start()
        QtWidgets.QApplication.processEvents()

    @pyqtSlot(object)
    def plot_results(self, results):
        self.plot_vessel_pose(results["eta"], results["t"])
        self.plot_vessel_XY_position(results["eta"], results["t"])

    def plot_vessel_pose(self, eta, t):
        """Plots vessel (x,y,z) position and yaw angle."""
        # Clear previous plot (self.ax2 technically contains two axis objects, hence the loop)
        for ax in self.ax2:
            ax.clear()

        # Plot xyz
        self.ax2[0].plot(t, eta[0, :], label="x")
        self.ax2[0].plot(t, eta[1, :], label="y")
        self.ax2[0].plot(t, eta[2, :], label="z")
        self.ax2[0].legend()
        self.ax2[0].grid()
        self.ax2[0].set_xlabel("Time [s]")
        self.ax2[0].set_ylabel("Position [m]")
        self.ax2[0].set_title("Position")

        # Plot yaw angle
        self.ax2[1].plot(t, eta[5, :], label=r"$\psi$")
        self.ax2[1].legend()
        self.ax2[1].grid()
        self.ax2[1].set_xlabel("Time [s]")
        self.ax2[1].set_ylabel("Angle [rad]")
        self.ax2[1].set_title("Angle")

        self.canvas2.draw()

    def plot_vessel_XY_position(self, eta, t):
        """Plots vessel XY position, with arrows indicating vessel orientation."""
        # Clear previous plot
        self.ax1.clear()

        # Plot XY position
        self.ax1.plot(eta[0, :], eta[1, :])

        # Plot orientation of the vessel described by arrows for every 30th time step
        for i in range(0, len(t), 400):
            self.ax1.arrow(
                eta[0, i],
                eta[1, i],
                0.1 * np.cos(eta[5, i]),
                0.1 * np.sin(eta[5, i]),
                head_width=0.05,
                head_length=0.05,
            )

        # Configure the plot
        self.ax1.axis("equal")
        self.ax1.grid()
        self.ax1.set_xlabel("x [m]")
        self.ax1.set_ylabel("y [m]")
        xlim = max(eta[0, :]) * 1.1
        ylim = max(eta[1, :]) * 1.1
        self.ax1.set_xlim(-xlim, xlim)
        self.ax1.set_ylim(-ylim, ylim)
        self.ax1.set_title("XY plot")

        self.canvas1.draw()


class SimulationThread(QThread):
    """Worker thread for the ocean wave and vessel simulation.
    Prevents the graphical user interface from freezing when the simulation is running
    by doing all calculations and plotting/visualization on a separate thread.
    """

    # Signals for sending updates and results to the GUI class instance
    message_signal = pyqtSignal(str)
    sim_results_signal = pyqtSignal(object)
    animation_finished_signal = pyqtSignal(str)

    def __init__(
        self,
        dt,
        simtime,
        sim_method,
        vessel,
        U_c,
        beta_c,
        hs,
        tp,
        N,
        wave_spectrum,
        qtf_method,
        animation_check,
    ):
        super().__init__()

        self.dt = dt
        self.simtime = simtime
        self.sim_method = sim_method
        self.vessel = vessel
        self.U_c = U_c
        self.beta_c = beta_c
        self.hs = hs
        self.tp = tp
        self.N = N
        self.wave_spectrum = wave_spectrum
        self.qtf_method = qtf_method
        self.animation_check = animation_check

    def run(self):
        results = self.simulate()
        self.sim_results_signal.emit(results)
        if self.animation_check == True:
            self.generate_3D_animation(
                wave_time_series=results["wave_time_series"],
                vessel_time_series=results["vessel_time_series"],
                X=results["X"],
                Y=results["Y"],
                eta=results["eta"],
                dt=results["dt"],
                hs=results["hs"],
            )

    def simulate(self):
        """Performs the simulation and returns the simulation results.

        To do:
        - Clean up in regards to the wave set up. The w array is calculated twice
        for instance (see run_button_clicked()).
        """
        # Wave setup
        wp = 2 * np.pi / self.hs
        wmin = 0.5 * wp
        wmax = 3.0 * wp
        w = np.linspace(wmin, wmax, self.N)
        k = w**2 / 9.81

        # Wave load setup
        dw = (wmax - wmin) / self.N
        wave_amps = np.sqrt(2 * self.wave_spectrum * dw)
        rand_phase = np.random.uniform(0, 2 * np.pi, size=self.N)
        wave_angles = np.ones(self.N) * np.pi / 4

        waveload = WaveLoad(
            wave_amps,
            freqs=w,
            eps=rand_phase,
            angles=wave_angles,
            config_file=self.vessel._config_file,
            interpolate=True,
            qtf_method=self.qtf_method,
            deep_water=True,
        )

        # Initialize arrays for eta and nu (for plotting)
        t = np.arange(0, self.simtime, self.dt)
        eta = np.zeros((6, len(t)))
        nu = np.zeros((6, len(t)))

        # Set control input
        tau_control = np.array([0, 0, 0, 0, 0, 0], dtype=float)

        # Assign initial position and pose if necessary
        eta_init = np.zeros(6)
        nu_init = np.zeros(6)

        self.vessel.set_eta(eta_init)
        self.vessel.set_nu(nu_init)

        if self.animation_check == True:
            """Clean up and find a better structure"""
            Nx = 100  # Number of discrete x locations
            Ny = 100  # Number of discrete y locations

            # temporary
            mesh_lim = 30
            x = np.linspace(-mesh_lim, mesh_lim, Nx)
            y = np.linspace(-mesh_lim, mesh_lim, Ny)

            X, Y = np.meshgrid(x, y)

            wave_time_series = np.zeros((len(t), Nx, Ny))

            vessel_points = self.define_vessel_geometry(self.vessel)
            vessel_time_series = np.zeros((len(t), vessel_points.shape[0], vessel_points.shape[1]))
            hps = np.ix_([0, 1, 2], [0, 1, 2])

        # Run the simulation
        self.message_signal.emit("Simulating with the given parameters.")

        for i in range(len(t)):
            # Calculate loads and integrate vessel
            tau_wave = waveload(t[i], self.vessel.get_eta())
            tau = tau_control + tau_wave
            eta[:, i] = self.vessel.get_eta()
            nu[:, i] = self.vessel.get_nu()
            self.vessel.integrate(self.U_c, self.beta_c, tau)

            # Generate time series for animation if requested
            if self.animation_check == True:
                wave_time_series[i, :, :] = self.calculate_wave_elevation(
                    t[i], self.N, X, Y, w, k, wave_angles, wave_amps, rand_phase
                )
                vessel_time_series[i, :, :] = np.array(
                    [eta[:3, i] + (J(eta[:, i])[hps]) @ point for point in vessel_points]
                )

        self.message_signal.emit("Simulation successful.")

        if self.animation_check == False:
            results = {"t": t, "eta": eta, "nu": nu, "tau_control": tau_control}

        else:
            results = {
                "t": t,
                "wave_time_series": wave_time_series,
                "vessel_time_series": vessel_time_series,
                "eta": eta,
                "nu": nu,
                "tau_control": tau_control,
                "X": X,
                "Y": Y,
                "dt": self.dt,
                "hs": self.hs,
            }

        return results

    def calculate_wave_elevation(self, t, N, X, Y, w, k, wave_angles, wave_amps, rand_phases):
        """Calculate wave elevation at each point of the XY meshgrid at time step t."""
        wave_elevation = 0

        for i in range(N):
            phase = (
                w[i] * t
                - k[i] * X * np.cos(wave_angles[i])
                - k[i] * Y * np.sin(wave_angles[i])
                + rand_phases[i]
            )
            wave_elevation += wave_amps[i] * np.cos(phase)

        return wave_elevation

    def generate_3D_animation(self, wave_time_series, vessel_time_series, X, Y, eta, dt, hs):
        """
        To do: Some dt values cause the animation to play at incorrect speeds due to fps issues (FPS needs to be integer).
        At low dt values the fps value becomes very high, causing the animation to lag? Find a fix for both of these problems.
        """

        wave_anim_fig, wave_anim_ax = plt.subplots(figsize=(6.51, 4.51), subplot_kw={"projection": "3d"})

        # Set plot labels and axis limits
        wave_anim_fig.suptitle("Wave Elevation & Vessel Motion")
        wave_anim_ax.set_xlabel("X (m)")
        wave_anim_ax.set_ylabel("Y (m)")
        wave_anim_ax.set_zlabel("Elevation (m)")

        # Generate animation
        self.message_signal.emit("Generating animation.")
        wave_animation = FuncAnimation(
            wave_anim_fig,
            self.update_animation,
            fargs=(wave_time_series, vessel_time_series, X, Y, eta, hs, wave_anim_ax, dt),
            frames=wave_time_series.shape[0],
            interval=1, # Only relevant if integrating directly into GUI, not when generating gif
            repeat=False,
            blit=False,
        )  

        # Save to gif file
        fps = 1 / dt
        wave_animation.save("3D_animation.gif", writer=PillowWriter(fps=fps), dpi=200)
        self.animation_finished_signal.emit("3D_animation.gif")
        self.message_signal.emit("Animation generated.")

    def update_animation(self, frame, wave_time_series, vessel_time_series, X, Y, eta, hs, wave_anim_ax, dt):
        # Clear the previous surface
        wave_anim_ax.clear()

        anim_axis_lim = 30
        wave_anim_ax.set_xlim(-anim_axis_lim, anim_axis_lim)
        wave_anim_ax.set_ylim(-anim_axis_lim, anim_axis_lim)
        wave_anim_ax.set_zlim(-anim_axis_lim, anim_axis_lim)

        # Display current time
        wave_anim_ax.set_title(f"t={round(frame*dt,1)}")

        # Plot wave surface mesh at current frame
        # ls = LightSource(azdeg=0, altdeg=60)
        # rgb = ls.shade(wave_time_series[frame], cmap=cm.Blues_r)
        wave_anim_ax.plot_surface(
            X,
            Y,
            wave_time_series[frame],
            cmap=cm.Blues_r,
            edgecolor="none",
            rstride=1,
            cstride=1,
            zorder=2,
            vmin=-hs * 0.5,
            vmax=hs * 1.5,
        )
        # facecolors=rgb

        # Plot vessel pose and position at current frame
        wave_anim_ax.plot(
            vessel_time_series[frame][:, 0],
            vessel_time_series[frame][:, 1],
            vessel_time_series[frame][:, 2],
            color="black",
            linestyle="-",
            linewidth=1,
            zorder=10,
            alpha=1,
        )
        wave_anim_ax.plot(
            eta[0, frame],
            eta[1, frame],
            eta[2, frame],
            "o",
            linewidth=0.5,
            zorder=10,
            color="black",
            alpha=0.5,
        )

        # Print progress to user
        if frame % 10 == 0:
            progress = frame / len(wave_time_series) * 100
            self.message_signal.emit(f"Generating animation - {round(progress,1)}%")

            print("Frame:", frame)

        return (wave_anim_ax,)

    def define_vessel_geometry(self, vessel):
        """Function to initialize a simplified visualization of the vessel based on its main dimensions."""

        with open(vessel._config_file, "r") as f:
            data = json.load(f)["main"]

        Lpp = data["Lpp"]
        B = data["B"]
        T = data["T"]
        L = Lpp / 2
        H = T
        scale = 3 / 7
        vessel_points = np.array(
            [
                # Four points in the back
                [-L, -B, -H],
                [-L, -B, H],
                [-L, B, H],
                [-L, B, -H],
                # Four points at 2/3 of lenght
                [L * scale, B, -H],
                [L * scale, B, H],
                [L * scale, -B, H],
                [L * scale, -B, -H],
                # Front of vessel
                [L, 0, -H],
                [L, 0, H],
                # Now go back to complete the figure
                [L * scale, B, H],
                [L * scale, B, -H],
                [L, 0, -H],
                [L * scale, -B, -H],
                [-L, -B, -H],
                [-L, -B, H],
                [L * scale, -B, H],
                [L, 0, H],
                [L * scale, B, H],
                [-L, B, H],
                [-L, B, -H],
                [-L, -B, -H],
                [L * scale, -B, -H],
                [L * scale, B, -H],
            ]
        )
        vessel_points[:, 2] = -vessel_points[:, 2]

        return vessel_points

    def stop(self):
        self._is_running = False


app = QtWidgets.QApplication(sys.argv)
window = GUI()
sys.exit(app.exec())
