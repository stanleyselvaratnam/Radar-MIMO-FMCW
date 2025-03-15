# support for matplotlib animation inside tk window
# yves.piguet@csem.ch

import tkinter as tk
import matplotlib.backends.backend_tkagg as backend
import matplotlib.figure as figure

class PlotWin(tk.Tk):
    """
    Simple tkinter application with a (live) matplotlib figure and a status bar.
    """

    DEFAULT_PACKING_ARGS = {"padx": 10, "pady": 10}
    DEFAULT_RADIO_PACKING_ARGS = {"padx": 10, "pady": 3}

    def __init__(self, title="Plot",
                 figure_update=None,
                 status_update=None,
                 init_config_panel=None):
        """
        PlotWin constructor.

        Parameters:
          title: window title
          figure_update: tuple with 3 elements: function with arguments (figure, *args),
          args, and time to wait for next execution. Arguments "figure" is the matplotlib
          figure and can be used e.g. to create subplots with figure.add_subplot(2,2,1).
          status_update: tuple with 3 elements, like figure_update but the function
          doesn't have a figure argument and must return a string to be display in the
          status bar at the bottom of the window.
        """
        
        super().__init__()
        self.figure_update = figure_update
        self.status_update = status_update
        self.title(title)

        # status
        if status_update is not None:
            self.status = tk.Label(self, text="status...", anchor=tk.W, relief="flat", bd=2)
            self.status.pack(fill="x", side="bottom", ipady=2)

        # configuration panel
        if init_config_panel is not None:
            # create a frame for configuration gui
            self.config_panel = tk.Frame(self)  # background="red" to debug layout
            init_config_panel(self.config_panel)
            self.config_panel.pack(side=tk.RIGHT, fill=tk.Y, expand=0)

        # figure
        if figure_update is not None:
            self.figure = figure.Figure(figsize = (10, 8), dpi=100)
            self.canvas = backend.FigureCanvasTkAgg(self.figure, master=self)
            self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.call_periodic_figure_update_fun()
        self.call_periodic_status_update_fun()

    def call_figure_update_fun(self, update_tuple):
        update_tuple[0](self.figure, *update_tuple[1])
        self.canvas.draw()

    def call_status_update_fun(self, update_tuple):
        s = update_tuple[0](*update_tuple[1])
        self.status.config(text=s)

    def call_periodic_figure_update_fun(self):
        if self.status_update is not None:
            self.call_figure_update_fun(self.figure_update)
            self.after(int(1000 * self.figure_update[2]), self.call_periodic_figure_update_fun)

    def call_periodic_status_update_fun(self):
        if self.status_update is not None:
            self.call_status_update_fun(self.status_update)
            self.after(int(1000 * self.status_update[2]), self.call_periodic_status_update_fun)
