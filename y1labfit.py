# -*- coding: utf-8 -*-
# pylint: disable=C0301, C0103, C0111, E0611, W0201

"""
Created on Mon Sep  2 20:01:15 2019

@author: Wolfgang Theis

GUI program for plotting and fitting for y1 physics lab

version 1.0beta2

Data can be pasted from excel into the table using the menu item or
selecting any cell and pressing cntr-v

Error values will be replicated using values entered in previous rows

The script should be called using python 3 as follows:
python y1labfit.py

Command-line arguments:
-x   Expose functionality to take account of delta_x uncertainty in the data.
-i   Infer confidence intervals from data scatter if no errors for data provided

The GUI can also be invoked from inside other scripts or from the python console
with optional parameters to pass in data:

import numpy as np
xarr = np.linspace(-10,10,30)
yarr = 0.1*xarr + 0.5 * np.random.rand(xarr.shape[0])
xerr_arr =  None
yerr_arr = np.ones(30)*0.05

import y1labfit

y1labfit.main(x=xarr, y=yarr, xerr=xerr_arr, yerr=yerr_arr)

"""

import sys
import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QComboBox, QTableWidget, QTableWidgetItem, QDialog, QAction
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSizePolicy, QLineEdit, QPushButton, QFormLayout, QFileDialog, QMessageBox, QCheckBox
from PyQt5 import QtPrintSupport

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

def error_string(v, e, sig=2):
    # extract exponent
    estr = f'{e:#.{sig}g}'.split('e')
    mant = estr[0]
    exp = None
    if len(estr) > 1:
        exp = int(estr[1])
        return f'({v/10**exp:.{len(mant)-2}f} $\\pm$ {mant}) 10$^{{{exp}}}$'
    return f'({v:.{len(mant)-2}f} $\\pm$ {mant})'

def show_info(msgtext):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(msgtext)
    msg.setWindowTitle("Notice")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()


class FitFunction():

    name = 'abstract'
    yofx_latex = 'none'
    param_names = []
    param_units = []

    @classmethod
    def calc(cls, x, a, b):
        return a*x+b

    @classmethod
    def detailed_text(cls, popt, perr):
        names = getattr(cls, 'param_names_latex', cls.param_names)
        detailed = cls.yofx_latex + '\n\nOptimal fit parameters:\n'
        detailed += ',\n'.join([f'${vn}$ = {error_string(poptv, perrv)} {vu}'
                                for vn, poptv, perrv, vu in
                                zip(names, popt, perr, cls.param_units)])
        return detailed

    @classmethod
    def pguess(cls, x, y):  # pylint: disable = W0613
        return np.ones(len(cls.param_names))

    @classmethod
    def legend(cls, popt):
        return 'fit with ' + ', '.join([f'{vn} = {popt:#.3g}' for vn, popt in zip(cls.param_names, popt)])

    @classmethod
    def calc_odr(cls, beta, x):
        return cls.calc(x, *beta)

    @classmethod
    def fit(cls, x, y, sigma=None, xsigma=None, p0=None):
        if p0 is None:
            p0 = cls.pguess(x, y)
        if np.any(np.isnan(xsigma)):
            xsigma = None
        if np.any(np.isnan(sigma)):
            sigma = None
        if xsigma is None:
            try:
                if sigma is None:
                    popt, pcov = curve_fit(cls.calc, x, y, p0=p0, absolute_sigma=False)
                    # sometimes it doesn't converge because a value is very small 1e-12 and maybe step sizes are too small
                    if np.any(np.isnan(pcov) | np.isinf(pcov)):
                        popt[np.abs(popt) < 1e-8] = 0.0
                        popt, pcov = curve_fit(cls.calc, x, y, p0=popt, absolute_sigma=False)
                else:
                    popt, pcov = curve_fit(cls.calc, x, y, p0=p0, sigma=sigma, absolute_sigma=True)
                    # sometimes it doesn't converge because a value is very small 1e-12 and maybe step sizes are too small
                    if np.any(np.isnan(pcov) | np.isinf(pcov)):
                        popt[np.abs(popt) < 1e-8] = 0.0
                        popt, pcov = curve_fit(cls.calc, x, y, p0=popt, sigma=sigma, absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))
                yfit = cls.calc(x, *popt)
                eps = (yfit-y)
                if sigma is not None:
                    eps /= sigma
            except (RuntimeError, Exception):
                return None
        else:
            data = RealData(x, y, sx=xsigma, sy=sigma)
            model = Model(cls.calc_odr)
            odr = ODR(data, model, p0)
            odr.set_job(fit_type=0)
            output = odr.run()
            popt = output.beta
            perr = np.sqrt(np.diag(output.cov_beta))
            yfit = cls.calc(x, *popt)
            eps = output.eps
            # print(output.__dict__)
        legend_text = cls.legend(popt)
        return popt, perr, yfit, eps, legend_text, cls.detailed_text(popt, perr)

    @classmethod
    def model(cls, x, p):
        y = cls.calc(x, *p)
        legend_text = cls.legend(p)
        return y, legend_text


class FitFunction_linear(FitFunction):

    name = 'linear, y = ax + b'
    yofx_latex = '$y = a x + b$'
    param_names = ['a', 'b']
    param_units = ['yunits/xunit', 'yunits']

    @classmethod
    def calc(cls, x, a, b):     # pylint: disable=W0221
        return a*x+b


class FitFunction_quadratic(FitFunction):

    name = 'quadratic, y = ax**2 + bx + c'
    yofx_latex = '$y = a x^2 + b x + c$'
    param_names = ['a', 'b', 'c']
    param_units = ['yunits/xunits$^2$', 'yunits/xunits', 'yunits']

    @classmethod
    def calc(cls, x, a, b, c):     # pylint: disable=W0221
        return a*x**2 + b*x + c

class FitFunction_exponential(FitFunction):

    name = 'exponential, y = a*exp(bx)'
    yofx_latex = '$y = a e^{b x}$'
    param_names = ['a', 'b']
    param_units = ['yunits', 'xunits$^{-1}$']

    @classmethod
    def calc(cls, x, a, b):     # pylint: disable=W0221
        return a*np.exp(b*x)

class FitFunction_cubic(FitFunction):

    name = 'cubic, y = ax**3 + bx**2 + cx + d'
    yofx_latex = '$y = a x^3 + b x^2 + c x + d$'
    param_names = ['a', 'b', 'c', 'd']
    param_units = ['yunits/xunits$^3$', 'yunits/xunits$^2$', 'yunits/xunits', 'yunits']

    @classmethod
    def calc(cls, x, a, b, c, d):     # pylint: disable=W0221
        return a*x**3 + b*x**2 + c*x + d

class FitFunction_cubic_odd(FitFunction):

    name = 'cubic odd powers, y = ax**3 + bx'
    yofx_latex = '$y = a x^3 + b x$'
    param_names = ['a', 'b']
    param_units = ['yunits/xunits$^3$', 'yunits/xunits']

    @classmethod
    def calc(cls, x, a, b):     # pylint: disable=W0221
        return a*x**3 + b*x

class FitFunction_power(FitFunction):

    name = 'power, y = a abs(x)**p'
    yofx_latex = '$y = a \\|x\\|^p$'
    param_names = ['a', 'p']
    param_units = ['yunits/xunit$^p$', '']

    @classmethod
    def calc(cls, x, a, p):     # pylint: disable=W0221
        return a*np.abs(x)**p


class FitFunction_resonance(FitFunction):

    name = 'resonance, y = ...'
    yofx_latex = '$y =  a (\\Delta x/2)^2 / [(x - x_0)^2 +(\\Delta x/2)^2]$'
    param_names = ['a', 'x_0', 'delta_x']
    param_names_latex = ['a', 'x_0', '\\Delta x']
    param_units = ['yunits', 'xunits', 'xunits']

    @classmethod
    def calc(cls, x, a, x_0, delta_x):     # pylint: disable=W0221
        return a * (delta_x/2)**2 / ((x - x_0)**2 +(delta_x/2)**2)

    @classmethod
    def pguess(cls, x, y):
        return np.array([y.max(), x.mean(), (x.max()-x.min())/5])


class MyMainWindow(QMainWindow):
    ''' the main window potentially with menus, statusbar, ... '''

    def __init__(self, argv, x=None, y=None, xerr=None, yerr=None):
        super().__init__()
        self.central_widget = MyCentralWidget(self, argv)
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle('Plot and Fit program for y1 lab')
        self.central_widget.control.set_data(x, y, xerr, yerr)
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        action = QAction('&Print...', self)
        action.setShortcut('Ctrl+P')
        action.setStatusTip('print plot')
        action.triggered.connect(self.handlePrint)
        fileMenu.addAction(action)
        action = QAction('&Save as png...', self)
        action.setShortcut('Ctrl+S')
        action.setStatusTip('save as png')
        action.triggered.connect(self.handleSave)
        fileMenu.addAction(action)
        action = QAction('Paste data', self)
        action.setShortcut('Ctrl+V')
        action.setStatusTip('paste data to data table')
        action.triggered.connect(self.handlePaste)
        fileMenu.addAction(action)
        self.statusBar().showMessage('')
        self.resize(800 + len(self.central_widget.control.data_col_names)*80, 900)
        self.move(400, 10)

    def handlePrint(self):
        dialog = QtPrintSupport.QPrintDialog()
        if dialog.exec_() == QDialog.Accepted:
            self.central_widget.mpl_widget.render(dialog.printer())

    def handleSave(self):
        name = QFileDialog.getSaveFileName(self, 'Save File', filter='.png')
        self.central_widget.mpl_widget.fig.savefig(name[0]+name[1], dpi=300)

    def handlePaste(self):
        try:
            clip = QApplication.clipboard()
            mime = clip.mimeData()
            if 'text/plain' in mime.formats():
                data = mime.data('text/plain').data()
                data = data.decode(encoding="ascii")
                # parse data
                d = np.array([row.split('\t') for row in data.split('\n') if len(row.split('\t')) > 1]).astype(float)
                if d.shape[1] == 4 and len(self.central_widget.control.data_col_names) == 4:
                    self.central_widget.control.set_data(d[:, 0], d[:, 1], d[:, 2], d[:, 3])
                elif d.shape[1] == 3:
                    self.central_widget.control.set_data(d[:, 0], d[:, 1], None, d[:, 2])
                elif d.shape[1] == 2:
                    self.central_widget.control.set_data(d[:, 0], d[:, 1], None, None)
                else:
                    show_info('Data to paste must have at least two columns and no more than the table')
            else:
                show_info('Data does not have a suitable format for pasting into the table')
        except Exception:
            show_info('Data does not have a suitable format for pasting into the table')


class MyCentralWidget(QWidget):
    ''' everything in the main area of the main window '''

    def __init__(self, main_window, argv):
        super().__init__()
        self.main_window = main_window
        # define figure canvas
        self.mpl_widget = MyMplWidget(self.main_window)
        self.control = ControlWidget(self.mpl_widget, argv)
        # place MplWidget into a vertical box layout
        hbox = QHBoxLayout()
        hbox.addWidget(self.control) # add the figure
        hbox.addWidget(self.mpl_widget) # add the figure
        #use the box layout to fill the window
        self.setLayout(hbox)


class ControlWidget(QWidget):

    nrpoints = 50
    max_param = 4
    fit_funtion_classes = [FitFunction_linear, FitFunction_quadratic,
                           FitFunction_cubic, FitFunction_cubic_odd,
                           FitFunction_power, FitFunction_resonance, FitFunction_exponential]
    data_col_names = ['X', 'Y', 'delta X', 'delta Y']
    #data_col_names = ['X', 'Y', 'delta Y']

    def __init__(self, mpl_widget, argv):
        super().__init__()
        self.mpl_widget = mpl_widget
        self.details = {}
        self.fitfunction = None
        self.argv = argv
        if '-x' in self.argv:
            self.data_col_names = ['X', 'Y', 'delta X', 'delta Y']
        else:
            self.data_col_names = ['X', 'Y', 'delta Y']
        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()
        self.createTable()
        vbox.addWidget(self.tableWidget)

        fbox = QFormLayout()
        lbl = QLabel("Title", self)
        self.ed_title = QLineEdit("Plot title", self)
        fbox.addRow(lbl, self.ed_title)

        lbl = QLabel("X label", self)
        self.ed_xlabel = QLineEdit("X label", self)
        fbox.addRow(lbl, self.ed_xlabel)

        lbl = QLabel("Y label", self)
        self.ed_ylabel = QLineEdit("Y label", self)
        fbox.addRow(lbl, self.ed_ylabel)

        lbl = QLabel("Signature", self)
        self.ed_name = QLineEdit("Name, date", self)
        fbox.addRow(lbl, self.ed_name)
        vbox.addLayout(fbox)

        btn = QPushButton('Plot data', self)
        btn.clicked.connect(self.onPlot)
        vbox.addWidget(btn)

        self.fitbox = QFormLayout()
        lbl = QLabel("fit model", self)
        self.combo = QComboBox(self)
        self.combo.addItem("None")
        for f in self.fit_funtion_classes:
            self.combo.addItem(f.name)
        self.combo.activated[str].connect(self.onActivated)
        self.fitbox.addRow(lbl, self.combo)
        self.ed_param = []
        self.p_lbl = []
        for _ in range(self.max_param):
            self.ed_param.append(QLineEdit("", self))
            self.p_lbl.append(QLabel("unused", self))
            self.fitbox.addRow(self.p_lbl[-1], self.ed_param[-1])
        self.cb_scaled = QCheckBox('show scaled residuals', self)
        self.cb_scaled.setChecked(True)
        self.fitbox.addRow(self.cb_scaled)
        showbtn = QPushButton('Plot data and model', self)
        showbtn.clicked.connect(self.onPlotWithModel)
        self.fitbox.addRow(showbtn)
        fitbtn = QPushButton('Plot data and fit', self)
        fitbtn.clicked.connect(self.onFit)
        self.fitbox.addRow(fitbtn)
        vbox.addLayout(self.fitbox)
        self.setLayout(vbox)

    def createTable(self):
       # Create table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(self.nrpoints)
        self.tableWidget.setColumnCount(len(self.data_col_names))
        self.tableWidget.setHorizontalHeaderLabels(self.data_col_names)
        for k in range(4):
            self.tableWidget.setColumnWidth(k, 70)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.move(0, 0)

    def get_values(self):
        self.data = np.zeros((self.tableWidget.rowCount(), 4))
        self.data[:, :] = np.nan
        for r in range(self.tableWidget.rowCount()):
            for c in range(4):
                try:
                    self.data[r, c] = float(self.tableWidget.item(r, c).text())
                except Exception:      # pylint: disable=W0703
                    pass
        self.data = self.data[~np.isnan(self.data[:, 0]), :]
        self.data = self.data[~np.isnan(self.data[:, 1]), :]
        if len(self.data_col_names) == 3:
            self.data[:, 3] = self.data[:, 2]
            self.data[:, 2] = np.nan
        # replicate y errors
        if not np.isnan(self.data[0, 3]):
            for k in range(1, self.data.shape[0]):
                if np.isnan(self.data[k, 3]):
                    self.data[k, 3] = self.data[k-1, 3]
        # if xerr or yerr only partial, provide warning
        if np.any(np.isnan(self.data[:, 2])) and np.any(~np.isnan(self.data[:, 2])):
            show_info('Some delta_x missing: All will be ignored when fitting')
        if np.any(np.isnan(self.data[:, 3])) and np.any(~np.isnan(self.data[:, 3])):
            show_info('Some delta_y missing: All will be ignored when fitting')
        self.details['data'] = self.data
        self.details['title'] = self.ed_title.text()
        self.details['xlabel'] = self.ed_xlabel.text()
        self.details['ylabel'] = self.ed_ylabel.text()
        self.details['signature'] = self.ed_name.text()
        self.details['fitdata'] = None
        self.details['detailed_text'] = None
        self.details['is_scaled'] = self.cb_scaled.isChecked()
        try:
            self.p0 = np.array([float(p.text()) for p in self.ed_param[:len(self.fitfunction.param_names)]])
        except Exception: # pylint: disable=W0703
            self.p0 = None

    def set_values(self, popt):
        if self.fitfunction is not None:
            for k, p in enumerate(self.ed_param):
                if k < len(self.fitfunction.param_names) and not np.isnan(popt[k]):
                    p.setText(f'{popt[k]:#.4g}')

    def set_data(self, x, y, xerr, yerr):
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(self.data_col_names)
        if len(self.data_col_names) == 3:
            # shift data for it to end up in the correct column
            xerr = yerr
            yerr = None
        if x is not None:
            for k, xv in enumerate(x):
                self.tableWidget.setItem(k, 0, QTableWidgetItem(f'{xv:.3g}'))
        if y is not None:
            for k, yv in enumerate(y):
                self.tableWidget.setItem(k, 1, QTableWidgetItem(f'{yv:.3g}'))
        if xerr is not None:
            for k, xerrv in enumerate(xerr):
                self.tableWidget.setItem(k, 2, QTableWidgetItem(f'{xerrv:.3g}'))
        if yerr is not None:
            for k, yerrv in enumerate(yerr):
                self.tableWidget.setItem(k, 3, QTableWidgetItem(f'{yerrv:.3g}'))

    def onActivated(self, text):
        self.fitfunction = None
        if self.combo.currentIndex() > 0:
            self.fitfunction = self.fit_funtion_classes[self.combo.currentIndex()-1]()
        self.details['fitlabel'] = text.split(',')[0]
        if self.fitfunction is not None:
            for k, lab in enumerate(self.p_lbl):
                if k < len(self.fitfunction.param_names):
                    lab.setText(self.fitfunction.param_names[k])
                else:
                    lab.setText('unused')
        for ed in self.ed_param:
            ed.setText("")

    def onPlot(self):
        self.get_values()
        self.mpl_widget.plot(self.details)

    def onPlotWithModel(self):
        self.get_values()
        if self.combo.currentIndex() != 0 and self.p0 is not None:
            try:
                x = np.linspace(np.nanmin(self.data[:, 0]), np.nanmax(self.data[:, 0]), 100)
                y, legendtext = self.fitfunction.model(x, self.p0)
                self.fitdata = np.array([x, y]).T
                self.details['fitdata'] = self.fitdata
                self.details['fitlabel'] = legendtext
                self.details['scaled_difference'] = (self.fitfunction.calc(self.data[:, 0], *self.p0) - self.data[:, 1])
                if not np.any(np.isnan(self.data[:, 3])) and self.cb_scaled.isChecked():
                    self.details['scaled_difference'] /= self.data[:, 3]
            except Exception:
                show_info('Function could not be evaluated for provided X values and parameters')
                self.mpl_widget.plot(self.details)
        else:
            show_info('Please select function and provide parameters')
        self.mpl_widget.plot(self.details)

    def onFit(self):
        self.get_values()
        if self.combo.currentIndex() != 0:
            if self.data.shape[0] < len(self.fitfunction.param_names):
                show_info('More data points are required for fitting')
            elif (np.any(np.isnan(self.data[:, 2])) and np.any(np.isnan(self.data[:, 3])) and '-i' not in self.argv):
                if len(self.data_col_names) == 3:
                    show_info('Please provide a complete set of delta_y values to allow fitting')
                else:
                    show_info('Please provide a complete set of delta_x and/or delta_y values to allow fitting')
            elif np.any(self.data[:, 3]==0):
                show_info('Some delta_y are exactly zero: Please change these to non-zero values')
            else:
                ret = self.fitfunction.fit(self.data[:, 0], self.data[:, 1],
                                           sigma=self.data[:, 3], xsigma=self.data[:, 2], p0=self.p0)
                if ret is None:
                    show_info('Fitting failed, try providing better starting values')
                    self.mpl_widget.plot(self.details)
                    return
                popt, _, _, eps, legend_text, detailed_text = ret
                self.details['fitlabel'] = legend_text
                self.details['detailed_text'] = detailed_text
                x = np.linspace(np.nanmin(self.data[:, 0]), np.nanmax(self.data[:, 0]), 100)
                y = self.fitfunction.calc(x, *popt)
                self.fitdata = np.array([x, y]).T
                self.details['fitdata'] = self.fitdata
                if self.cb_scaled.isChecked():
                    self.details['scaled_difference'] = eps
                else:
                    self.details['scaled_difference'] = (self.fitfunction.calc(self.data[:, 0], *popt) - self.data[:, 1])
                self.set_values(popt)
        else:
            show_info('Please select fit function')
        self.mpl_widget.plot(self.details)

    def onChanged(self, text):
        pass
        #self.main_window.central_widget.mpl_widget.plot()


class MyMplWidget(FigureCanvas):
    ''' both a QWidget and a matplotlib figure '''

    def __init__(self, main_window, parent=None, figsize=(10, 4), dpi=72):
        self.main_window = main_window
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        #self.plot()
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.cid2 = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)


    def plot(self, details):
        ''' clear figure and plot fit '''

        data = details['data']
        fitdata = details['fitdata']
        has_errorx = not np.any(np.isnan(data[:, 2]))
        has_errory = True

        self.fig.clf()
        # create the figure
        if fitdata is not None:
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        else:
            gs = gridspec.GridSpec(1, 1)

        # main panel with fit
        ax = self.fig.add_subplot(gs[0])
        ax.tick_params(axis='both', labelsize=12)
        # plot data with errorbars if provided
        if has_errorx and has_errory:
            ax.errorbar(data[:, 0], data[:, 1], xerr=data[:, 2], yerr=data[:, 3], fmt='ko', label='data')
        elif has_errorx:
            ax.errorbar(data[:, 0], data[:, 1], xerr=data[:, 2], fmt='ko', label='data')
        elif has_errory:
            ax.errorbar(data[:, 0], data[:, 1], yerr=data[:, 3], fmt='ko', label='data')
        else:
            ax.plot(data[:, 0], data[:, 1], 'ko', label='data')
        if fitdata is not None:
            ax.plot(fitdata[:, 0], fitdata[:, 1], 'r', label=details['fitlabel'])
        #ax.set_yscale("log")
        #ax.set_ylim((0, 10))
        #ax.set_ylim((1e2, 1e5))
        ax.set_xlabel(details['xlabel'], fontsize=14)
        ax.set_ylabel(details['ylabel'], fontsize=14)
        ax.set_title(details['title'], fontsize=14)
        ax.legend(loc='upper right', fontsize=14)
        if details['detailed_text'] is not None:
            ax.text(0.02, 0.98, details['detailed_text'], transform=ax.transAxes,
                    verticalalignment='top', fontsize=16)
        if details['signature'] is not None:
            ax.text(0.98, 0.02, details['signature'], transform=ax.transAxes,
                    horizontalalignment='right', fontsize=16)
        # second panel with normalized deviation of data and fit
        if fitdata is not None and not has_errorx:
            ax2 = self.fig.add_subplot(gs[1])
            ax2.tick_params(axis='both', labelsize=12)
            if details['is_scaled']:
                select = np.abs(details['scaled_difference']) <= 1
                ax2.errorbar(data[select, 0], details['scaled_difference'][select], yerr=1, fmt='go', label='data')
                select = (np.abs(details['scaled_difference']) > 1) & (np.abs(details['scaled_difference']) < 3)
                ax2.errorbar(data[select, 0], details['scaled_difference'][select], yerr=1, c='orange', marker='o', ls='', label='data')
                select = np.abs(details['scaled_difference']) >= 3
                ax2.errorbar(data[select, 0], details['scaled_difference'][select], yerr=1, fmt='ro', label='data')
                d = data[:, 0].max() - data[:, 0].min()
                mi = data[:, 0].min() - d
                ma = data[:, 0].max() + d
                ax2.plot([mi, ma], [1, 1], 'k:')
                ax2.plot([mi, ma], [0, 0], 'k-')
                ax2.plot([mi, ma], [-1, -1], 'k:')
                r = np.max([3, 1.1*(np.max(np.abs(details['scaled_difference']))+1)])
                if np.isnan(r) or np.isinf(r):
                    r = 100
                ax2.set_ylim((-r, r))
                ax2.set_xlim(ax.get_xlim())
                ax2.set_xlabel(details['xlabel'], fontsize=14)
                ax2.set_ylabel("scaled difference", fontsize=14)
                ax2.set_title("Difference between fit and data scaled by expected error", fontsize=14)
            else:
                ax2.errorbar(data[:, 0], details['scaled_difference'], yerr=data[:, 3], fmt='ko', label='data')
                d = data[:, 0].max() - data[:, 0].min()
                mi = data[:, 0].min() - d
                ma = data[:, 0].max() + d
                ax2.plot([mi, ma], [0, 0], 'k-')
                r = np.max(1.1*(np.max(np.abs(details['scaled_difference']) + data[:, 3])))
                if np.isnan(r) or np.isinf(r):
                    r = 100
                ax2.set_ylim((-r, r))
                ax2.set_xlim(ax.get_xlim())
                ax2.set_xlabel(details['xlabel'], fontsize=14)
                ax2.set_ylabel("difference", fontsize=14)
                ax2.set_title("Difference between fit and data", fontsize=14)
        self.fig.tight_layout()
        self.draw()

    def on_mouse_move(self, event):
        ''' displays a status message when the mouse pointer moves over the figure '''
        if event.xdata is not None and event.ydata is not None:                             # <-- checks that position is inside axes
            msg = f'x={event.xdata:.2f}, y={event.ydata:.2f}'
            self.main_window.statusBar().showMessage(msg)


def main(x=None, y=None, xerr=None, yerr=None):
    app = QApplication(sys.argv)
    if xerr is not None:
        w = MyMainWindow(['', '-x'], x=x, y=y, xerr=xerr, yerr=yerr)
    else:
        w = MyMainWindow(sys.argv, x=x, y=y, xerr=xerr, yerr=yerr)
    w.show()
    app.exec()

if __name__ == '__main__':
    main()
