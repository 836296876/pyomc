from pyomc.my_decorator import timer
get_runtime = timer

import pyomc.file_process
windowsPath = pyomc.file_process.windowsPath
mumax3_convert = pyomc.file_process.mumax3_convert
fold_2_axis = pyomc.file_process.fold_2_axis
fold_2_axis_peraxis = pyomc.file_process.fold_2_axis_peraxis
tocomsol = pyomc.file_process.tocomsol

import pyomc.ft
sw_ft2d = pyomc.ft.sw_ft2d
sw_ft2d_peraxis = pyomc.ft.sw_ft2d_peraxis
sw_ft1d = pyomc.ft.sw_ft1d

import pyomc.plot_process
plot_dispersion = pyomc.plot_process.plot_dispersion