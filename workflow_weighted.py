#%%
from inspect import stack
from re import split
from typing import NamedTuple
from astropy.io.fits import file
#from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from numpy.ma import asanyarray
plt.style.use(astropy_mpl_style)

#from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata import utils
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve
from astropy import constants as const
#from astropy.modeling import Fittable1DModel
import numpy as np
#from astropy.io.fits import compression
from photutils.segmentation import make_source_mask
import os
import sys
from scipy import stats, optimize
from scipy.stats import loguniform
from datetime import date, datetime
import timeit
import pandas as pd
from numpy import add, multiply, divide, float64, random, pi, float32

#maps = []

#coadds = []
coadd = fits.open('/data/spt3g/products/maps/yearly_winter_2020/yearly_150GHz_winter_2020_tonly.fits.fz')
wcs_coadd = WCS(header=coadd[1].header)
coadd_data = coadd[1].data

def make_mask():
    coadd_ = fits.open('/data/spt3g/products/maps/yearly_winter_2020/yearly_150GHz_winter_2020_tonly.fits.fz')
    mask = make_source_mask(coadd_[1].data, nsigma=6, npixels=12, mask=None, filter_fwhm=None,
    filter_size=3, filter_kernel=None, sigclip_sigma=3, sigclip_iters=7, dilate_size=11)
    mask = np.invert(mask)
    return mask

def load_mask():
    f = fits.open('/data/pitocco2/workspace/PointSourceMask/1500d_ptsrc_mask.fits')
    data = f[1].data
    mask = np.logical_not(data).astype(int)
    return mask


def select_file(day, month, year, hour, minute, second, frequency:str):
    frequency = frequency + 'GHz'
    telescope_time = datetime(year=2017, month=1, day=1, hour=0, second=0, minute=0)
    search_time = datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    print(search_time)
    difference = (search_time - telescope_time)
    search_time = difference.total_seconds()

    file_list = list_files('/data/spt3g/products/maps')
    split_file_list = []
    for filename in file_list:
        split_ = filename.split("_")
        if (split_[1] == frequency):
            split_file_list.append(split_)
    number_list = []
    for filename in split_file_list:
        number_list.append(filename[0])
    number_list = [int(i) for i in number_list]
    closest_time = closest(number_list, search_time)
    print("CLOSEST IS")
    print(closest_time)
    filename_ = str(closest_time) + '_' + frequency + '_tonly.fits.fz'
    print(filename_)
    return filename_

def list_files(folder: str):
    path = folder
    file_list = []
    for path, folders, files in os.walk(path):
        for file in files:
            file_list.append(file)
            #print(file + '\t' +  fits.open('/data/spt3g/products/maps/' + file)[1].header['OBJECT'])
    # for filename in file_list:
    #     print(filename)
    return file_list

def closest(list, number):
    list = np.asarray(list)
    idx = (np.abs(list - number)).argmin()
    return list[idx]


def load_maps(listfile: str, ra, dec):
    start = timeit.default_timer()
    readfile = open(listfile, "r")
    Lines = readfile.readlines()
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='fk5')
    print(coord.to_string)
    #cocut = cutout_fits_compressed('/data/spt3g/products/maps/yearly_winter_2020/yearly_150GHz_winter_2020_tonly.fits.fz', coord, (12.2*u.degree, 12*u.degree))
    cocut = Cutout2D(fits.open('/data/spt3g/products/maps/yearly_winter_2020/yearly_150GHz_winter_2020_tonly.fits.fz')[1].data, coord, (12.2*u.degree, 12*u.degree), wcs=wcs_coadd, copy=True)
    i=1
    maps = []
    weights = []
    wcs_ = []
    while i < len(Lines):
        line = Lines[i].split()
        filename_ = '/data/spt3g/products/maps/' + line[2] + '-' + line[3] + '/' + line[0]
        print (filename_)
        start = timeit.default_timer()
        cutout_data, cutout_weight = cutout_fits_compressed(filename_, coord, (12.2*u.degree, 12*u.degree))
        maps.append(np.asanyarray(cutout_data.data))
        weights.append(np.asanyarray(cutout_weight.data))
        #maps.append(cutout.data)
        wcs_.append(cutout_data.wcs)
        stop = timeit.default_timer()
        print ('Cutout time: ', stop - start)
        i=i+1
    
    # while i < 301:
    #     maps.append(cutout_fits_compressed('Freda Observations\Freda_fits_maps\\111897613_150GHz_tonly.fits', coord, 1440))
    #     i=i+1
    stop = timeit.default_timer()
    print("LOAD TIME: ", stop-start)
    return maps, weights, wcs_, cocut

def cutout_fits_compressed(map_file, sky_coord: SkyCoord, size: int):
    f = fits.open(map_file)
    w = WCS(f[1].header)
    #data = np.divide(f[1].data, f[2].data, where=f[2].data!=0)
    cutout_data: Cutout2D
    cutout_weight: Cutout2D
    try:
        start = timeit.default_timer()
        cutout_data = Cutout2D(f[1].data, sky_coord, size, wcs=w, copy=True)
        cutout_weight = Cutout2D(f[2].data, sky_coord, size, wcs=w, copy=True)
        cutout_mask = Cutout2D(mask, sky_coord, size, wcs=wcs_coadd)
        cutout_coadd = Cutout2D(coadd_data, sky_coord, size, wcs=wcs_coadd)
        cutout_data.data = np.subtract(cutout_data.data, cutout_coadd.data)
        cutout_data.data = np.multiply(cutout_data.data, cutout_mask.data)
        cutout_data.data = convolve(cutout_data.data, Gaussian2DKernel(1))
        stop = timeit.default_timer()
        print('Get Data Time: ', stop - start)
    except:
        print('IN CUTOUT_FITS_COMPRESSED: ' + sky_coord.to_string())
        print('Arrays do not overlap. Requesting data outside of map. Returning nothing')
        return
    f.close()
    return cutout_data, cutout_weight


def cutouts_from_list(orbitfile, mapset, wcsset):
    #start_time = timeit.default_timer()
    readfile = open(orbitfile, "r")
    Lines = readfile.readlines()
    i=1
    #blank = np.zeros((50,50))
    cut_name_tup_list = []
    length_lines = len(Lines)
    print('length', length_lines)
    length_maps=len(mapset)
    while i < length_lines:
        #orbit_cutouts = blank[:]
        orbit_cutouts = []
        params = Lines[i]
        i=i+1
        j=0
        #start = timeit.default_timer()
        while j < length_maps:
            line = Lines[i].split()
            #v this line takes 0.0005456600338220596 sec
            #coord = SkyCoord(ra=np.float64(line[0])*u.degree, dec=np.float64(line[1])*u.degree, frame='fk5')
            #print(coord.to_string())
            #orbit_cutouts.append(Cutout2D(mapset[j], coord, 50, wcs=wcsset[j]))
            x,y = wcsset[j].all_world2pix(float64(line[0]),float64(line[1]), 0)
            x = int(x.round())
            y = int(y.round())
            orbit_cutouts.append(mapset[j][y-24:y+26,x-24:x+26])
            #cutout = mapset[j][y-24:y+26,x-24:x+26]
            #orbit_cutouts = np.add(orbit_cutouts, cutout )
            #orbit_cutouts.append(mapset[j][y-12:y+13,x-12:x+13])
            #orbit_cutouts.append(utils.extract_array(mapset[j], shape=(50,50), position=(y,x)))
            #orbit_cutouts.append(Cutout2D(mapset[j], position=wcsset[j].all_world2pix(np.float64(line[0]),np.float64(line[1]), 0), size=50))
            i=i+1
            j=j+1
        #orbit_cutouts = orbit_cutouts / leng
        cut_name_tup_list.append([orbit_cutouts, params])
        #stop = timeit.default_timer()
        #print('stack one orbit: ', stop - start)
    #stop_time = timeit.default_timer()
    #print('smaller file time: ', stop_time-start_time)
    return cut_name_tup_list



# if we want to divide by the number of of cutouts then [ x / len(cutouts_) for x in cutouts_[0].data ]
def stack_cutouts(cutouts_):
    #print('before ' + '\n', cutouts_)
    n=1
    stacked = cutouts_[0]
    length = len(cutouts_)
    #print(cutouts_[0].data)
    while n<length:
        stacked = add(stacked, cutouts_[n])
        n=n+1
    length = len(cutouts_)
    #stacked = [ x / length for x in stacked ]
    stacked = stacked / length
    #print('after', '\n', stacked )
    return stacked

def to_signal(stack):
    #a = np.asanyarray(stack)
    sd = stack.std(axis=None, ddof=0)
    #print(sd)
    #signals = [x /sd for x in stack ]
    #print( stack,'\n', sd)
    signals = stack / sd
    max = np.amax(signals, axis=None)
    return max

def to_signal_all(stack):
    a = np.asanyarray(stack)
    sd = a.std(axis=None, ddof=0)
    signals = [x /sd for x in stack ]
    return signals

def main(orbitfile_folder, mapset):
    #maps = load_maps(listfile, ra, dec)

    orbitfiles = list_files(orbitfile_folder)
    print(orbitfiles[0])
    #stacked_orbits = []
    stacked_tuples = []
    i = 0
    while i < len(orbitfiles):
        #coadds = []
        print(len(orbitfiles))
        print(orbitfile_folder + '/' + orbitfiles[i])
        #stacked_orbits.append(stack_cutouts(cutouts_from_list(orbitfile_folder + '/' + orbitfiles[i], mapset)))
        #stack_name_tuple = [stack_cutouts(cutouts_from_list(orbitfile_folder + '/' + orbitfiles[i], mapset)), orbitfiles[i]]
        stacked_tuples.append( [ stack_cutouts(cutouts_from_list(orbitfile_folder + '/' + orbitfiles[i], mapset)), orbitfiles[i]] )
        i=i+1
    print(len(orbitfiles))
    return stacked_tuples

#coadd = cutout_fits_compressed('yearly_90GHz_winter_2020.fits', SkyCoord(ra=342*u.degree, dec=-51*u.degree, frame='fk5'), 1440)
#maps = load_maps('/data/pitocco2/workspace/Maps/Tanete/Tanete_May-June.txt', 313, -46.74915285)

#coadd = load_coadd('yearly_90GHz_winter_2020.fits', 342, -51)

def main2(datfile_folder, number_of_loops):
    #maps = load_maps(listfile, ra, dec)
    start = timeit.default_timer()
    orbitfiles = list_files(datfile_folder)
    print(orbitfiles[0])
    stacked_orbits = []
    i = 0
    while i < number_of_loops:
        stacked_orbits.append(stack_cutouts(cutouts_from_list('Orbits\RADEC.txt')))
        i=i+1
    print(len(orbitfiles))
    stop = timeit.default_timer()
    print('FULL STACK TIME:', stop-start )
    return stacked_orbits

def main3(orbitfile_folder, mapset, wcsset):
    start_whole = timeit.default_timer()
    #maps = load_maps(listfile, ra, dec)
    start = timeit.default_timer()
    orbitfiles = list_files(orbitfile_folder)
    stop = timeit.default_timer()
    print('listfiles: ', stop - start)
    print(len(orbitfiles))
    #print(orbitfiles[0])
    #stacked_orbits = []
    stacked_tuples = []
    i = 0
    while i < len(orbitfiles):
        start = timeit.default_timer()
        #print(orbitfile_folder + '/' + orbitfiles[i])
        orbit_name_tuples = cutouts_from_list(orbitfile_folder + '/' + orbitfiles[i], mapset, wcsset)
        for x in orbit_name_tuples:
            x[0] = stack_cutouts(x[0])
            #sigmas
            if ( to_signal(x[0]) > 5 ):
                stacked_tuples.append(x)   
        i=i+1
        stop = timeit.default_timer()
        print('Time per file: ', stop - start)
    print(len(orbitfiles))
    stop_whole = timeit.default_timer()
    print('Whole Time: ', stop_whole-start_whole)
    return stacked_tuples

def main_track(orbitfile_folder, mapset, wcsset, xlfile, cut_coadd, directory):
    stacked_tuples = main3(orbitfile_folder, mapset, wcsset)
    dt, eps, x_he, y_he, z_he = helper_set_window(xlfile)
    filenames = [x[1] for x in stacked_tuples]
    radec_tracks = []
    for x in filenames:
        params = x[x.find('[')+1:x.find(']')].split(', ')
        params = [np.float64(i) for i in params]
        print("Parameters: ", params)
        radec_tracks.append( helper_get_coords(params, dt, eps, x_he, y_he, z_he) )
    plot_tracks(radec_tracks, cut_coadd, filenames, directory)
    plot_all(stacked_tuples, directory)
    return

def plot_all(stacked_orbits, directory):
    if (len(stacked_orbits) == 0):
        return
    #min_max_param(stacked_orbits)
    #print(e_max, inc_max, lan_max, argp_max, tan_max, sma_max)
    if (directory == "None"):
        print("not saving images")
        i = 0
        for x in stacked_orbits:
            plt.figure()
            plt.title(" ".join(['Temperature from CMB of stacked sources\n', 'from ', str(x[2]), '\n', str(i)]))
            #smoothed = convolve(x[0], Gaussian2DKernel(1))
            map = plt.imshow(x[0], cmap='gray')
            plt.colorbar(map, label=r'($K_{CMB}$)')
            i=i+1
        return
    else:
        for x in stacked_orbits:
            print('x[2]', x[2])
            plt.figure()
            plt.title(" ".join(['Temperature from CMB of stacked sources\n', 'from ', str(x[2])]))
            #smoothed = convolve(x[0], Gaussian2DKernel(1))
            map = plt.imshow(x[0], cmap='gray')
            plt.colorbar(map, label=r'($K_{CMB}$)')
            #x[1] = x[1].replace('\n', '')
            plt.savefig(directory + '/Tan' +  str(x[2]) + '.png')
            plt.close()

def min_max_param(stacked_orbits):
    print('STACKED ORBITS: ',stacked_orbits)
    filenames = [x[1] for x in stacked_orbits]
    #print('LENGTH, ',len(filenames))
    #print('FILENAMES: ',filenames)
    e_list, inc_list, lan_list, argp_list, tan_list, sma_list = np.array([[],[],[],[],[],[]])
    for x in filenames:
        #print('X: ', x)
        params = x[x.find('[')+1:x.find(']')].split(', ')
        if (params == '' or params == ['']):
            continue
        #params = y[1][y[1].find('[')+1:y[1].find(']')].split(', ')
        #print('PARAMS: ', params)
        # e_list.append(np.float64(params[0])), inc_list.append(np.float64(params[1])), lan_list.append(np.float64(params[2]))
        # argp_list.append(np.float64(params[3])), tan_list.append(np.float64(params[4])), sma_list.append(np.float64(params[5]))
        try:
            e_list, inc_list = np.append(e_list, np.float64(params[0])), np.append(inc_list, np.float64(params[1]))
            lan_list, argp_list = np.append(lan_list, np.float64(params[2])), np.append(argp_list, np.float64(params[3]))
            tan_list, sma_list = np.append(tan_list, np.float64(params[4])), np.append(sma_list, np.float64(params[5]))
        except:
            continue
        #print('ecc ', np.float64(params[0]))
        #print('E List in loop: ', e_list)
    #print('E List: ', e_list)
    e_min, e_max = np.amin(e_list, axis=None), np.amax(e_list, axis=None)
    inc_min, inc_max = np.amin(inc_list, axis=None), np.amax(inc_list, axis=None)
    lan_min, lan_max = np.amin(lan_list, axis=None), np.amax(lan_list, axis=None)
    argp_min, argp_max = np.amin(argp_list, axis=None), np.amax(argp_list, axis=None)
    tan_min, tan_max = np.amin(tan_list, axis=None), np.amax(tan_list, axis=None)
    sma_min, sma_max = np.amin(sma_list, axis=None), np.amax(sma_list, axis=None)
    print('Eccentricity: [' + str(e_min) + ', ' + str(e_max) + ']')
    print('Inclination: [' + str(inc_min) + ', ' + str(inc_max) + ']')
    print('Long. Asc. Node: [' + str(lan_min) + ', ' + str(lan_max) + ']')
    print('Arg. of Peri.: [' + str(argp_min) + ', ' + str(argp_max) + ']')
    print('True Anomaly: [' + str(tan_min) + ', ' + str(tan_max) + ']')
    print('Semi Major Axis: [' + str(sma_min) + ', ' + str(sma_max) + ']')

def histogram(stacked_orbits):
    arrays = [to_signal_all(x[0]) for x in stacked_orbits]
    arrays = np.array(arrays)
    
    plt.figure()
    hist = plt.hist(np.square(arrays[arrays>0].flatten()), bins=100, color='Blue',density=True)
    plt.xlabel(xlabel=r'($(s/n)^{2}$)')
    plt.ylabel(ylabel=r'($log(N)$)')
    ax = plt.subplot()
    ax.set_yscale('log')
    plt.title(r'($(s/n)^{2}$) vs ($log(N)$)')
    hist2 = plt.hist(np.square(arrays[arrays<0].flatten()), bins=100, color='Red', density=True)
    ax = plt.subplot()
    ax.set_yscale('log')
    plt.title('SNR for Positive and Negative Noise')
    plt.savefig('/data/pitocco2/workspace/pngs/FredaSubmit' + '/SNR')

def mainall(orbitfile_folder, mapset, wcsset):
    orbitfiles = list_files(orbitfile_folder)
    stacked_tuples = []
    i = 0
    while i < len(orbitfiles):
        print(len(orbitfiles))
        print(orbitfile_folder + '/' + orbitfiles[i])
        orbit_name_tuples = cutouts_from_list(orbitfile_folder + '/' + orbitfiles[i], mapset, wcsset)
        for x in orbit_name_tuples:
            x[0] = stack_cutouts(x[0])
            stacked_tuples.append(x)
        i=i+1
    print(len(orbitfiles))
    histogram(stacked_tuples)
    return stacked_tuples

def parse_and_plot(stacked_tuples, sigma, directory):
    good_orbits = []
    i=0
    print('Length: ', len(stacked_tuples))
    for x in stacked_tuples:
        if ( to_signal(x[0]) > sigma ):
            good_orbits.append(x)
    plot_all(good_orbits, directory=directory)
    return

def cutouts_from_list_old(orbitfile, mapset, wcsset):
    readfile = open(orbitfile, "r")
    Lines = readfile.readlines()
    i=1
    cut_name_tup_list = []
    while i < len(Lines):
        orbit_cutouts = []
        params = Lines[i]
        i=i+1
        j=0
        start = timeit.default_timer()
        while j < len(mapset):
            if (Lines[i][0] == '['):
                break
            line = Lines[i].split()
            #v this line takes 0.0005456600338220596 sec
            #coord = SkyCoord(ra=np.float64(line[0])*u.degree, dec=np.float64(line[1])*u.degree, frame='fk5')
            #print(coord.to_string())
            #orbit_cutouts.append(Cutout2D(mapset[j], coord, 50, wcs=wcsset[j]))
            x,y = wcsset[j].all_world2pix(np.float64(line[0]),np.float64(line[1]), 0)
            x = int(x.round())
            y = int(y.round())
            orbit_cutouts.append(mapset[j][y-24:y+26,x-24:x+26])
            #orbit_cutouts.append(utils.extract_array(mapset[j], shape=(50,50), position=(y,x)))
            #orbit_cutouts.append(Cutout2D(mapset[j], position=wcsset[j].all_world2pix(np.float64(line[0]),np.float64(line[1]), 0), size=50))
            i=i+1
            j=j+1
        cut_name_tup_list.append([orbit_cutouts, params])
        stop = timeit.default_timer()
        print('stack one orbit: ', stop - start)
    return cut_name_tup_list

def plot_tracks(radec_tracks, cut_coadd, filenames, directory):
    plt.figure(dpi=200)
    plt.title('Tracks on Coadd')
    ax = plt.subplot(projection=cut_coadd.wcs)
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_major_formatter('d.d')
    lat.set_major_formatter('d.d')
    lat.set_ticks(spacing=2*u.degree)
    smoothed = convolve(cut_coadd.data , Gaussian2DKernel(1))
    #print('length radec ',len(radec_tracks))
    n=0
    for z in radec_tracks:
        #do the iterator thing for RAs and DECs
        RAs = [i[0] for i in z]
        # print('len RAs', len(RAs))
        # print(RAs)
        DECs = [i[1] for i in z]
        # print(RAs)
        # print(DECs)
        x_coords = []
        y_coords = []
        i = 0
        while i < len(RAs):
            x,y = cut_coadd.wcs.all_world2pix(RAs[i],DECs[i], 0)
            x = int(x.round())
            y = int(y.round())
            x_coords.append(x)
            y_coords.append(y)
            i=i+1
        # print('x', x_coords)
        # print('y', y_coords)
        # print('len of x + y ')
        # print(len(x_coords), len(y_coords))
        plt.scatter(x_coords, y_coords, s=4, alpha=.85, label=filenames[n])
        plt.plot(x_coords, y_coords, linewidth=2, alpha=.85)
        n=n+1

    map = plt.imshow(smoothed, cmap='gray', vmin=-.1, vmax=.1)
    leg = plt.legend(fontsize=6, loc=3, markerscale=5, bbox_to_anchor=(-0.15, -0.2 - .047*len(filenames)), fancybox=True)
    plt.title('RA DEC Tracks')
    #plt.colorbar(map, label=r'($K_{CMB}$)')
    if (directory == 'None'):
        return
    plt.savefig(directory + '/Tracks.png', dpi=600)
    return

###############################################################################################################

def main_track(orbitfile_folder, mapset, wcsset, xlfile, cut_coadd, directory):
    stacked_tuples = main3(orbitfile_folder, mapset, wcsset)
    dt, eps, x_he, y_he, z_he = helper_set_window(xlfile)
    filenames = [x[1] for x in stacked_tuples]
    radec_tracks = []
    for x in filenames:
        params = x[x.find('[')+1:x.find(']')].split(', ')
        params = [np.float64(i) for i in params]
        print("Parameters: ", params)
        radec_tracks.append( helper_get_coords(params, dt, eps, x_he, y_he, z_he) )
    plot_tracks(radec_tracks, cut_coadd, filenames, directory)
    #plot_all(stacked_tuples, directory)
    return







###############################################################################################################
def stack_cut_weights(cutouts_, weights_):
    n=1
    stacked = cutouts_[0]
    stacked_weights = weights_[0]
    length = len(cutouts_)
    while n<length:
        stacked = add(stacked, multiply(cutouts_[n], weights_[n]))
        stacked_weights = add(stacked_weights, weights_[n])
        n=n+1
    length = len(cutouts_)
    #stacked = [ x / length for x in stacked ]
    stacked = divide(stacked, stacked_weights)
    #print('after', '\n', stacked )
    return stacked

def cutouts_from_memory(orbits, mapset, weightset, wcsset):
    print('length of orbits: ', len(orbits))
    length_maps=len(mapset)
    cut_name_tup_list = []
    #print('orbits', orbits)
    for z in orbits:
        j = 0
        orbit_cutouts = []
        weight_cutouts = []
        #print('length', length_maps)
        while j < length_maps:
            x, y = wcsset[j].all_world2pix((z[0][j]), z[1][j], 0)
            x, y = int(x.round()), int(y.round())
            # x = int(x.round())
            # y = int(y.round())
            orbit_cutouts.append(mapset[j][y-24:y+26,x-24:x+26])
            weight_cutouts.append(weightset[j][y-24:y+26,x-24:x+26])
            j=j+1
        #print('o o while')
        cut_name_tup_list.append([orbit_cutouts, weight_cutouts, z[2]])
    return cut_name_tup_list

def main4(orbits, mapset, weightset, wcsset, directory):
    start = timeit.default_timer()
    stacked_tuples = []
    orbit_name_tuples = cutouts_from_memory(orbits, mapset, weightset, wcsset)
    for x in orbit_name_tuples:
        #print('x[0]', x[0])
        x[0] = stack_cut_weights(x[0], x[1])
        #sigmasa
        if ( to_signal(x[0]) > 6.2):
            stacked_tuples.append(x)
        continue
    stop = timeit.default_timer()
    plot_all(stacked_tuples, directory)
    print('stack+sort time: ', stop-start)
    print('good ones', len(stacked_tuples))
    return stacked_tuples

###############################################################################################################
#Generation = NamedTuple('Generation', 'ece_bot ece_top inc_bot inc_top lan_bot lan_top argp_bot argp_top tan_bot tan_top sma_bot sma_top itterations perfile dec_bot dec_top')

class Generation_():
    def __init__(self, ece_bot, ece_top, inc_bot, inc_top, lan_bot, lan_top, argp_bot, argp_top, tan_bot, tan_top, sma_bot, sma_top,
                 iterations, ra_bot, ra_top, dec_bot, dec_top):
        self.ece_bot, self.ece_top, self.inc_bot, self.inc_top, self.lan_bot, self.lan_top = ece_bot, ece_top, inc_bot, inc_top, lan_bot, lan_top
        self.argp_bot, self.argp_top, self.tan_bot, self.tan_top, self.sma_bot, self.sma_top = argp_bot, argp_top, tan_bot, tan_top, sma_bot, sma_top
        self.iterations, self.ra_bot, self.ra_top, self.dec_bot, self.dec_top = iterations, ra_bot, ra_top, dec_bot, dec_top

def everything(mapset, weightset, wcsset, xlfile, cut_coadd, directory, GenParams: Generation_):
    start = timeit.default_timer()
    ece_bot, ece_top, inc_bot, inc_top, lan_bot, lan_top = GenParams.ece_bot, GenParams.ece_top, GenParams.inc_bot, GenParams.inc_top, GenParams.lan_bot, GenParams.lan_top
    argp_bot, argp_top, tan_bot, tan_top, sma_bot, sma_top  = GenParams.argp_bot, GenParams.argp_top, GenParams.tan_bot, GenParams.tan_top, GenParams.sma_bot,GenParams.sma_top
    dt, eps, x_he, y_he, z_he = helper_set_window(xlfile)
    it = GenParams.iterations
    #orbits = [[ra, dec, params], ...]
    stacked_tuples = []
    for i in range(it):
        orbits = []
        parameters = helper_generates_params(ece_bot, ece_top, inc_bot, inc_top, lan_bot, lan_top, argp_bot, argp_top, tan_bot, tan_top, sma_bot, sma_top)
        #parameters = [0.271650183, 25.20215372, 55.6256526, 251.1874401, 348.6395016, 3.131789167]
        orbit= helper_generate_coords(parameters, dt, eps, x_he, y_he, z_he, GenParams.ra_bot, GenParams.ra_top, GenParams.dec_bot, GenParams.dec_top)
        if (orbit == None):
            continue
        orbits.append(orbit)
        print('got one')
        print('i', i, orbit[2])
        par_low, par_up = lowup(orbit[2])
        for j in range(50000):
            #params = helper_generates_params2(par_low[0], par_up[0])
            orbit_= helper_generate_coords(helper_generates_params2(par_low[0], par_up[0]), dt, eps, x_he, y_he, z_he, GenParams.ra_bot, GenParams.ra_top, GenParams.dec_bot, GenParams.dec_top)
            if (orbit_ == None):
                continue
            orbits.append(orbit_)
        good_orbits = main4(orbits, mapset, weightset, wcsset, directory)
        stacked_tuples[0:0] = good_orbits
        more_stacked_tuples = []
        for z in good_orbits:
            print('success loop')
            close_orbits = []
            par_low, par_up = lowup2(z[2])
            for j in range(50000):
                params = helper_generates_params2(par_low[0], par_up[0])
                orbit_= helper_generate_coords(params, dt, eps, x_he, y_he, z_he, GenParams.ra_bot, GenParams.ra_top, GenParams.dec_bot, GenParams.dec_top)
                if (orbit_ == None):
                    continue
                close_orbits.append(orbit_)
            more_stacked_tuples[0:0] = main4(close_orbits, mapset, weightset, wcsset, directory)
        stacked_tuples[0:0] = more_stacked_tuples
    print('done w orbits')
    #stacked_tuples = main4(orbits, mapset, wcsset)
    orbit_params = [x[2] for x in stacked_tuples]
    radec_tracks = []
    for x in orbit_params:
        radec_tracks.append( helper_get_coords(x, dt, eps, x_he, y_he, z_he) )
    plot_tracks(radec_tracks, cut_coadd, orbit_params, directory)
    #print('stacked tuples:', '\n', stacked_tuples)
    #plot_all(stacked_tuples, directory)
    stop = timeit.default_timer()
    print('everything:', stop-start)
    return

def helper_generates_params(ece_bot, ece_top, inc_bot, inc_top, lan_bot, lan_top, argp_bot, argp_top, tan_bot, tan_top, sma_bot, sma_top):
    orbit_paramter_list = []
    ece =  random.uniform(ece_bot, ece_top)
    orbit_paramter_list.append(ece)
    
    inc = random.uniform(inc_bot, inc_top) * pi/180
    orbit_paramter_list.append(inc)
    
    # y = []
    
    # while not y:
    #     OM = random.uniform(lan_bot, lan_top)
    #     if OM< 100 or OM> 300:
    #         y.append(OM*pi/180) 
    OM = random.uniform(lan_bot, lan_top)
    orbit_paramter_list.append(OM*pi/180)
    
    w = random.uniform(argp_bot, argp_top) * pi/180
    orbit_paramter_list.append(w)
    
    TA = random.uniform(tan_bot, tan_top) * pi/180
    orbit_paramter_list.append(TA)
    
    # SA = random.uniform(sma_bot, sma_top)
    # orbit_paramter_list.append(SA)

    SA = loguniform.rvs(sma_bot, sma_top, size=1)
    orbit_paramter_list.append(SA[0])
    return orbit_paramter_list

def helper_set_window(xlfile):
    # # x_he = []
    # # y_he = []
    # # z_he = []
    ast = pd.read_excel(xlfile)
    year, month, day = ast['Year'], ast['Month'], ast['Day']
    hour, minute, sec = ast['Hour'], ast['Minute'], ast['Sec']
    dt = []
    for time in range(year.shape[0]):
    # datetime(year, month, day, hour, minute, second)
    #Start
        a = datetime(year[0], month[0], day[0], hour[0], minute[0], sec[0])
    #End
        b = datetime(year[time], month[time], day[time], hour[time], minute[time], sec[time])
    # returns a timedelta object
        c = b-a 
        dt.append(c.total_seconds() / 60)

    #Need 8 orbital parameters  for both Mars and Earth

    # # e,a,i = ast['e'][0], ast['a'][0], ast['i'][0]

    # # Omega, omega, Theta = ast['Omega'][0], ast['omega'][0], ast['Theta'][0]

    eps = ast['eps'][0] * np.pi/180
    x_he = ast['X']
    y_he = ast['Y']
    z_he = ast['Z']
    #e, i, OM, w, TA, a
    #0, 1, 2,  3, 4,  5

    # # parameters_earth = [e, i*np.pi/180, Omega*np.pi/180, omega*np.pi/180, Theta*np.pi/180, a]

    # # E_0 = 2*np.arctan( ( (1-parameters_earth[0])/(1+parameters_earth[0]) )**0.5 * np.tan(parameters_earth[4]/2) ) #rads

    # # M_0 = np.float32(E_0 - parameters_earth[0]*np.sin(E_0))  #rads

    # # n = np.float32(60*np.sqrt(const.GM_sun/(const.au**3)) * 1/(parameters_earth[5]**(3/2))) #rad per day, a in AU

    # # for num in dt:

    # #     M_e = M_0 + n*num #rads

    # #     E_i = 0
    # #     if M_e < np.pi:
    # #         E_i = M_e + parameters_earth[0]/2
    # #     else:
    # #         E_i = M_e - parameters_earth[0]/2
        
    # #     def f(E_1):
    # #         return E_1-parameters_earth[0]*np.sin(E_1) - M_e
        
    # #     E_a = optimize.bisect(f, E_i-2, E_i + 2)
    # #     # E_a = optimize.newton(f, E_i, tol = 1*10**(-4))
        
    # #     Theta_e = (2*np.arctan( np.sqrt( (1+parameters_earth[0])/(1-parameters_earth[0]) ) * np.tan(E_a/2) ))
            
    # #     r_e = (parameters_earth[5]*(1-parameters_earth[0]**2))/(1 + parameters_earth[0] * np.cos(Theta_e))
        
    # #     x_he.append(r_e * ( np.cos(parameters_earth[2]) * np.cos((parameters_earth[3] + Theta_e)) - np.sin(parameters_earth[2])*np.cos(parameters_earth[1])*np.sin(parameters_earth[3]+Theta_e)))
        
    # #     y_he.append(r_e * ( np.sin(parameters_earth[2]) * np.cos((parameters_earth[3] +Theta_e)) + np.cos(parameters_earth[2])*np.cos(parameters_earth[1])*np.sin((parameters_earth[3]+Theta_e))))
        
    # #     z_he.append(r_e * np.sin(parameters_earth[1]) * np.sin((parameters_earth[3] + Theta_e)) )
    return dt, eps, x_he, y_he, z_he

def helper_generate_coords(parameters, dt, eps, x_he, y_he, z_he, ra_bot, ra_top, dec_bot, dec_top):
    # ecc, inc, lan, argp, tan, sma = parameters
    # parameters[1] = inc*np.pi/180
    # parameters[2] = lan*np.pi/180
    # parameters[3] = argp*np.pi/180
    # parameters[4] = tan*np.pi/180
    E_2 = 2*np.arctan( ( (1-parameters[0])/(1+parameters[0]) )**0.5 * np.tan(parameters[4]/2) ) #rads
    
    M_1 = float32(E_2 - parameters[0]*np.sin(E_2))  #rads
    
    n_1 = float32(60*np.sqrt(const.GM_sun/(const.au**3)) * 1/(parameters[5]**(3/2))) #rad per day, a in AU
    
    BETA = []
    alpha_d = []
    
    
    for index, num in enumerate(dt):
    
        M_m = M_1 + n_1*num #rads
        
        E_t = 0
        if M_m < np.pi:
            E_t= M_m + parameters[0]/2
        else:
            E_t = M_m - parameters[0]/2
                
        def y(E_3):
            return E_3-parameters[0]*np.sin(E_3 ) - M_m
        
        E_m = optimize.bisect(y, E_t - 2, E_t + 2)
        
        Theta_m = (2*np.arctan( np.sqrt( (1+parameters[0])/(1-parameters[0]) ) * np.tan(E_m/2) ))
        

        r_m = (parameters[5]*(1-parameters[0]**2))/(1 + parameters[0] * np.cos(Theta_m))
        
        x_hm = (r_m * ( np.cos(parameters[2] ) * np.cos(parameters[3] +Theta_m) - np.sin(parameters[2] )*np.cos(parameters[1] )*np.sin(parameters[3] +Theta_m)) )
        
        y_hm = (r_m * ( np.sin(parameters[2] ) * np.cos(parameters[3] +Theta_m)  + np.cos(parameters[2])*np.cos(parameters[1])*np.sin(parameters[3]+Theta_m)) )
        
        z_hm = (r_m * np.sin(parameters[1]) * np.sin(parameters[3] + Theta_m) )
              

        x_ge = x_hm - x_he[index]
    
        y_ge = y_hm - y_he[index]
    
        z_ge = z_hm - z_he[index]
        
        x_ra= x_ge
        
        y_ra = y_ge *np.cos(eps) - z_ge*np.sin(eps)
        
        z_ra = y_ge*np.sin(eps) + z_ge*np.cos(eps)
        
        
        #RA
        alpha = np.arctan2(y_ra, x_ra)*180/np.pi
        while alpha <0:
            alpha += 360
        
        while alpha>=360:
            alpha -= 360
        if alpha < ra_bot or alpha > ra_top:
                return
        else:
            alpha_d.append(alpha)
            #Ra_sin.append(np.sin(np.arctan2(y_ra, x_ra)))
        # ALPHA.append(alpha)
    
        #DEC
        beta = (np.arctan2(z_ra, np.sqrt(x_ra**2 + y_ra**2))*180/np.pi)
        if beta < dec_bot or beta > dec_top:
            return
        else:
            BETA.append(beta)
            #Dec_sin.append(np.sin(np.arctan2(z_ra, np.sqrt(x_ra**2 + y_ra**2) )))
    #if all(element > ra_bot and element < ra_top for element in alpha_d) and all(kuro > dec_bot and kuro < dec_top for kuro in BETA):
        # print('OOOHHH')
    # Ra_list.append(alpha_d)
    # Dec_list.append(BETA)
    # Ra_list_sin.append(Ra_sin)
    # Dec_list_sin.append(Dec_sin)
    par = [parameters[0], parameters[1]*180/pi, parameters[2]*180/pi,parameters[3]*180/pi, parameters[4]*180/pi, parameters[5]]
    # good_orbits.append(par)
    # parameters_list.append(par)
    #print ([alpha_d, BETA])
    return [alpha_d, BETA, par]

def helper_get_coords(parameters, dt, eps, x_he, y_he, z_he):      #list w 6 params
    ecc, inc, lan, argp, tan, sma = parameters
    parameters[1] = inc*np.pi/180
    parameters[2] = lan*np.pi/180
    parameters[3] = argp*np.pi/180
    parameters[4] = tan*np.pi/180
    E_2 = 2*np.arctan( ( (1-parameters[0])/(1+parameters[0]) )**0.5 * np.tan(parameters[4]/2) ) #rads
    
    M_1 = np.float32(E_2 - parameters[0]*np.sin(E_2))  #rads
    
    n_1 = np.float32(60*np.sqrt(const.GM_sun/(const.au**3)) * 1/(parameters[5]**(3/2))) #rad per day, a in AU
    
    BETA = []
    alpha_d = []
    
    
    for index, num in enumerate(dt):
    
        M_m = M_1 + n_1*num #rads
        
        E_t = 0
        if M_m < np.pi:
            E_t= M_m + parameters[0]/2
        else:
            E_t = M_m - parameters[0]/2
                
        def y(E_3):
            return E_3-parameters[0]*np.sin(E_3 ) - M_m
        
        E_m = optimize.bisect(y, E_t - 2, E_t + 2)
        
        Theta_m = (2*np.arctan( np.sqrt( (1+parameters[0])/(1-parameters[0]) ) * np.tan(E_m/2) ))
        

        r_m = (parameters[5]*(1-parameters[0]**2))/(1 + parameters[0] * np.cos(Theta_m))
        
        x_hm = (r_m * ( np.cos(parameters[2] ) * np.cos(parameters[3] +Theta_m) - np.sin(parameters[2] )*np.cos(parameters[1] )*np.sin(parameters[3] +Theta_m)) )
        
        y_hm = (r_m * ( np.sin(parameters[2] ) * np.cos(parameters[3] +Theta_m)  + np.cos(parameters[2])*np.cos(parameters[1])*np.sin(parameters[3]+Theta_m)) )
        
        z_hm = (r_m * np.sin(parameters[1]) * np.sin(parameters[3] + Theta_m) )
              

        x_ge = x_hm - x_he[index]
    
        y_ge = y_hm - y_he[index]
    
        z_ge = z_hm - z_he[index]
        
        x_ra= x_ge
        
        y_ra = y_ge *np.cos(eps) - z_ge*np.sin(eps)
        
        z_ra = y_ge*np.sin(eps) + z_ge*np.cos(eps)
        
        
        #RA
        alpha = np.arctan2(y_ra, x_ra)*180/np.pi
        
        while alpha <0:
            alpha += 360
        
        while alpha>=360:
            alpha -= 360
        
        alpha_d.append(alpha)
        # ALPHA.append(alpha)
    
        #DEC
        BETA.append(np.arctan2(z_ra, np.sqrt(x_ra**2 + y_ra**2))*180/np.pi)
    return list(zip(alpha_d, BETA))

def lowup(param_s):
    e_v, i_v, o_v, w_v, ta_v, sa_v = 0.05, 3, 3, 3, 3, 0.05
    par_low = []
    par_up = []
    par_low.append([abs(param_s[0] - e_v), abs(param_s[1] - i_v), abs(param_s[2] - o_v), abs(param_s[3] - w_v), abs(param_s[4] - ta_v), abs(param_s[5] - sa_v)])
    par_up.append([param_s[0] + e_v, param_s[1] + i_v, param_s[2] + o_v, param_s[3] + w_v, param_s[4] + ta_v, param_s[5] + sa_v])
    return [par_low, par_up]

def lowup2(param_s):
    e_v, i_v, o_v, w_v, ta_v, sa_v = 0.05, 1, 1, 1, 1, 0.05
    par_low = []
    par_up = []
    par_low.append([abs(param_s[0] - e_v), abs(param_s[1] - i_v), abs(param_s[2] - o_v), abs(param_s[3] - w_v), abs(param_s[4] - ta_v), abs(param_s[5] - sa_v)])
    par_up.append([param_s[0] + e_v, param_s[1] + i_v, param_s[2] + o_v, param_s[3] + w_v, param_s[4] + ta_v, param_s[5] + sa_v])
    return [par_low, par_up]

def helper_generates_params2(par_low, par_up):
    
    # orbit_paramter_list = []
    
    # ece =  np.random.uniform(par_low[0], par_up[0])
    # orbit_paramter_list.append(ece)
    
    # inc = np.random.uniform(par_low[1], par_up[1]) * np.pi/180
    # orbit_paramter_list.append(inc)
    
    # OM = np.random.uniform(par_low[2], par_up[2]) * np.pi/180
    # orbit_paramter_list.append(OM)
        
    # w = np.random.uniform(par_low[3], par_up[3]) * np.pi/180
    # orbit_paramter_list.append(w)
    
    # TA = np.random.uniform(par_low[4], par_up[4]) * np.pi/180
    # orbit_paramter_list.append(TA)
    
    # SA = np.random.uniform(par_low[5], par_up[5])
    # orbit_paramter_list.append(SA)
    return [random.uniform(par_low[0], par_up[0]), random.uniform(par_low[1], par_up[1]) * pi/180, random.uniform(par_low[2], par_up[2]) * pi/180, random.uniform(par_low[3], par_up[3]) * pi/180, random.uniform(par_low[4], par_up[4]) * pi/180, loguniform.rvs(par_low[5], par_up[5], size=1)[0]]

# %%
# %%
