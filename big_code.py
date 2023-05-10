# -*- coding: utf-8 -*-
# ..............................................................................
 #   Name        : cdi_prdmanager.py
 #   Application :
 #   Author      : Drought IT team
 #   Created     : 2022-11-02
 #   Purpose     : Manage the CDI procedure
 #
# ..............................................................................

# ..............................................................................
# IMPORTS
# ..............................................................................
import argparse
import calendar
import sys
import logging.config
import warnings
import itertools
import os
import xarray as xr
import numpy as np
import json
import datetime
import rasterio
logging.getLogger('rasterio').setLevel(logging.WARNING)

from datetime import date
from datetime import timedelta

import db_in_ex as dbin
import cdi_calculations as calc
# import cdi_preprocessing as prep


# ..............................................................................
# CLASSES
# ..............................................................................
# To be taken from droop, time_management mode
class Dekad:
    
    def __init__(self, year, month, day):

        self._year = year
        self._month = month
        self._day = day

    @property
    def start(self):
        return self._get_start_date_from_params(self._year, self._month,
                                                self._day)

    @property
    def end(self):
        return self._get_end_date_from_start_date(self.start)

    @property
    def year(self):
        return self.start.year

    @property
    def month(self):
        return self.start.month

    @property
    def first_day(self):
        return self.start.day

    @property
    def day(self):
        return self.start.day

    @property
    def last_day(self):
        return self.end.day

    @property
    def number(self):
        if self.start.day == 1:
            return self.start.month + 2 * (self.start.month - 1)
        if self.start.day == 11:
            return (self.start.month * 2) + (self.start.month - 1)
        if self.start.day == 21:
            return self.start.month * 3

    def _get_dekad_number(self, start_datetime):
        if start_datetime.day == 1:
            return start_datetime.month + 2 * (start_datetime.month - 1)
        if start_datetime.day == 11:
            return (start_datetime.month * 2) + (start_datetime.month - 1)
        if start_datetime.day == 21:
            return start_datetime.month * 3

    def _get_start_date_from_params(self, year, month, day):
        try:
            input_date = datetime.datetime(year, month, day)
            if input_date.day < 11:
                start_day = 1
            elif input_date.day < 21:
                start_day = 11
            else:
                start_day = 21

            return datetime.datetime(year, month, start_day)

        except:
            raise Exception('Invalid inputs.')

    def _get_end_date_from_start_date(self, start_datetime):

        if start_datetime.day == 21:
            last_day = self._get_last_day_of_month(start_datetime)
        elif start_datetime.day == 11:
            last_day = 20
        elif start_datetime.day == 1:
            last_day = 10
        else:
            raise Exception('Dekad start day should be 1, 11, or 21')

        return datetime.datetime(start_datetime.year, start_datetime.month, last_day)

    def _get_last_day_of_month(self, dt):
        return calendar.monthrange(dt.year, dt.month)[1]

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __neq__(self, other):
        return self.start != other.start or self.end != other.end

    def __add__(self, other):

        if type(other) == int:
            if other < 0:
                return self - abs(other)

            elif other > 1:
                for i in range(other - 1):
                    self = self + 1

            new = self.end + datetime.timedelta(days=1)
            return Dekad(new.year, new.month, new.day)

        if type(other) == Dekad:
            dates_list = [self.start + timedelta(i) for i in range(int((other.start -
                                                                        self.start).days) + 1)]
            dekads_list = []
            for day_date in dates_list:
                dekad_date = Dekad(day_date.year, day_date.month,
                                   day_date.day)
                if dekad_date not in dekads_list:
                    dekads_list.append(dekad_date)
            return dekads_list

    def __sub__(self, num_dekads):

        if num_dekads < 0:
            return self + abs(num_dekads)

        elif num_dekads > 1:
            for i in range(num_dekads - 1):
                self = self - 1

        new = self.start - datetime.timedelta(days=1)
        return Dekad(new.year, new.month, new.day)

    def __repr__(self):
        return '<Dekad {0:04d}-{1:02d}-{2:02d}>'.format(self.start.year, self.start.month, self.start.day)

    def __hash__(self):
        return hash(self.__repr__())

class bothfilesPresentError(Exception):
    """Exception raised for errors.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="img and nc files are both present"):
        self.message = message
        super().__init__(self.message)

class DekadNotFoundException(Exception):
    """Exception raised for errors.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="The run dekad was not found on the count files"):
        self.message = message
        super().__init__(self.message)

# ..............................................................................
# FUNCTIONS
# ..............................................................................
def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_type",
                        help="Type of run. Options: automatic a, manual m ",
                        dest="run_type",
                        required=True, type=str, default='a', choices=['a', 'm'])
    parser.add_argument("-d", "--date",
                        help="Calculation date, format: YYYY-MM-DD",
                        dest="date", required=False, type=str,
                        default=None)
    parser.add_argument("-i", "--start_date",
                        help="Start date, format: YYYY-MM-DD",
                        dest="start_date",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("-e", "--end_date",
                        help="End date, format: YYYY-MM-DD", dest="end_date",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("--update-count",
                        help="option for update count files", dest="update_count",
                        required=False,
                        action="store_true")
    parser.add_argument("--database-inputs",
                        help="option to get inputs from the database", dest="database-inputs",
                        required=False,
                        action="store_true")
    parser.add_argument("--database-import",
                        help="option to import results in the database", dest="database-import",
                        required=False,
                        action="store_true")
    parser.add_argument("--parallelize",
                        help="option to parallelize preprocessing using Process library", dest="parallelize",
                        required=False,
                        action="store_true")
    parser.add_argument("-v",
                        help="CDI algoritmn version", dest="version",
                        required=False,
                        type=str,
                        default="2.3.0")
    parsed_args = parser.parse_args(args)
    return vars(parsed_args)

# To be taken from droop, utilities
def replace_all(text, dic):
    for i, j in iter(dic.items()):
        text = text.replace(i, j)
    return text

def render_filename(filename, dekad):
    replacements = {'[YYYY]': '{}'.format(dekad.year), '[DD]': '{:02d}'.format(dekad.day),'[NN]': '{:02d}'.format(dekad.number), '[MM]': '{:02d}'.format(dekad.month)}
    filename_rendered = replace_all(filename, replacements)
    return filename_rendered

# To be taken from droop, netcdf_handling mode
def format_string_date(date):
    formated_date = datetime.datetime(int(date.split('-')[0]),
                                    int(date.split('-')[1]),
                                    int(date.split('-')[2]))
    return formated_date


# To be taken from droop, netcdf_handling mode
def export_dataset(dataset, fillvalue, variables_list, variables_units, espg, 
                   output_file_path, output_file_name,
                   variables_attributes, general_attributes,
                   compression):
    """
    Exports a xarray dataset into a NetCDF file using a configuration file.
    """
    # setting coordinate reference system parameters
    dataset[espg] = -2147483647.0  # dummy variable to set the crs
    dataset[espg].attrs = variables_attributes['projections_attributes'][espg]
    try:
        dataset.time.attrs = variables_attributes['coordinates_attributes']['time_attributes']
    except:
        pass
    dataset.lon.attrs = variables_attributes['coordinates_attributes']['lon_attributes']
    dataset.lat.attrs = variables_attributes['coordinates_attributes']['lat_attributes']
    # setting global attributes
    dataset.attrs = variables_attributes['global_attributes']
    dataset.attrs['date_created'] = str(date.today())
    # adding variable metadata
    if general_attributes:
        for attribute in general_attributes:
            try:
                dataset.attrs[attribute] = general_attributes[attribute]
            except:
                pass
    # setting variables attributes
    for variable, units in itertools.zip_longest(variables_list, variables_units):
        dataset[variable].attrs = variables_attributes['variable_attributes']
        dataset[variable].attrs['units'] = units
        if fillvalue:
            dataset[variable].attrs['_FillValue'] = fillvalue
    # compressing the file before saving
    if compression:
        """setting compression parameters"""
        comp = variables_attributes['compression']['comp']
        encoding = {var: comp for var in dataset.data_vars}
        dataset.to_netcdf(output_file_path + '/' + output_file_name, mode='w', 
                          format='NETCDF4', encoding=encoding)
    else:
        """Save the new netcdf file"""
        dataset.to_netcdf(os.path.join(output_file_path, output_file_name), 
                          mode='w', 
                          format='NETCDF4')
    return 0


# To be taken from droop, netcdf_handling mode
def create_1tdataset_from_array(array, time, lat, lon, variable_name):
    """Creates an xarray dataset of one variable and one time
     
    Args
    ----------
    arrays_list
        List of arrays. The order of the array should correspond to the order of
        the variables_list.
        e.g [[list of arrays variable 1], [list of arrays variable 2], ...[list of arrays variable n]]
        e.g list of arrays variable 1 = [array1(t1, var1), array2(t2, var1), ..., arrayn(tn, var1)]
        type: list of list of numpy 2d arrays
    times
        list of dates in np.datetime64
        type : list
    lat
        list of latitude coordinates in number format
        type : list
    lon
        list of longitude coordinates in number format
        type : list
    variable_name
        Nme of the variable for the new dataset. 
        type : string
       
    Returns
    ----------
    xarray dataset
    
    Notes
    ----------

    """
    dataset = xr.Dataset({variable_name: (['time', 'lat', 'lon'], array)}, 
                          coords={'lat': (['lat'], lat), 'lon': (['lon'], lon), 
                                  'time': time})
    return dataset


# To be taken from droop, netcdf_handling mode
def create_1tlaea_dataset(array, date, variable_name):
    lon = np.arange(2502500, 7502500, 5000)
    lat = np.arange(752500, 5502500, 5000)[::-1]
    return create_1tdataset_from_array(array,date,lat,lon,variable_name)


# To be taken from droop, netcdf_handling mode
def save_laea_arrays(variable_name, arrays, dates, output_file_path):
    lon = np.arange(2502500, 7502500, 5000)
    lat = np.arange(752500, 5502500, 5000)[::-1]
    dataset = xr.Dataset({variable_name: (['time', 'lat', 'lon'], arrays)},
                          coords={'lat': (['lat'], lat), 'lon': (['lon'], lon),
                                  'time': (['time'], dates)})
    with open('netcdf_config.json') as netcdf_config_file:
        data = json.load(netcdf_config_file)
    file_name = "{}.nc".format(variable_name)
    variable_attributes = data['3035']
    export_dataset(dataset=dataset, fillvalue=-9999,
                   variables_list=[variable_name],
                   variables_units=['undimensional'], espg='3035',
                   output_file_path=output_file_path, output_file_name=file_name,
                   general_attributes='',
                   variables_attributes=variable_attributes,
                   compression=True)


def save_results(variable_name, array, date, output_file_path):
    dataset = create_1tlaea_dataset(array, [date], variable_name)

    with open('netcdf_config.json') as netcdf_config_file:
        data = json.load(netcdf_config_file)
    file_name = "{}_m_euu_{}{:02d}{:02d}_t.nc".format(variable_name, date.year, date.month, date.day)
    variable_attributes = data['3035']
    export_dataset(dataset=dataset, fillvalue=-9999,
                   variables_list=[variable_name],
                   variables_units=['undimensional'], espg='3035',
                   output_file_path=output_file_path, output_file_name=file_name,
                   general_attributes='',
                   variables_attributes=variable_attributes,
                   compression=True)


def get_required_cdi_files_to_calculate_dekad(dekad, directory_path, loggers):
    file_list = os.listdir(directory_path)
    try:
        dekad_p = dekad - 1
        cdi_p_fp = get_cdi_file_name_for_dekad(dekad_p, file_list)
        cdi_p = rasterio.open(f'{directory_path}/{cdi_p_fp}').read()
        #loggers['log'].debug(f'cdi_p input: \n{directory_path}/{cdi_p_fp}')
    except FileNotFoundError as e:
        # This applies at the beggining of the run
        loggers['log'].info(f"No previuos CDI for calculations. Setting all to zero: {e} ")
        cdi_p = np.zeros((1, 950, 1000)).astype('int16')
    
    try:
        dekad_2p = dekad - 2
        cdi_2p_fp = get_cdi_file_name_for_dekad(dekad_2p, file_list)
        cdi_2p = rasterio.open(f'{directory_path}/{cdi_2p_fp}').read()
        #loggers['log'].debug(f'cdi_2p input: \n{directory_path}/{cdi_2p_fp}')
    except FileNotFoundError as e:
        # This applies at the beggining of the run
        loggers['log'].info(f"No previuos CDI for calculations. Setting all to zero: {e} ")
        cdi_2p = np.zeros((1, 950, 1000)).astype('int16')
    
    try:
        dekad_3p = dekad - 3
        cdi_3p_fp = get_cdi_file_name_for_dekad(dekad_3p, file_list)
        cdi_3p = rasterio.open(f'{directory_path}/{cdi_3p_fp}').read()
        #loggers['log'].debug(f'cdi_3p input: \n{directory_path}/{cdi_3p_fp}')
    except FileNotFoundError as e:
        # This applies at the beggining of the run
        loggers['log'].info(f"No previuos CDI for calculations. Setting all to zero: {e} ")
        cdi_3p = np.zeros((1, 950, 1000)).astype('int16')

    return cdi_p.astype('int16'), cdi_2p.astype('int16'), cdi_3p.astype('int16')


def get_cdi_file_name_for_dekad(dekad, file_list):

    nc_file_name = get_nc_file_name_for_dekad(dekad)
    img_file_name = get_img_file_name_for_dekad(dekad)
    cdi_file_names = {'img_file_name': None, 'nc_file_name': None}
    for file_name in file_list:
        if file_name == img_file_name:
            cdi_file_names['img_file_name'] = file_name
        elif file_name == nc_file_name:
            cdi_file_names['nc_file_name'] = file_name
    
    if cdi_file_names['img_file_name'] is not None and cdi_file_names['nc_file_name'] is not None:
        raise bothfilesPresentError()
    # The img file has the precedence over the nc
    if cdi_file_names['img_file_name'] is not None:
        return cdi_file_names['img_file_name']
    elif cdi_file_names['nc_file_name'] is not None:
        return cdi_file_names['nc_file_name']
    
    raise FileNotFoundError(f'No cdi file found for dekad {dekad}')


def get_nc_file_name_for_dekad(dekad):
    return f'cdinx_m_euu_{dekad.year}{dekad.month:02d}{dekad.day:02d}_t.nc'


def get_img_file_name_for_dekad(dekad):
    return f'CDI_v2_{dekad.year}_{dekad.number:02d}.img'


def get_inputs_from_db(current_dekad, loggers):
    # Gathering inputs
    inputs['spi1'] = spi1
    inputs['spi3'] = spi3
    inputs['sma'] = sma
    inputs['fapar'] = fapar
    inputs['cdi_p'] = cdi_p
    inputs['cdi_2p'] = cdi_2p
    inputs['cdi_3p'] = cdi_3p
    loggers['log'].debug(f'inputs: \n{spi1_fp},\n{spi3_fp},\n{sma_fp},\n{fapar_fp}')
    return inputs


def get_smian(dekad, inp_variables, loggers): 
    smian_in_files_ls = []    
    for smian_file in os.listdir(inp_variables['sma']['wd']):
        if f'_{dekad.year}' in smian_file and smian_file.endswith('.nc'):
            smian_in_files_ls.append(os.path.join(inp_variables['sma']['wd'], smian_file))
    latest_file = max(smian_in_files_ls, key=os.path.getctime)
    loggers['log'].debug(f'smian latest_file: {latest_file}')
    dataset = xr.open_dataset(latest_file, decode_cf=True, decode_times=True, decode_coords=True)
    dataset_sel = dataset.sel(time=str(dekad.start))
    sma = dataset_sel['smian'].values
    sma = sma[np.newaxis, :, :]
    return sma


def apply_phenology_mask(fapar, dekad, inp_variables):
    # Phenology mask ------------------> to be deprecated after preprocesing !!
    phmask = get_input(dekad, inp_variables, "phenology_mask")
    # If the crop is active (=1), or any type of vegetation (=np.nan) keep the fapar value, otherwise (=0) set the value to 9999
    fapar = np.where(phmask == 0, 9999.0, fapar)
    return fapar


def get_input(dekad, inp_variables, input_variable_name):
    variable_filename = render_filename(inp_variables[input_variable_name]['fn'], dekad)
    variable_filepath = os.path.join(inp_variables[input_variable_name]['wd'], variable_filename)
    if os.path.exists(variable_filepath):
        variable_array = rasterio.open(variable_filepath).read()
    else:
        try:
            if input_variable_name == 'fapar':
                variable_array = prep.fapar_preprocesser.run(dekad)
            elif input_variable_name == 'spi1':
                spi_pp = prep.spi_preprocesser.run(dekad, variable_name=input_variable_name)
                variable_array = spi_pp['01']
            elif input_variable_name == 'spi3':
                spi_pp = prep.spi_preprocesser.run(dekad, variable_name=input_variable_name)
                variable_array = spi_pp['03']
        except Exception as e:
            loggers['log'].info(f'There is a problem with the preprocessing. The CDI cannot be computed. See {e}')
            sys.exit()
    return variable_array


def masking_cdi(inputs, cdi):
    """"""
    # Move working directory from Python installation to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # loading configuration file
    with open('cdi_config.json') as json_data_file:
        data = json.load(json_data_file)
    lisflood_mask_filepath = data['lisflood_mask_filepath']

    # Lisflood_mask for land an sea
    lisflood_mask = xr.open_rasterio(lisflood_mask_filepath).values
    lisflood_domain = np.where(lisflood_mask == 1, 1, np.nan) # to make sure the mask works well using np.nan                                
    # Areas with missing data in spi and sma
    mask_nodata = np.logical_or.reduce((np.isnan(inputs['spi1']), 
                                        np.isnan(inputs['spi3']), 
                                        np.isnan(inputs['sma'])))

    # Masking    
    cdi[mask_nodata] = 7
    cdi = lisflood_domain * cdi
    cdi = np.where(np.isnan(lisflood_domain), 8, cdi)
    return cdi


def cdi_run(inputs, formated_date, count_sma_recovery, count_fapar_recovery, cdi_version, loggers):
    # Arguments for the cdi_calculation function
    args = {}
    args ['date'] = formated_date
    args ['count_sma_recovery'] = count_sma_recovery 
    args ['count_fapar_recovery'] = count_fapar_recovery
    args ['loggers'] = loggers
    kwargs = {**inputs, **args}
    Cases = None
    # -----------------------------------------------------------
    # Run the CDI
    if cdi_version == '3.0.0':
        cdi, cdi_checks, count_sma_recovery, count_fapar_recovery, cases, zspi, cdi_p_masked = calc.cdi_calculation_v300(**kwargs)
    if cdi_version == '2.3.0':
        cdi, cdi_checks, count_sma_recovery, count_fapar_recovery, cases = calc.cdi_calculation_v230(**kwargs)
        zspi, cdi_p_masked = None
    if cdi_version == '2.2.0':
        cdi, cdi_checks, count_sma_recovery, count_fapar_recovery = calc.cdi_calculation_v220(**kwargs)
        cases = cdi * False # for version 2.2.0, the cases are not calculated
        zspi, cdi_p_masked = None
    # -----------------------------------------------------------
    # Copying the result to collect it later in the loop
    count_sma_recovery_p_dekad_temp = np.copy(count_sma_recovery)
    count_fapar_recovery_p_dekad_temp = np.copy(count_fapar_recovery)

    return cdi, cases, count_sma_recovery_p_dekad_temp, count_fapar_recovery_p_dekad_temp, zspi, cdi_p_masked


def find_dekad(current_dekad, count_sma_recovery, loggers):
    previous_dekad = current_dekad - 1
    dekad_found = False
    max_count = 5
    current_count = 0
    while dekad_found is False:
        try:
            previous_dekad_str = previous_dekad.start.strftime("%Y-%m-%d") 
            count_sma_recovery.sel(time = previous_dekad_str)['count_sma_recovery']
        except KeyError:
            previous_dekad = previous_dekad - 1
            current_count += 1
            if current_count == max_count:
                loggers['log'].info(f'Maximum number of tries reached: {max_count} to search for a previous dekad in the count files')
                break
            continue
        dekad_found = True
        
        break
    
    if dekad_found == False:
        raise DekadNotFoundException
    
    return previous_dekad


def run_cdi_with_count(dekads, cdi_start, count_sma_recovery, count_fapar_recovery, cdi_version, inp_variables, out_path, inputs_type, loggers):
    ## Selecting the count arrays of the previous dekad of start_dekad
    # define the dekad to run
    start_dekad = dekads[0]
    if start_dekad.start == cdi_start:
        # This case is when the file doesn't exists because
        # it is the CDI start date
        count_sma_recovery_p_dekad = np.zeros((1, 950, 1000)).astype('int16')
        count_fapar_recovery_p_dekad = np.zeros((1, 950, 1000)).astype('int16')
    else:
        try:
            # Selecting the count arrays of the previous dekad   
            previous_dekad = start_dekad - 1
            # Implemented but not used _find_dekad_ function
            #previous_dekad = find_dekad(start_dekad, count_sma_recovery, loggers)
            previous_dekad_str = previous_dekad.start.strftime("%Y-%m-%d") 
            count_sma_recovery_p_dekad = count_sma_recovery.sel(time = previous_dekad_str)['count_sma_recovery'].values
            count_fapar_recovery_p_dekad = count_fapar_recovery.sel(time = previous_dekad_str)['count_fapar_recovery'].values
            # Adding a new axis to the count arrays (1, 950, 1000)
            count_sma_recovery_p_dekad = count_sma_recovery_p_dekad[np.newaxis, :, :].astype('int16')
            count_fapar_recovery_p_dekad = count_fapar_recovery_p_dekad[np.newaxis, :, :].astype('int16')
        except:
            loggers['log'].info(f'Previuos dekad {previous_dekad} not present in the count files. Setting them to zero')
            count_sma_recovery_p_dekad = np.zeros((1, 950, 1000)).astype('int16')
            count_fapar_recovery_p_dekad = np.zeros((1, 950, 1000)).astype('int16')
    
    ## Initializing variables 
    processed_dekads = []
    processed_cdi = []
    count_sma_recovery_arrays = []
    count_fapar_recovery_arrays = []
    ## Run CDI for the selected range of dekads
    for dekad in dekads:
        # Arguments for the cdi_run function
        file_name = "cdinx_m_euu_{}{:02d}{:02d}_t.nc".format(dekad.year, dekad.month, dekad.day)
        file_path = os.path.join(out_path, file_name)
        if os.path.isfile(file_path) == True: 
            #cdi = rasterio.open(file_path).read()
            loggers['log'].info(f'CDI for dekad {dekad} it already exists. File path {file_path}')
            sys.exit()
        else:
            try:
                if inputs_type == 'database':
                    inputs = get_inputs_from_database(dekad, loggers)
                else:
                    inputs = {}
                    # Universal way to import data for testing (.img and .nc) and regular procedures (.nc)
                    cdi_p, cdi_2p, cdi_3p = get_required_cdi_files_to_calculate_dekad(dekad, out_path, loggers)
                    inputs['cdi_p'] = cdi_p
                    inputs['cdi_2p'] = cdi_2p
                    inputs['cdi_3p'] = cdi_3p
                    #import fapar, and spi
                    for input_variable_name in ['fapar','spi1','spi3']:
                        loggers['log'].debug(f'Importing {input_variable_name}')
                        if input_variable_name == 'fapar':
                            try:
                                variable_array = get_input(dekad, inp_variables, input_variable_name)
                            except Exception as e:
                                # Fix for missing fAPAR data in 2022, only for a time series fix, 
                                # Not for operational runs
                                # Set the whole fAPAR map to 9999 so the CDI is calculated without fAPAR
                                loggers['log'].info(f'variable {input_variable_name} not found {e}. setting the array to 9999')
                                variable_array = np.zeros_like(cdi_p) + 9999.0
                            # fapar fix for northern europe: where fAPAR is no data, is set to 9999
                            variable_array = variable_array/1000
                            variable_array = np.where(np.isnan(variable_array), 9999.0, variable_array)
                            # Phenology mask
                            variable_array = apply_phenology_mask(variable_array, dekad, inp_variables)
                        else:
                            variable_array = get_input(dekad, inp_variables, input_variable_name)
                            variable_array = variable_array/1000
                        inputs[input_variable_name] = variable_array
                    #import sma
                    sma = get_smian(dekad, inp_variables, loggers)
                    inputs['sma'] = sma
                    #loggers['log'].info(f'inputs {inputs}')
            except Exception as e:
                loggers['log'].info(f'Some of the inputs are not available. Skipping dekad {dekad}: {e}')
                continue
            args = {}
            args['inputs'] = inputs
            args['formated_date'] = dekad.start
            args['count_sma_recovery'] = count_sma_recovery_p_dekad
            args['count_fapar_recovery'] = count_fapar_recovery_p_dekad
            args['cdi_version'] = cdi_version
            args['loggers'] = loggers

            # Run CDI --------------------------------------------------------------------------------------
            cdi, cases, count_sma_recovery_p_dekad_temp, count_fapar_recovery_p_dekad_temp, zspi, cdi_p_masked = cdi_run(**args)
            #-----------------------------------------------------------------------------------------------
            # Masking CDI
            cdi = masking_cdi(inputs, cdi)
            if np.all(inputs['fapar'] == 9999.0):
                # Fix for missing fAPAR data in 2022, only for a time series fix, 
                # Not for operational runs
                # Set the whole fAPAR map to 9999 so the CDI is calculated without fAPAR
                cdi_modis_fp = get_nc_file_name_for_dekad(dekad)
                cdi_modis = rasterio.open(f'/drought_prod/edo_cdinx_mgt/ver3-0-0_MODIS-VIIRS/{cdi_modis_fp}').read()
                cdi = np.where(cdi_modis == 3, 3, cdi)
            # Testing results---------------------------------------------------------------------------------
            # x = 679 #5895123
            # y = 708 #1957745
            # x = 605 #5527995
            # y = 695 #2022497
            # xcol = 580 #3035541
            # yrow = 719 #2061159
            # point_values = {}
            # inputs['cdi_p_masked'] = cdi_p_masked
            # inputs['zspi'] = zspi
            # inputs['cases'] = cases
            # for input in inputs.keys():
            #     #save_results(input, inputs[input], dekad.start, f"{out_path}")
            #     point_values[input] = inputs[input][0][yrow,xcol]
            # loggers['log'].info(f'Point testing values \n {point_values}')
            # point_file = f"{out_path}/point_values_{x}-{y}_{dekad.start}.json"
            # jsonFile = open(point_file, "w+")
            # jsonFile.write(json.dumps(point_values))
            # jsonFile.close()

            # -----------------------------------------------------------------------------------------------
        
        # Collecting results
        processed_dekads.append(dekad)      
        processed_cdi.append(cdi[0])             
        count_sma_recovery_arrays.append(count_sma_recovery_p_dekad_temp[0])
        count_fapar_recovery_arrays.append(count_fapar_recovery_p_dekad_temp[0])
        # Saving results as an nc files
        cdi_nparray = np.array([cdi[0]])
        save_results('cdinx', cdi_nparray, dekad.start, out_path )
        if cases.any():
            cases_nparray = np.array([cases[0]])  
            save_results('cdinx_cases', cases_nparray, dekad.start, out_path)  
        loggers['log'].info(f"CDI dekad {dekad} calculated")
        loggers['log'].info("--------------------------------------------------------------------------")

    return count_sma_recovery_arrays, count_fapar_recovery_arrays, processed_dekads, processed_cdi


def update_count(cdi_start, previous_dekad, processed_dekads,
                count_sma_recovery, count_fapar_recovery,
                count_sma_recovery_arrays, count_fapar_recovery_arrays,
                loggers):
    ## Update count arrays and time array
    # Selecting the arrays that don't need to be updated
    not_processed_sma_count_dataset = count_sma_recovery.sel(time=slice(cdi_start, previous_dekad.start ))
    not_processed_fapar_count_dataset = count_fapar_recovery.sel(time=slice(cdi_start, previous_dekad.start))
    # Getting only the numpy arrays
    not_processed_sma_count_arrays = not_processed_sma_count_dataset['count_sma_recovery'].values
    not_processed_fapar_count_arrays = not_processed_fapar_count_dataset['count_fapar_recovery'].values
    # Getting the dates that don't need to be updated 
    not_processed_time = not_processed_fapar_count_dataset.time
    times = [np.datetime64(not_processed_time.values[i], 'D') for i in range(0, len(not_processed_time.values))]
    # Closing the files to avoid conflicts while updating
    count_sma_recovery.close()   
    count_fapar_recovery.close()
    # Converting the list of processed arrays in a numpy array  
    processed_count_sma_recovery_arrays = np.array(count_sma_recovery_arrays)
    processed_count_fapar_recovery_arrays = np.array(count_fapar_recovery_arrays)
    if times == []:
        updated_sma_count_arrays = count_sma_recovery_array
        updated_fapar_count_arrays = count_fapar_recovery_array
    else:
        updated_sma_count_arrays = np.vstack((not_processed_sma_count_arrays, processed_count_sma_recovery_arrays))
        updated_fapar_count_arrays = np.vstack((not_processed_fapar_count_arrays, processed_count_fapar_recovery_arrays))  
    # Update times list for the new file
    for i in processed_dekads:
        if i not in times:
            times.append(i)
    updated_times = times
    count_sma_recovery_arrays = updated_sma_count_arrays 
    count_fapar_recovery_arrays = updated_fapar_count_arrays 
    processed_dekads = updated_times
    # Closing the count files 
    count_sma_recovery.close()   
    count_fapar_recovery.close()

    return count_sma_recovery_arrays, count_fapar_recovery_arrays, processed_dekads


def calculate_statistics(masks, processed_dekads, processed_cdi, out_path, run_version):
    # to be put in the configuration file
    cdi_values = {'watch':1, 'warning':2,'alert':3,'recovery':4,'sma_recovery':5,'fapar_recovery':6}
    # Loop for the input masks eu27 and eu
    for mask in masks.keys(): 
        stats_file = f"cdi_statistics_{mask}_ver{run_version}.json"
        stats_file_path = f"{out_path}/{stats_file}"
        try:
            f = open(stats_file, "r")
            statistics = json.load(f)
        except:
            # Starting statistics dictionary
            statistics = {}
        # Eliminating nan value -9999
        land = np.where(masks[mask] != 1, 0, 1)
        # Total number of pixels
        tot_land = np.count_nonzero(land)
        for dekad, cdi in list(zip(processed_dekads, processed_cdi)):
            dekad_str = f"{np.datetime64(dekad.start, 'D')}"
            #dekad_str = np.datetime_as_string(dekad.start, unit='D')
            
            if dekad_str not in statistics.keys():
                statistics[dekad_str] = {}
            # Masking the cdi map with only the eu or eu27 pixels
            cdi = cdi * land
            for category in cdi_values.keys():
                category_array = np.where(cdi == cdi_values[category], 1, 0)
                category_pixels = np.count_nonzero(category_array)
                percentage = (category_pixels/tot_land)*100      
                percentage = np.round_(percentage, decimals = 5)
                if f'{category} %' not in statistics[dekad_str].keys():
                    statistics[dekad_str][f'{category} %'] = [category_pixels, tot_land, percentage]
            if stats_file in os.listdir(out_path):
                os.remove(stats_file_path)
            jsonFile = open(stats_file_path, "w+")
            jsonFile.write(json.dumps(statistics))
            jsonFile.close()


def main():
    """
    usage: cdi_prdmanager.py [-h] -r {a,m} [-d DATE] [-i START_DATE] [-e END_DATE] [--u] [--v VERSION]
    """
    args = sys.argv[1:]
    args = parse_args(args)
    today = date.today()

    # Move working directory from Python installation to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Setting logging events
    with open('cdi_logging_config.json') as logging_config_file:
        config = json.load(logging_config_file)
        logging.config.dictConfig(config)
    
    logging.info(
        """\n\n------------------------------------------------------------\n""" \
        """ CDI log file start \n""" \
        """------------------------------------------------------------\n""")
    logging.getLogger('paramiko.transport').setLevel(logging.WARNING)
    logging.getLogger('paramiko.transport.sftp').setLevel(logging.WARNING)
    warnings.filterwarnings('ignore')
    loggers = dict([('log', logging.getLogger('log')),
                  ('email', logging.getLogger('email'))])
    loggers['log'].debug('Arguments:{} '.format(args))

    # loading configuration file
    with open('cdi_config.json') as json_data_file:
        data = json.load(json_data_file)

    # Setting working directories
    out_path = data['w_dirs']['out_path']
    inp_path = data['w_dirs']['inp_path']
    input_variables = ['spa01', 'spa01', 'fapan', 'smian']
    eu27_mask_filepath = data['eu27_mask_filepath']
    eu_mask_filepath = data['eu_mask_filepath']
    custom_mask_filepath = data['custom_mask_filepath']
    cdi_start_string = data['cdi_start']

    count_sma_recovery_file = '{}/count_sma_recovery.nc'.format(out_path)
    count_fapar_recovery_file ='{}/count_fapar_recovery.nc'.format(out_path)

    # defining al necessary inpaths
    inp_variables = data['input_variables']

    # importing masks for statistics calculations
    eu27_mask = xr.open_rasterio(eu27_mask_filepath).values
    eu_mask = xr.open_rasterio(eu_mask_filepath).values
    try:
        custom_mask = xr.open_rasterio(eu_mask_filepath).values
        masks = {'EU27': eu27_mask, 'EU': eu_mask, 'custom': custom_mask}
    except:    
        masks = {'EU27': eu27_mask, 'EU': eu_mask}

    # Start date of the CDI timeseries
    cdi_start = format_string_date(cdi_start_string)
    cdi_version = args['version']
    run_version = cdi_version.replace('.','-')

    # Define the dates for the run, according with the run type
    formated_date = None
    if args['run_type'] == 'm' and args['date']:
        run_date = args['date'] 
        formated_start_date = format_string_date(run_date) # Render date into datetime
        formated_end_date = formated_start_date

    if args['run_type'] == 'a':
        formated_start_date = formated_end_date = today # Render date into datetime

    if args['start_date'] and args['end_date'] and args['run_type'] == 'm':
        start_date = args['start_date']
        end_date = args['end_date'] 
        formated_end_date = format_string_date(end_date)
        formated_start_date = format_string_date(start_date)
    
    # Check on the date inserted by the user
    if formated_start_date < cdi_start:
        loggers['log'].info(f'The requested start date {formated_start_date} is < than the start date of the inputs {cdi_start}')
        sys.exit()

    # Starting the count for temporary recovery classes
    try:
        # Open the count files if they exist
        count_sma_recovery = xr.open_dataset(count_sma_recovery_file, drop_variables=['3035'])
        count_fapar_recovery = xr.open_dataset(count_fapar_recovery_file, drop_variables=['3035'])
    except:
        # Otherwise set the count to zero
        count_sma_recovery = np.zeros((1, 950, 1000)).astype('int16')
        count_fapar_recovery = np.zeros((1, 950, 1000)).astype('int16')
        if formated_start_date > cdi_start:
            loggers['log'].info(f'The count files are not present or were elimnated. Recreate them using a timeseries run')
            sys.exit()

    # Define inputs type (filesystem/database)
    if args['database-inputs']:
        inputs_type = 'database'
    else:
        inputs_type = 'files'

    # Creating the list of dekads for the run 
    start_dekad = Dekad(formated_start_date.year,formated_start_date.month, formated_start_date.day)
    end_dekad = Dekad(formated_end_date.year,formated_end_date.month, formated_end_date.day)
    dekads = start_dekad + end_dekad
    previous_dekad = start_dekad - 1 # previous dekad from the start of the run
    loggers['log'].debug('start dekad:{}, end dekad:{}, \n dekads to process:{}'.format(start_dekad, end_dekad, dekads))

    # Run CDI taking into account the temporary classes constraint ---------------------------------------------------------------------
    count_sma_recovery_processed_arrays, count_fapar_recovery_processed_arrays, processed_dekads, processed_cdi = run_cdi_with_count(dekads, cdi_start, count_sma_recovery, count_fapar_recovery, cdi_version, inp_variables, out_path, inputs_type, loggers)
    # ----------------------------------------------------------------------------------------------------------------------------------                      
    formated_processed_dekads = []
    string_processed_dekads = []
    for d in processed_dekads:
        fd = np.datetime64(d.start, 'D')
        formated_processed_dekads.append(fd)
        string_processed_dekads.append(np.datetime_as_string(fd, unit='D'))
    loggers['email'].info(f"CDI procedure has finished for dekad {string_processed_dekads}." 
                          f" Please check the database and output nc file.")
    
    # Saving results in the database
    if args['database-import']:
        for dekad, cdi in zip(processed_dekads, processed_cdi):
            loggers['log'].info(f">>> Start DB import of dekad: {dekad.start} ")
            # Set value 8 as NaN so it wont be saved in the db
            cdi = np.where(cdi == 8.0, np.nan, cdi)
            # Importing into the database
            dbin.cdi_database_import(dekad, cdi)
            loggers['log'].info("Results were imported in the database")
    
    # <<proccesed_cdi>> variable can be used for saving all cdi results ->
    # in one netCDF file -> to be developed if necessary

    # Calculate statistics
    calculate_statistics(masks, processed_dekads, processed_cdi, out_path, run_version)

    # Update counts
    if args['update_count'] and (start_dekad.start != cdi_start):
        # Update the count 
        updated_count_sma_recovery_arrays, updated_count_fapar_recovery_arrays, updated_processed_dekads = update_count(cdi_start, previous_dekad, formated_processed_dekads, count_sma_recovery, count_fapar_recovery, count_sma_recovery_processed_arrays, count_fapar_recovery_processed_arrays, loggers)
        # update the processed arrays and dekads
        count_sma_recovery_processed_arrays = updated_count_sma_recovery_arrays
        count_fapar_recovery_processed_arrays = updated_count_fapar_recovery_arrays
        formated_processed_dekads = updated_processed_dekads
    
    if args['update_count']:
        # Saving the updated or new count_sma_recovery & count_fapar_recovery
        # Operationally it takes 3 min
        save_laea_arrays('count_sma_recovery', count_sma_recovery_processed_arrays, formated_processed_dekads, out_path)
        save_laea_arrays('count_fapar_recovery', count_fapar_recovery_processed_arrays, formated_processed_dekads, out_path)

if __name__ == '__main__':
    main()
