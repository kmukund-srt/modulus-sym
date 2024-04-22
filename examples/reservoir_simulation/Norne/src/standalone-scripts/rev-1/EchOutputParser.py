"""Utility for parsing ECHELON/ECLIPSE binary files."""
from typing import Any, Dict
import math
from struct import unpack
import numpy as np
import re
import fnmatch
import os
import pandas as pd

class EchBinaryOutputParser():
    """ Parser for UNRST, EGRID, INIT files
        This is a parser class to extract relevant data for ML from EGRID, UNRST binary files
        and the structured CSV file that ECHELON outputs for summary report.

        `get_simulation_data` is the main function to be called to get the data

        The output has several data structures with the following contents.
        1. Report times: np.array -> Returns reporting times from CSV file

        2. states data:Dict ->  Keys correspond to state variable ex. "PRESSURE', 'SWAT', 'SGAS', 'RS', 'RV' etc.
                                Currently, we have hard-coded the state variables in self.STATE_VARS tuple below.
                                Each state variable is a [time_slices x NX x NY x NZ] np.array where time_slices
                                correspond to discrete report times that user wants the data from.
                                NX, NY, NZ are the grid dimensions.

        3. Wells and connections information data: Dict -> Nested dictionary object of data
                        Data[time_index]['Well Name'] = {
                                                          'Connections': {
                                                            'Indices': np.array([I, J, K] indices for each connection), 
                                                            'Status': np.array(0 or 1 open or shut status for each connection),
                                                            'Dir': np.array('X', 'Y', 'Z' penetration dirrection for each connection as defined in self.CONN_DIR)
                                                          },
                                                          'TYPE': Well type as defined in self.WELL_TYPE
                                                          'STATUS': OPEN/SHUT
                                                         } 

        4. Wells solution summary data: Dict
                       Data['UNSMRY quantity such as WOPR, WGPR, WWIR, WCUT']['Well Name'] = np.array of length time_slices
 
    Args:
    """
    
    def __init__(self, **kwargs: Any)->None:
        """ 
            Initialize params. 
            All indices and values are from UNRST file documentation
        """        
        self.SUPPORTED_DATA_TYPES = {
            'INTE': (4, 'i', 1000),
            'REAL': (4, 'f', 1000),
            'LOGI': (4, 'i', 1000),
            'DOUB': (8, 'd', 1000),
            'CHAR': (8, '8s', 105),
            'MESS': (8, '8s', 105),
            'C008': (8, '8s', 105)
        }

        self.INTEHEAD_VARS_INDEX = {
            'NWELLS': 16, # 0-index -- number of wells
            'NCWMAX': 17, # 0-index -- maximum number of completions per well
            'NICONZ': 32, # 0-index -- number of data elements per completion in ICON array
            'NIWELZ': 24, # 0-index -- number of data elements per well in IWEL array
            'NZWELZ': 27, # 0-index -- number of characters in 8-character well name
            'ICON_VARSTART': 1, # 0-index -- index inside ICON data of connection index I for (I, J, K) triplet (2 is for J, 3 is for K)
            'ICON_STATUS': 5, # 0-index -- index inside ICON data for connection status (> 0 open, <= 0 shut)
            'ICON_DIR': 13, # 0-index -- index inside ICON data for connection direction (1=x, 2=y, 3=z, 4=fractured in x-direction, 5=fractured in y-direction) If undefined or zero, assume Z
            'IWEL_TYPE': 6, # 0-index -- index inside IWEL data denoting the type of well (producer:1, oil inj:2, wat inj:3, gas inj:4)
            'IWEL_STATUS': 10, # 0-index -- index inside IWEL data denoting the status of well (> 0 open, <= 0 shut)
        }

        self.WELL_TYPE = {
            1: 'PRODUCER',
            2: 'OIL INJECTOR',
            3: 'WATER INJECTOR',
            4: 'GAS INJECTOR'
        }

        self.CONN_DIR = {
            1: 'X',
            2: 'Y',
            3: 'Z',
            4: 'XFRAC',
            5: 'YFRAC',
            0: 'Z', # default
        }

        self.STATE_VARS = ('PRESSURE', 'SGAS', 'SWAT', 'RS', 'RV') 

        self.INIT_ATTRIBS = ('PORO', 'PERMX', 'PERMY', 'PERMZ')

        self.EGRID_ATTRIBS = ('GRIDHEAD', 'ACTNUM')
        
        self.UNRST_ATTRIBS = ('INTEHEAD', 'DOUBHEAD', 'IWEL', 'ICON', 'ZWEL')

    def _check_and_fetch_type_info(self, data_type):
        """Returns element size, format and element skip for the given data type.

        Parameters
        ----------
        data_type: str
            Should be a key from the SUPPORTED_DATA_TYPES

        Returns
        -------
        type_info: tuple
        """
        try:
            return self.SUPPORTED_DATA_TYPES[data_type]
        except KeyError as exc:
            raise ValueError('Unknown datatype %s.' % data_type) from exc

    def _check_and_fetch_file(self, path, pattern, return_relative=False):
        """Find files matching pattern ignoring case-style, only in the given directory."""
        found = []
        reg_expr = re.compile(fnmatch.translate(pattern), re.IGNORECASE)

        # Listing files in the specified directory
        for f in os.listdir(path):
            # Check if the file matches the pattern
            if re.match(reg_expr, f):
                f_path = os.path.join(path, f)
                if return_relative:
                    found.append(os.path.relpath(f_path, start=path))
                else:
                    found.append(f_path)

        return found

    def _get_file_path(self, dir_path, filename):
        """Find a file withihn the directory. Raise error if multiple files found."""
        files = self._check_and_fetch_file(dir_path, filename)
        if not files:
            raise FileNotFoundError('Directory {} does not contain file {}.'.format(dir_path, filename))
        
        if len(files) > 1:
            raise ValueError('Directory {} contains multiple {} files.'.format(dir_path, filename))
        
        return files[0]
        

    def _parse_keywords(self, path, attrs=None, time_slices_to_include=None):
        """Parses and populates binary file data into a dictionary.

        Parameters
        ----------
        path : str
            Path to the binary file to be read
        attrs : list or None, optional
            List of keywords requested (e.g. 'PRESSURE', 'SWAT').

        Returns
        -------
        data: dict
            Data from the binary file in the form of dict.
            data[key] contains BINARY data and data type info
        """

        # A simple utility to check if a given time slice should
        # be included or not in the final list
        def _should_include_slice(slice):
            if time_slices_to_include is None:
                return True
            return slice in time_slices_to_include

        sections_counter = {} if attrs is None else {attr: 0 for attr in attrs}

        with open(path, 'rb') as f:
            header = f.read(4)
            sections = dict()
            while True:
                try:
                    section_name = unpack('8s', f.read(8))[0].decode('ascii').strip().upper()
                except:
                    break
                n_elements = unpack('>i', f.read(4))[0]
                data_type = unpack('4s', f.read(4))[0].decode('ascii')
                f.read(8)
                element_size, fmt, element_skip = self._check_and_fetch_type_info(data_type)
                f.seek(f.tell() - 24)
                binary_data = f.read(24 + element_size * n_elements + 8 * (math.floor((n_elements - 1) / element_skip) + 1))
                if (attrs is None) or (section_name in attrs):
                    sections_counter[section_name] = sections_counter.get(section_name, 0) + 1
                    if section_name not in sections:
                        sections[section_name] = []
                    section = (n_elements, data_type, element_size, fmt, element_skip, binary_data)
                    slice = sections_counter[section_name] - 1 # 0-indexing
                    # Include only those time slices specified in the time_slices_to_include array.
                    if _should_include_slice(slice):
                        section = self._fetch_keyword_data(section)
                        sections[section_name].append(section)

        return header, sections

    def _fetch_keyword_data(self, section):
        """Reads data for the specified keyword from the binary file.

        Parameters
        ----------
        section: list

        Returns
        -------
        decoded_section: ndarray
        """
        n_elements, data_type, element_size, fmt, element_skip, binary_data = section
        n_skip = math.floor((n_elements - 1) / element_skip)
        skip_elements = 8 // element_size
        skip_elements_total = n_skip * skip_elements
        data_format = fmt * (n_elements + skip_elements_total)
        data_size = element_size * (n_elements + skip_elements_total)
        if data_type in ['INTE', 'REAL', 'LOGI', 'DOUB']:
            data_format = '>' + data_format
        decoded_section = list(unpack(data_format, binary_data[24: 24 + data_size]))
        del_ind = np.repeat(np.arange(1, 1 + n_skip) * element_skip, skip_elements)
        del_ind += np.arange(len(del_ind))
        decoded_section = np.delete(decoded_section, del_ind)
        if data_type in ['CHAR', 'C008']:
            decoded_section = np.char.decode(decoded_section, encoding='ascii')
        return decoded_section

    def _parse_ech_bin(self, path, attrs=None, time_slices_to_include=None):
        """Reads binary ECLIPSE file into a dictionary.

        Parameters
        ----------
        path : str
            Path to the binary file to be read
        attrs : list or None, optional
            List of keywords requested (e.g. 'PRESSURE', 'SWAT').

        Returns
        -------
        sections : dict
            Data from the binary file.
        """
        if attrs is None:
            raise ValueError('Keyword attribute cannot be empty')
 
        if isinstance(attrs, str):
            attrs = [attrs]

        attrs = [attr.strip().upper() for attr in attrs]
        _, sections = self._parse_keywords(path, attrs, time_slices_to_include)

        return sections

    def _parse_init(self, path_to_result, basename):
        """Parses EGRID file and dumps grid and actnum info into a dictionary.

        Parameters
        ----------
        path_to_result : str
            Directory where the EGRID file is located
        basename : file rootname
            Rootname of the file (DATA file root name).

        Returns
        -------
        egrid : dict
            Data from the binary file.
        """
        init_path = self._get_file_path(path_to_result, basename + '.INIT')
        init_data = self._parse_ech_bin(init_path, self.INIT_ATTRIBS)

        return init_data

    def _parse_egrid(self, path_to_result, basename):
        """Parses EGRID file and dumps grid and actnum info into a dictionary.

        Parameters
        ----------
        path_to_result : str
            Directory where the EGRID file is located
        basename : file rootname
            Rootname of the file (DATA file root name).

        Returns
        -------
        egrid : dict
            Data from the binary file.
        """
        egrid_path = self._get_file_path(path_to_result, basename + '.EGRID')
        egrid_data = self._parse_ech_bin(egrid_path, self.EGRID_ATTRIBS)

        return egrid_data

    def _parse_unrst(self, path_to_result, basename, time_slices_to_include=None):
        """Parses UNRST file and dumps grid and actnum info into a dictionary.

        Parameters
        ----------
        path_to_result : str
            Directory where the UNRST file is located
        basename : file rootname
            Rootname of the file (DATA file root name).
        time_slices_to_include: A subset of time slices that should be parsed
            The default will parse all the time slices in the UNRST file

        Returns
        -------
        states : dict
            Data from the binary file.
        """
        unrst_path = self._get_file_path(path_to_result, basename + '.UNRST')
        attrs = self.STATE_VARS + self.UNRST_ATTRIBS
        unrst_data = self._parse_ech_bin(unrst_path, attrs, time_slices_to_include)

        return unrst_data

    def _parse_csv(self,path_to_result, basename, time_slices_to_include=None):
        """Parses csv file and dumps grid and actnum info into a dictionary.

        Parameters
        ----------
        path_to_result : str
            Directory where the UNRST file is located
        basename : file rootname
            Rootname of the file (DATA file root name).
        time_slices_to_include: A subset of time slices that should be parsed
            The default will parse all the time slices in the UNRST file

        Returns
        -------
        all_report_times: np.array
            All report times except t = 0
        wells_solution_data : dict(well quantity: dict(well_name: np.array(values)))
            Data from the binary file except at t = 0.
        """
        csv_path = self._get_file_path(path_to_result, basename + '.csv')

        # Assuming the CSV file has the data starting from row 1 with no header rows
        df = pd.read_csv(csv_path, header=None)

        # These numbers are specific to Echelon's csv output
        num_columns_to_skip = 4
        # 4th row is where the wells output begins. However, we skip the t = 0 output
        # to be consistent with what is output in the binary UNRST file
        row_data_begins = 5 

        # Skipping the first four columns that specify times and are not associated with wells
        all_report_times = df.iloc[5:, 2].to_list()
        all_report_times = np.array([float(x.strip()) for x in all_report_times]) 

        df = df.iloc[:, num_columns_to_skip:]

        # Create an empty dictionary to store the wells data
        wells_solution_data = {}

        # Extract the names for the dictionaries from the first row
        dict_names = df.iloc[0]

        # Iterate over the columns
        for col in df.columns:
            # Use the first row as the dictionary name, third row as the key
            dict_name = dict_names[col].strip() # 4 columns are ignored
            key = df.iloc[2, col-num_columns_to_skip].strip()

            # The rest of the column are the values. We are using a naive float conversion
            # instead of pd.numeric (given below) so as to not get rid of 0 values for now.
            values = df.iloc[row_data_begins:, col-num_columns_to_skip].to_list()
            values = np.array([float(x.strip()) for x in values]) 

            # Get the time-sliced data. If time slice array is empty, include all the values
            if time_slices_to_include is None:
                values = np.array(values)
            else:
                values = np.array(values)[time_slices_to_include]           

            # Add to the dictionary (default dict={})
            wells_solution_data.setdefault(dict_name, {})[key] = values

        return all_report_times, wells_solution_data

    def _parse_well_conn_data(self, unrst_data) -> Dict:
        intheaders = unrst_data['INTEHEAD']
        well_names = unrst_data['ZWEL']
        well_type_status_info = unrst_data['IWEL']
        conn_info = unrst_data['ICON']
        conn_id_start = self.INTEHEAD_VARS_INDEX['ICON_VARSTART'] # from UNRST description for 'ICON'
        conn_id_end = conn_id_start + 3 # (I, J, K) triplet is 3 and so offset is 3
        wells_conns_type_status_dict = {}
        for idx, _ in enumerate(well_names):
            nwells, ndata_per_well, wellname_char_length , ndata_per_conn, nconn_per_well  = \
              intheaders[idx][self.INTEHEAD_VARS_INDEX['NWELLS']], \
              intheaders[idx][self.INTEHEAD_VARS_INDEX['NIWELZ']], \
              intheaders[idx][self.INTEHEAD_VARS_INDEX['NZWELZ']], \
              intheaders[idx][self.INTEHEAD_VARS_INDEX['NICONZ']], \
              intheaders[idx][self.INTEHEAD_VARS_INDEX['NCWMAX']] 
            
            ####################################################################
            # Reshape the 1D arrays into multidimensional arrays appropriately.                
            ####################################################################
            well_names[idx] = well_names[idx].reshape(nwells, wellname_char_length)
            conn_info[idx] = conn_info[idx].reshape(nwells, nconn_per_well, ndata_per_conn)
            well_type_status_info[idx] = well_type_status_info[idx].reshape(nwells, ndata_per_well)

            active_wells = []
            for well in range(nwells):
                name = "".join([str(i) for i in well_names[idx][well, :]]).split()[0]
                active_wells.append(name)
                ################################
                # Get the well type and status.                
                ################################
                well_type_int = well_type_status_info[idx][well, self.INTEHEAD_VARS_INDEX['IWEL_TYPE']]
                i, j, k = well_type_status_info[idx][well, 0], well_type_status_info[idx][well, 1], well_type_status_info[idx][well, 2]
                well_status = 'SHUT' if well_type_status_info[idx][well, self.INTEHEAD_VARS_INDEX['IWEL_STATUS']] <= 0 else 'OPEN'
                well_type = ''
                if well_type_int not in self.WELL_TYPE:
                    raise KeyError(f'{well_type_int} is not a valid well type (allowed values are 1, 2, 3, 4).')
                else:
                    well_type = self.WELL_TYPE[well_type_int]

                ####################################################################
                # Get the connections and subtract 1 for 0-indexing.
                # Remove rows with -1 -1 -1 since they are non-existent
                # padded rows to satisfy the nconn_per_well (NCWMAX) dimensionality
                ####################################################################
                conns = conn_info[idx][well, :, conn_id_start:conn_id_end]-1 
                actual_conns = ~(conns == -1).all(axis=1)
                trim_conns = conns[actual_conns]

                ####################################################################
                # Get the connections status and trim it to actual ones.
                ####################################################################
                conn_status = conn_info[idx][well, :, self.INTEHEAD_VARS_INDEX['ICON_STATUS']] 
                conn_status = conn_status[actual_conns]

                ####################################################################
                # Get the connections direction and trim it to actual ones.
                ####################################################################
                conn_dir = conn_info[idx][well, :, self.INTEHEAD_VARS_INDEX['ICON_DIR']] 
                conn_dir = conn_dir[actual_conns]
                conn_dir = np.array([self.CONN_DIR.get(dir_int, 'Z') for dir_int in conn_dir])

                ####################################################################
                # Now populate the dictionary.
                ####################################################################
                wells_conns_type_status_dict.setdefault(idx, {})[name] = \
                    {'Connections': {'Indices': trim_conns, 'Status':conn_status, 'Dir': conn_dir}, \
                     'TYPE': well_type, 'STATUS': well_status}
        
        return wells_conns_type_status_dict

    def _test_parser(self):
        path_to_result = './test'
        NORNE = True
        time_slices_to_include = None
        if NORNE:
            basename =  'FULLNORNE2'
            time_slices_to_include = np.linspace(1, 246, 10, dtype=int)-1 # 0-indexing
            output = self.get_simulation_data(path_to_result, basename, time_slices_to_include=time_slices_to_include)
        else:
            basename =  '5-CELL' # 'FULLNORNE2' 
            output = self.get_simulation_data(path_to_result, basename)
            
        init_data = self.get_init_data(path_to_result, basename)
        poro = init_data['INITIAL DATA']['PORO']
        permx = init_data['INITIAL DATA']['PERMX']
        permy = init_data['INITIAL DATA']['PERMY']
        permz = init_data['INITIAL DATA']['PERMZ']
        print("----------------- POROSITY ARRAY -------------: len = ", len(poro[0]))
        print(poro)
        print("----------------- PERMX ARRAY -------------: len = ", len(permx[0]))
        print(permx)
        print("----------------- PERMY ARRAY -------------: len = ", len(permy[0]))
        print(permy)
        print("----------------- PERMZ ARRAY -------------: len = ", len(permz[0]))
        print(permz)

        # output = self.get_simulation_data(path_to_result, basename)
        wells_conns_type_status_dict = output.get('WELLS AND CONNECTIONS', {})
        # print(wells_conns_type_status_dict)

        wells_soln = output.get('WELLS SOLUTION SUMMARY', {})
        for key, val in wells_soln.items():
            print('------ Indx: ', key, ' ------')
            for key1, val1 in val.items():
                print(key1, val1)

        report_times = output['REPORT TIMES']
        press_slices = output['STATES']["PRESSURE"]
        for idx, pr in enumerate(press_slices):
            print("Time: ", report_times[idx], "  Pressure: ", pr)
        swat = output['STATES']["SWAT"]
        for idx, sw in enumerate(swat):
            print("Time: ", report_times[idx], "  Swater: ", sw)
        sgas = output['STATES']["SGAS"]
        for idx, sg in enumerate(sgas):
            print("Time: ", report_times[idx], "  Sg: ", sg)
        # all_report_times, states, wells_conns_type_status_dict, wells_solution_summary_data_dict
        # for key, val in wells_conns_type_status_dict[9].items():
        #     print(f'Well: {key}, {val}')   
        # b2h_wopr = wells_solution_summary_data_dict.get('WOPR', {}).get('B-2H')
        # print("b2h_wopr = ", b2h_wopr)
        # Extract all the producer wells for time index 9
        # To filter for all time indices
        
        # watinj_well_names = [
        #     "C-1H",
        #     "C-2H",
        #     "C-3H",
        #     "C-4AH",
        #     "C-4H",
        #     "F-1H",
        #     "F-2H",
        #     "F-3H",
        #     "F-4H",
        # ]

        producer_wells_subset_all_times = {time_index: {well_name: details for well_name, details in wells_dict.items()
                                                if details.get('TYPE') == 'PRODUCER'}
                                   for time_index, wells_dict in wells_conns_type_status_dict.items()}
        water_injector_wells_subset_all_times = {time_index: {well_name: details for well_name, details in wells_dict.items()
                                                if details.get('TYPE') == 'WATER INJECTOR'}
                                   for time_index, wells_dict in wells_conns_type_status_dict.items()}
        gas_injector_wells_subset_all_times = {time_index: {well_name: details for well_name, details in wells_dict.items()
                                                if details.get('TYPE') == 'GAS INJECTOR'}
                                   for time_index, wells_dict in wells_conns_type_status_dict.items()}
        print("----------- Printing Producer Details -------------")
        for key, val in producer_wells_subset_all_times.items():
            # print("Time index: ", key, "   Producers:", len(val), "  Wells:", [well for well,_ in val.items() if well in watinj_well_names])
            print("Time index: ", key, "   Producers:", len(val), "  Wells:", [well for well,_ in val.items()])
        print("----------- Printing Water Injector Details -------------")
        for key, val in water_injector_wells_subset_all_times.items():
            print("Time index: ", key, "   Water injectors:", len(val), "  Wells:", [well for well,_ in val.items()])
        print("----------- Printing Gas Injector Details -------------")
        for key, val in gas_injector_wells_subset_all_times.items():
            print("Time index: ", key, "   Gas injectors:", len(val), "  Wells:", [well for well,_ in val.items()])
        
        prod_wells_list = [
            "B-1BH",
            "B-1H" ,
            "B-2H" ,
            "B-3H" ,
            "B-4BH",
            "B-4DH",
            "B-4H" ,
            "D-1CH",
            "D-1H" ,
            "D-2H" ,
            "D-3AH",
            "D-3BH",
            "D-4AH",
            "D-4H" ,
            "E-1H" ,
            "E-2AH",
            "E-2H" ,
            "E-3AH",
            "E-3CH",
            "E-3H" ,
            "E-4AH",
            "K-3H" ,
        ]

        producer_wells_subset_all_times = {time_index: {well_name: details for well_name, details in wells_dict.items()
                                if details.get('TYPE') == 'PRODUCER' and well_name in prod_wells_list}
                    for time_index, wells_dict in wells_conns_type_status_dict.items()}
    
        print("############################# len(producer_wells_subset_all_times) = ", len(producer_wells_subset_all_times), " ###############################")
        # Extract just the connection indices dict
        producer_wells_IJK_dict = {well_name: details['Connections']['Indices']\
            for well_name, details in producer_wells_subset_all_times[len(producer_wells_subset_all_times)-1].items()}
        for idx, (well, conn_indices) in enumerate(producer_wells_IJK_dict.items()):
            print(well, conn_indices[:, :])
            # print(conn_indices[:, :])

        # print([key for key,_ in producer_wells_subset_all_times[9].items()])

    def get_grid_data(self, path_to_result, basename):
        egrid = self._parse_egrid(path_to_result, basename)
        return {'GRID DATA': egrid}

    def get_init_data(self, path_to_result, basename):
        init = self._parse_init(path_to_result, basename)
        return {'INITIAL DATA': init}

    def get_simulation_data(self, path_to_result, basename, time_slices_to_include=None):
        unrst_data = self._parse_unrst(path_to_result, basename, time_slices_to_include=time_slices_to_include)
        wells_conns_type_status_dict = self._parse_well_conn_data(unrst_data)
        all_report_times, wells_solution_summary_data = self._parse_csv(path_to_result, basename, time_slices_to_include=time_slices_to_include)
        output = {
                  'REPORT TIMES': all_report_times,
                  'STATES': unrst_data,
                  'WELLS AND CONNECTIONS': wells_conns_type_status_dict,
                  'WELLS SOLUTION SUMMARY': wells_solution_summary_data
                  }
        return output


if __name__ == "__main__" :
    ech_bin_parser = EchBinaryOutputParser()
    ech_bin_parser._test_parser()


