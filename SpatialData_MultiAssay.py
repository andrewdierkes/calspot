#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pathlib as path
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# In[29]:


_path = path.Path.cwd()

assay_name_dataset = []
dataset_dict = {}
spatial_list = []

for _filepath in _path.iterdir():
    if _filepath.suffix != r'.log':
        continue
    elif _filepath.suffix == r'.log':
        
        with open(_filepath, 'r') as file:
            
            #individual assay data across 4 channels
            assay_dict = {}
            wl_index = []
            
            data = file.read()
            
            assay_name_finder = re.compile(r'@DES:.*')
            pixelData_finder = re.compile(r'pixelData.*]')
            numberPixel_finder = re.compile(r'numberPixels=[0-9][0-9]')           
            beginPosition_finder = re.compile(r'beginPosition.*[0-9]')
            numberOfSteps_finder = re.compile(r'numberOfSteps.*[0-9]')
            stepSize_finder = re.compile(r'stepSize.*[0-9]')
            spd_finder = re.compile(r'COMMAND.*Sent:.*INS.*SPD.*')
            windowStartWavelength_finder = re.compile(r'windowStartWavelength.*[0-9] ')
            windowEndWavelength_finder = re.compile(r'Sent INS SSCPARAM.*[0-9]')
                                                      
            
            assay_name = re.findall(assay_name_finder, data)
            pixelData = re.findall(pixelData_finder, data)
            numberPixel = re.findall(numberPixel_finder, data)
            beginPosition = re.findall(beginPosition_finder, data)
            numberOfSteps = re.findall(numberOfSteps_finder, data)
            stepSize = re.findall(stepSize_finder, data)
            windowStartWavelength = re.findall(windowStartWavelength_finder, data)
            windowEndWavelength = re.findall(windowEndWavelength_finder, data)
            spd = re.findall(spd_finder, data)

            #number of scans you hope to take
            #if len(pixelData) == 80:
            #scrub & transform data into float
            if len(pixelData) == 80:    
                pixel_data_remove = re.compile(r'pixelData:.\[')
                pixel_data_removal = re.compile(r'\]')

                number_pixel_removal = re.compile(r'numberPixels=')
                begin_position_removal = re.compile(r'beginPosition.*=..')
                number_of_steps_removal = re.compile(r'numberOfSteps.*=..')
                step_size_removal = re.compile(r'stepSize.*=..')
                window_start_wl_removal = re.compile(r'windowStartWavelength=')
                spd_removal = re.compile('COMMAND\tSent: INS ')
                assay_name_remove = re.compile(r'@DES:.')

                    #print(pixelData)
                pixel_data_semi = [pixel_data_remove.sub('', string) for string in pixelData]
                pixel_data = [var.split(', ') for var in [pixel_data_removal.sub('', string) for string in pixel_data_semi]]

                pixel_dataset = []

                for scan in pixel_data:
                    scan_float = []
                    for var in scan:
                        try:
                            scan_float.append(float(var))
                        except ValueError as e:
                                f'Cannot convert {var} to float'
                    pixel_dataset.append(scan_float)

                    #converting into floats and strings for parsing 

                number_pixel_list = [int(var) for var in [number_pixel_removal.sub('', string) for string in numberPixel]]
                begin_position = [int(var) for var in [begin_position_removal.sub('', string) for string in beginPosition]]
                number_of_steps_list = [int(var) for var in [number_of_steps_removal.sub('', string) for string in numberOfSteps]]
                step_size_list = [int(var) for var in [step_size_removal.sub('', string) for string in stepSize]]
                window_start_wl_list = [float(var) for var in [window_start_wl_removal.sub('', string) for string in windowStartWavelength]]
                SPD = [spd_removal.sub('', string) for string in spd]
                window_end_wl_string = ''.join(windowEndWavelength)
                window_end_wl = int(window_end_wl_string[40:43])
                assay_list = [assay_name_remove.sub('', string) for string in assay_name]
 
                number_pixel = int(number_pixel_list[0])
                number_of_steps = int(number_of_steps_list[0])
                step_size = int(step_size_list[0])
                window_start_wl = int(window_start_wl_list[0])
                assay = assay_list[0]

                assay_name_dataset.append(assay)

                    #print(number_pixel, begin_position, number_of_steps, step_size, window_start_wl)

                    #using generator; divide scans into groups of 20 for each channel
                def chunks(lst, chunk):
                    for i in range(0, len(lst), chunk):
                        yield lst[i:i+chunk]

                    #unpack with generator into a list with *
                pixel_channels = [*chunks(pixel_dataset, 20)]
                pixel_ch0 = pixel_channels[0]
                pixel_ch1 = pixel_channels[1]
                pixel_ch2 = pixel_channels[2]
                pixel_ch3 = pixel_channels[3]


                    #create spatial read values from data within assay
                def channel_chunk(ch_pos, step_size, num_steps):
                    '''iterate thru spatial coords using ch_pos we start with, step size and num_steps from assay.. range(start,stop,interval)'''
                    spatial_lister = []
                    for i in range(ch_pos, ch_pos+(step_size*num_steps), step_size):
                                    spatial_lister.append(i)
                    return spatial_lister

                    #spatial locations of assay used for lambda plotter...
                spatial_list.append(channel_chunk(int(begin_position[0]), step_size, number_of_steps))
                spatial_list.append(channel_chunk(int(begin_position[1]), step_size, number_of_steps))
                spatial_list.append(channel_chunk(int(begin_position[2]), step_size, number_of_steps))
                spatial_list.append(channel_chunk(int(begin_position[3]), step_size, number_of_steps))

                    #create index using starting and ending wl and number of pixels in each scan array; adding outside of loop
                wl_step = ((window_end_wl-window_start_wl)/number_pixel)
                start_wl = (window_start_wl-wl_step)
                end_wl = int(window_end_wl_string[40:43])-wl_step

                offset = start_wl 

                while offset < end_wl:

                    i = offset + wl_step
                    wl_index.append(i)
                    offset += wl_step

                    #add spectral and spatial data to outside of log reading script
                assay_dict[f'{assay} ch0'] = dict(zip(spatial_list[0], pixel_ch0))
                assay_dict[f'{assay} ch1'] = dict(zip(spatial_list[1], pixel_ch1))
                assay_dict[f'{assay} ch2'] = dict(zip(spatial_list[2], pixel_ch2))
                assay_dict[f'{assay} ch3'] = dict(zip(spatial_list[3], pixel_ch3))

                dataset_dict[f'{assay}'] = assay_dict

                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)                
                
            else:
                print(f'number of scans if off from 80, check file {file}')


# In[30]:


data_dict = {}

def lambda_plotter(df, wl_index, channel_sp, graph_channel):
    ch_max = [df.max(axis=1)]

    ch_spatial = channel_sp
    
    #iterate thru max list 
    for var in ch_max:
        lambda_max = (max(var))

    lmax_index = []

    #iterate thru df columns looking for the lamda max value
    for col in df.columns:
        index_lookup = df[df[col] == lambda_max].index.tolist()
        try:
        #if a variable exists... appended
            lmax_index.append(index_lookup[0])
        except:
            pass

    var = float(lmax_index[0])
    
    lambda_max_row = wl_index.index(var)

    #take the spatial data across the max wavelength (df.iloc) and assign variables as arrays for find_peaks
    x = np.array(channel_sp)
    y = np.array(df.iloc[lambda_max_row])

    #scipy peak finder; use 1d np.array and height = min value for peak 
    peaks = find_peaks(y, height = 15000, distance=10)
    peak_pos = x[peaks[0]]
    height = peaks[1]['peak_heights']
    
    #plot spatial vs RFU at max wl & maxima via scipy
    fig, ax = plt.subplots()
    plt.scatter(ch_spatial, y)
    plt.scatter(peak_pos, height, color = 'r', s = 30, marker = 'D', label = 'maxima')
    

    ax.set(xlabel='Spatial Position', ylabel='RFU', title= f'Spot {graph_channel} Spatial RFU & Maxima at {round(var, 3)}')
    ax.legend()
    ax.grid()
    
    data_dict['Spot Location'] = graph_channel
    data_dict['Max RFU'] = height
    data_dict['Lambda Max'] = var
    data_dict['Spatial Position for Max RFU'] = peak_pos
    
    return f'Max RFU output {lambda_max}, seen at wavelength: {var}'



# In[31]:


def heat_map(dataframe):
    hm = sns.heatmap(dataframe, cmap='YlOrRd')


# In[32]:


def delist(args):
    delist = [var for small_list in args for var in small_list]
    return(delist) 

#generate a list of the assay names pairing number of repeats with number of channels
assay_single = []
for assay in dataset_dict.keys():
    assay_single.append(np.repeat(assay, 4))
    
assay_list = delist(assay_single)

#for lambda plotter title
channel = [1400,2200,3000,3800]
channel_assay = channel*(len(dataset_dict.keys()))


#create an iterator to go thru assay_list and pair the assay_name with the four corresponding dataframes
assay_chunk = 1
offset = 0

#list_of_data = []

#iterate thru dataset_dict getting into specific channels to create dataframes & link assay_name with DF
for assay in dataset_dict.values():
    for channel in assay.values():
        
        df = pd.DataFrame(channel, index=wl_index)
        df.index.name = assay_list[offset]
        
        
        #list_of_data.append(df)
        
        display(df)
        display(lambda_plotter(df, wl_index, spatial_list[offset], channel_assay[offset]))
        
        offset += assay_chunk
        
        #display(heat_map(df))
        
        #display(hm = sns.heatmap(df, cmap='YlOrRd'))
        

        


# In[60]:


for var in list_of_data:
    with pd.ExcelWriter(f'test.xlsx') as writer:
            var.to_excel(writer, sheet_name='test')


# In[85]:


display(df_ch_0)
hm_ch_0 = sns.heatmap(df_ch_0, cmap='YlOrRd')
lambda_plotter(df_ch_0, wl_index, ch_0_spatial, 1400)

#summary of data
df_ch_summary_0 = pd.DataFrame(data_dict)
display(df_ch_summary_0)


# In[51]:


for var in list_of_data:
    display(hm = sns.heatmap(var, cmap='YlOrRd'))


# In[64]:


display(df_ch_0)
hm_ch_0 = sns.heatmap(df_ch_0, cmap='YlOrRd')
display(lamda_plotter(df_ch_0, wl_index, ch_0_spatial, 0))


# In[28]:


display(df_ch_1)
display(hm_ch_1 = sns.heatmap(df_ch_1, cmap='YlOrRd'))
display(lamda_plotter(df_ch_1, wl_index, ch_1_spatial, 1))


# In[29]:


display(df_ch_2)
hm_ch_3 = sns.heatmap(df_ch_2, cmap='YlOrRd')
display(lamda_plotter(df_ch_2, wl_index, ch_2_spatial, 2))


# In[30]:


display(df_ch_3)
hm_ch_3 = sns.heatmap(df_ch_3, cmap='YlOrRd')
display(lamda_plotter(df_ch_3, wl_index, ch_3_spatial, 3))


# In[366]:


with pd.ExcelWriter('spatial_calstrip.xlsx') as writer:
    df_ch_0.to_excel(writer, sheet_name='channel 1400')
    df_ch_1.to_excel(writer, sheet_name='channel 2200')
    df_ch_2.to_excel(writer, sheet_name='channel 3000')
    df_ch_3.to_excel(writer, sheet_name='channel 3800')


# In[ ]:


##NEED MORE SCANS 


# In[76]:


#find max value  in df and label it 
#lamda_max, lmax_index or var (all mean the same thing)

#find max df values in each row
def lamda_plotter(df, wl_index,
ch_2_max = [df_ch_2.max(axis=1)]
print(len(ch_2_max))
#print(ch_2_max)
#iterate thru max list 
for var in ch_2_max:
    lamda_max = (max(var))
    print(lamda_max)

lmax_index = []

#iterate thru df columns looking for the lamda max value
for col in df_ch_2.columns:
    index_lookup = df_ch_2[df_ch_2[col] == lamda_max].index.tolist()
    try:
        #if a variable exists... appended
        lmax_index.append(index_lookup[0])
    except:
        pass

var = float(lmax_index[0])
    
                  ########
lamda_max_row = wl_index.index(var)
#print(lamda_max_row)

lm_spectra = [(df_ch_2.iloc[lamda_max_row])]
#print(lm_spectra)
#print(ch_0_spatial)
fig, ax = plt.subplots()

plt.scatter(ch_0_spatial, lm_spectra)
#print(len(wl_index))
ax.grid()
ax.set(xlabel='Spatial Position', ylabel='RFU', title= f'{ Spatial RFU at {var}')


# In[ ]:




