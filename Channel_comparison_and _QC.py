#!/usr/bin/env python
# coding: utf-8

# In[26]:


pip install pingouin


# In[140]:


import re
import numpy as np
import pathlib as path
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats
from datetime import date
import pingouin as pg
from itertools import combinations


# In[141]:


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
            
            #find specific names of assay
            #print(assay_name, file)
            
            
            #number of scans you hope to take
            #scrub & transform data into float
            #if len(pixelData) == 80:    
            pixel_data_remove = re.compile(r'pixelData:.\[')
            pixel_data_removal = re.compile(r'\]')

            number_pixel_removal = re.compile(r'numberPixels=')
            begin_position_removal = re.compile(r'beginPosition.*=..')
            number_of_steps_removal = re.compile(r'numberOfSteps.*=..')
            step_size_removal = re.compile(r'stepSize.*=..')
            window_start_wl_removal = re.compile(r'windowStartWavelength=')
            spd_removal = re.compile('COMMAND\tSent: INS ')
            assay_name_remove = re.compile(r'@DES:.')


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
            #print(number_pixel_list, file)
            begin_position = [int(var) for var in [begin_position_removal.sub('', string) for string in beginPosition]]
            number_of_steps_list = [int(var) for var in [number_of_steps_removal.sub('', string) for string in numberOfSteps]]
            step_size_list = [int(var) for var in [step_size_removal.sub('', string) for string in stepSize]]
            window_start_wl_list = [float(var) for var in [window_start_wl_removal.sub('', string) for string in windowStartWavelength]]
            SPD = [spd_removal.sub('', string) for string in spd]
            window_end_wl_string = ''.join(windowEndWavelength)
            #ending WL edited as changing IT will effect the result (Old code: int(window_end_wl_string[40:43])
            window_end_wl = 650
            assay_list = [assay_name_remove.sub('', string) for string in assay_name]
                
            number_pixel = int(number_pixel_list[0])
            
            
            
            number_of_steps = int(number_of_steps_list[0])
            step_size = int(step_size_list[0])
            window_start_wl = int(window_start_wl_list[0])
            assay = assay_list[0]
            
            assay_name_dataset.append(assay)

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
            #ending WL edited as changing IT will effect the result (Old code=int(window_end_wl_string[40:43])-wl_step)
            end_wl = 650-wl_step

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
            

            #else:
             #   print(f'number of scans if off from 80, check file {file}')


# In[142]:


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
    
    #avg function
    def average(data, length):
        avg = sum(data)/len(data)
        return avg
    
    #create an average list similar to plateau if it is within 10% of the RFU
    RFU_max_list = []
    for z in df.iloc[lambda_max_row]:
        if z >= lambda_max*.92:
            RFU_max_list.append(z)
    
    num_spatial_pos = len(RFU_max_list)
    
    average = average(RFU_max_list, num_spatial_pos)

    
    #take the spatial data across the max wavelength (df.iloc) and assign variables as arrays for find_peaks
    x = np.array(channel_sp)
    y = np.array(df.iloc[lambda_max_row])

    #scipy peak finder; use 1d np.array and height = min value for peak 
    peaks = find_peaks(y, height = 2000, distance=10)
    peak_pos = x[peaks[0]]
    height = peaks[1]['peak_heights']
    
    
    
    #plot spatial vs RFU at max wl & maxima via scipy
    #fig, ax = plt.subplots()
    #plt.scatter(ch_spatial, y)
    #plt.scatter(peak_pos, height, color = 'r', s = 30, marker = 'D', label = 'maxima')
    

    #ax.set(xlabel='Spatial Position', ylabel='RFU', title= f'Formulation {graph_channel} Spatial RFU & Lambda Max at {round(var, 3)}')
    #ax.legend()
    #ax.grid()
    
    data_dict['Assay Name'] = graph_channel
    data_dict['Max RFU'] = height
    data_dict['Average +/- 8%'] = average 
    data_dict['Spatial Positions Taken for Average'] = num_spatial_pos 
    data_dict['Lambda Max'] = var
    data_dict['Spatial Position for Max RFU'] = peak_pos
    
    #return f'Max RFU output {lambda_max}, seen at wavelength: {var}'


# In[143]:


def heat_map(dataframe):
    hm = sns.heatmap(dataframe, cmap='YlOrRd')


# In[144]:


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


summary_list = []

#iterate thru dataset_dict getting into specific channels to create dataframes & link assay_name with DF
for assay in dataset_dict.values():
    for channel in assay.values():
        
        df = pd.DataFrame(channel, index=wl_index)
        df.index.name = assay_list[offset]
        
        
        #list_of_data.append(df)
        
        #display(df)
        lambda_plotter(df, wl_index, spatial_list[offset], assay_list[offset])
        df_summary = pd.DataFrame(data_dict)
        summary_list.append(df_summary)
        
        offset += assay_chunk
        
        #display(heat_map(df))
        
        #display(hm = sns.heatmap(df, cmap='YlOrRd'))
        

        


# In[145]:


#### generate dataframe of individual channels 
df_dataset_sum = pd.concat(summary_list).sort_index().reset_index(drop=True)
df_dataset_sum.sort_values(by=['Assay Name', 'Spatial Position for Max RFU'], inplace=True)
#display(df_dataset_sum)


# In[146]:


def chunk_cv(df_col, chunk=4):
    '''This function will iterate over a df_col and return the average and stdev, using chunk_iterator.. so if you'd like the average & stdev values occuring every 3 rows... use 3 as your chunk iterator'''

    offset_mean = 0
    offset_stdev = 0
    
    number_list = [var for var in range(len(df_col))]
    
    dataset_average = []
    dataset_stdev = []
    
    
    while offset_mean < len(number_list):
        i_mean = number_list[offset_mean:chunk+offset_mean]
        average = df_col.iloc[i_mean].mean(axis=0)
        
        dataset_average.append(average)
        #dataset_array.append(_array)
        
        offset_mean += chunk
    
    while offset_stdev < len(number_list):
        i_stdev = number_list[offset_stdev:chunk+offset_stdev]
        stdev = df_col.iloc[i_stdev].std(ddof=1)
        
        dataset_stdev.append(stdev)
        
        offset_stdev += chunk
    
    return dataset_average, dataset_stdev


# In[147]:


def unique(df_col, chunk=4):
    '''This function will iterate over a df_col and return only unique values using chunk_iterator.. so if you'd like the unique value of something occuring every 3 rows... use 3 as your chunk iterator'''

    offset = 0
    
    number_list = [var for var in range(len(df_col))]
    
    dataset_array = []
    dataset_list = []
    
    while offset < len(number_list):
        i = number_list[offset:chunk+offset]
        _array = df_col.iloc[i].unique()
        
        dataset_array.append(_array)
        
        offset += chunk
    
    for var in dataset_array:
        dataset_list.append(var.tolist())
    
    unique = delist(dataset_list)
    
    return unique


# In[148]:


#find unique names from Assay Name row, slice each name to get rid of the rep number and then pass through unique function
#name_len = length of names we want to slice

def slice_name(df, row, name_len, end_offset):
    '''Function SLICE_NAME slices strings (usually assay name) that are similar in nature (df row). It works to 
    remove the ends (end_offset) which are different (usually rep number) to allow the UNIQUE function to work
    df = dataframe to use
    row = the index of the column you want to slice
    name_len = the length of each assay name, they all should be the same
    end_offset = what position you want to end with'''
    
    assay__name = [var for var in df.iloc[:,row]]

    unique_name = []


    for var in assay__name:
        if len(var) == name_len:
            unique_name.append(var[0:end_offset])

        else:
            print(file)
            print(f'labeling error for assay:{var}, length {len(var)}')
            for count, var2 in enumerate(var):
                print(count, var2)
    
    return pd.DataFrame(unique_name)


# In[210]:


#generate intrastrip data, avg, stdev & cv using previous dataframe with all channel values
strip = unique(df_dataset_sum['Assay Name'])
strip_avg, strip_stdev = chunk_cv(df_dataset_sum['Max RFU'], 4)
strip_avg_avg, strip_avg_stdev = chunk_cv(df_dataset_sum['Average +/- 8%'], 4)

#generate dictionary for pd.df
strip_dict = {}
strip_dict['Strip Name'] = strip
strip_dict['Intra Strip Average RFU'] = strip_avg
strip_dict['Intra Strip StDev'] = strip_stdev
strip_dict['Intra Strip Average +/- 8% Average'] = strip_avg_avg
strip_dict['Intra Strip Average +/- 8% StDev'] = strip_avg_stdev



# In[211]:


#df for intrastrip avg & stdev, inserting CVs
df_strip = pd.DataFrame(strip_dict)
df_strip.insert(3, 'Intra Strip CV, n=5', round(((df_strip['Intra Strip StDev']/df_strip['Intra Strip Average RFU'])*100),4))
df_strip.insert(6, 'Intra Strip Average +/- 8% CV, n=5', round(((df_strip['Intra Strip Average +/- 8% StDev']/df_strip['Intra Strip Average +/- 8% Average'])*100),4))
display(df_strip)


# Data Below is for Strip Repeats of Individual Channels

# In[212]:


def channel_analysis(channel, df, position_index_start,position_iterator=4):
    '''channel=target channel,
    df = dataframe to iterate over(ie df_dataset_sum), 
    position_index_start = index of 1400 (0),2200(1),3000(2),3800(3) to start at
    position_iterator = 4 by default, so every 4 points'''
    
    #setup list and dataframe for all iterations of specified channel
    ch_index = [var for var in range(position_index_start, len(df), position_iterator)]
    
    ch_df = []
    for var in ch_index:
        data = df.iloc[var]
        ch_df.append(data)
    
    df_channel = pd.DataFrame(ch_df) #RETURN
    #display(df_channel)
    
    #set up strip repeat dictionary and dataframe
    df_stripname = slice_name(df_channel, 0,16,13)
    strip_channel = unique(df_stripname.iloc[:,0],3)
    
    strip_ch_sr_avg, strip_ch_sr_stdev = chunk_cv(df_channel['Max RFU'], 3)

    strip_channel_dict = {}
    strip_channel_dict['Strip Repeat Name'] = strip_channel
    strip_channel_dict[f'CH{channel} Repeat Average'] = strip_ch_sr_avg
    strip_channel_dict[f'CH{channel} Repeat StDev'] = strip_ch_sr_stdev
    
    df_strip_repeat = pd.DataFrame(strip_channel_dict)
    df_strip_repeat[f'CH{channel} Strip Repeat CV'] = round((df_strip_repeat.iloc[:,2]/df_strip_repeat.iloc[:,1])*100,3) #RETURN
    #display(df_strip_repeat)
    
    #begin to analyze by bag
    ch_bag_name = slice_name(df_strip_repeat,0,13,10)
    ch_bag_unique = unique(ch_bag_name.iloc[:,0],11)
    
    #data analysis grouping all assays by bag (not by strip repeat)
    ch_bag_avg, ch_bag_stdev = chunk_cv(df_channel['Max RFU'], 33)
    
    ch_bag_dict = {}
    ch_bag_dict['Bag'] = ch_bag_unique
    ch_bag_dict[f'CH{channel} Average for Max RFU, n={len(df_strip_repeat)}'] = ch_bag_avg
    ch_bag_dict[f'CH{channel} StDev for Max RFU, n={len(df_strip_repeat)}'] = ch_bag_stdev
    
    df_ch_bag = pd.DataFrame(ch_bag_dict)
    df_ch_bag[f'CH{channel} Bag CV'] = round((df_ch_bag.iloc[:,2]/df_ch_bag.iloc[:,1])*100,3)
    #display(df_ch_bag) #RETURN
    
    #data analysis group with bag strip repeats
    ch_sr_avg, ch_sr_stdev = chunk_cv(df_strip_repeat[f'CH{channel} Strip Repeat CV'], 11)
    
    ch_bag_sr_dict = {}
    ch_bag_sr_dict['Bag'] = ch_bag_unique
    ch_bag_sr_dict[f'CH{channel} Avg Bag CV for Strip Repeat'] = ch_sr_avg
    
    df_ch_sr_bag = pd.DataFrame(ch_bag_sr_dict) #RETURN
    #display(df_ch_sr_bag)
    
    return df_channel, df_strip_repeat, df_ch_bag, df_ch_sr_bag
    


# In[213]:


df_1400channel, df_1400strip_repeat, df_1400bag, df_1400sr_bag = channel_analysis(1400, df_dataset_sum, 0)
df_2200channel, df_2200strip_repeat, df_2200bag, df_2200sr_bag = channel_analysis(2200, df_dataset_sum, 1)
df_3000channel, df_3000strip_repeat, df_3000bag, df_3000sr_bag = channel_analysis(300, df_dataset_sum, 2)
df_3800channel, df_3800strip_repeat, df_3800bag, df_3800sr_bag = channel_analysis(3800, df_dataset_sum, 3)



# In[214]:


df_master_channel = pd.concat([df_1400channel.reset_index(drop=True), df_2200channel.reset_index(drop=True), df_3000channel.reset_index(drop=True), df_3800channel.reset_index(drop=True)], axis=1)
df_master_repeat = pd.concat([df_1400strip_repeat.reset_index(drop=True), df_2200strip_repeat.reset_index(drop=True), df_3000strip_repeat.reset_index(drop=True), df_3800strip_repeat.reset_index(drop=True)], axis=1)
df_master_bag = pd.concat([df_1400bag.reset_index(drop=True), df_2200bag.reset_index(drop=True), df_3000bag.reset_index(drop=True), df_3800bag.reset_index(drop=True)], axis=1)
df_master_sr_bag = pd.concat([df_1400sr_bag.reset_index(drop=True), df_2200sr_bag.reset_index(drop=True), df_3000sr_bag.reset_index(drop=True), df_3800sr_bag.reset_index(drop=True)], axis=1)
display(df_master_channel,df_master_repeat,df_master_bag,df_master_sr_bag)


# In[215]:


#take avg CV for each strip

unique_strip_slicer = slice_name(df_strip, 0, 16, -3)

unique_strip = unique(unique_strip_slicer.iloc[:,0], 3)

df_strip_cv_avg, df_strip_cv_stdev = chunk_cv(df_strip.iloc[:,3], 3)

df_strip_cv_data = {}
#df_strip_cv_data['Strip Name'] = unique_strip
df_strip_cv_data['Average Intrastrip CV, n=3'] = df_strip_cv_avg


df_intrastrip_avg = pd.DataFrame(df_strip_cv_data)


# In[216]:


#should equal number of assays ran
print(len(df_strip['Strip Name']))


# In[217]:


#generate pd.df for repeat averages for strips ran

#pass previous df through functions to generate mean & stdev
repeat_strip_avg, repeat_strip_stdev = chunk_cv(df_strip['Intra Strip Average RFU'], 3)
repeat_strip_avg_avg, repeat_strip_avg_stdev = chunk_cv(df_strip['Intra Strip Average +/- 8% Average'], 3)

#generate dictionary for pd.df
repeat_strip_dict = {}
repeat_strip_dict['Strip Name'] = unique_strip
repeat_strip_dict['Strip Repeat Average RFU'] = repeat_strip_avg
repeat_strip_dict['Strip Repeat StDev'] = repeat_strip_stdev
repeat_strip_dict['Strip Repeat Average +/- 8% Average'] = repeat_strip_avg_avg
repeat_strip_dict['Strip Repeat Average +/- 8% StDev'] = repeat_strip_avg_stdev

#create pd.df & insert cv values
df_repeat_strip = pd.DataFrame(repeat_strip_dict)
df_repeat_strip.insert(3, 'Strip Repeat CV', round(((df_repeat_strip['Strip Repeat StDev']/df_repeat_strip['Strip Repeat Average RFU'])*100),4))
df_repeat_strip.insert(6, 'Strip Repeat Average +/- 8% CV', round(((df_repeat_strip['Strip Repeat Average +/- 8% StDev']/df_repeat_strip['Strip Repeat Average +/- 8% Average'])*100),4))
display(df_repeat_strip)


# In[218]:


#comparing channels to average for strip repeat

avg_for_sr = df_repeat_strip.iloc[:,3].mean()


# In[219]:


#ANALYSIS OF BEST STRIPS USED TO SHIP OUT!

#add bag column using slice function
df_repeat_strip['Bag'] = slice_name(df_repeat_strip,0,13,10)

#add avg intrastrip cv 
df_cv_avg = pd.concat([df_repeat_strip.reset_index(drop=True), df_intrastrip_avg.reset_index(drop=True)], axis=1)

#drop cols
#df_repeat_strip.drop(columns=['Average Intrastrip CV, n=3'], axis=1, inplace=True)
df_cv_avg.drop(columns=['Strip Repeat StDev', 'Strip Repeat Average +/- 8% Average','Strip Repeat Average +/- 8% StDev','Strip Repeat Average +/- 8% CV'], inplace=True)

low_cv_repeat = []

#find strip repeat with CV lower than 0.2 and add the index to low_cv_repeat
for var in df_cv_avg.iloc[:,2]:
    if var < 0.2:
        indexer = df_cv_avg.index[df_cv_avg['Strip Repeat CV'] == var].tolist()
        low_cv_repeat.append(indexer)

low_cv_repeat = delist(low_cv_repeat)

#using index from low_cv_repeat, pair it with intrastrip values <2
intra_strip_match_list = []
for var in low_cv_repeat:
    
    intra_strip_match = df_cv_avg.iloc[var,4]
    intra_strip_match_list.append(intra_strip_match)

#using strips with strip repeat < 0.2, find ones with intrastrip cvs less than 2.6
index_strips_to_ship = []
for var in intra_strip_match_list:
    if var < 2.6:
        indexer = df_cv_avg.index[df_cv_avg['Average Intrastrip CV, n=3'] == var].tolist()
        index_strips_to_ship.append(indexer)

index_strips_to_ship = delist(index_strips_to_ship)

#find name of strips with low CV
strips_to_ship = []
for var in index_strips_to_ship:
    strip_details = df_cv_avg.iloc[var]
    strips_to_ship.append(strip_details)
    
df_ship= pd.DataFrame(strips_to_ship)
#display(df)

#final QC on RFU output +/- 2% from Mean of low CV strips
avg_rfu = df_ship.iloc[:,1].mean()

final_shipment_index = []

for var in df_ship.iloc[:,1]:
    if avg_rfu*.98 <= var <= avg_rfu*1.18:
        indexer = df_ship.index[df_ship['Strip Repeat Average RFU'] == var].tolist()
        final_shipment_index.append(indexer)

final_shipment_index = delist(final_shipment_index)

final_shipment = []

for var in final_shipment_index:
    final = df_cv_avg.iloc[var]
    final_shipment.append(final)


df_final = pd.DataFrame(final_shipment)
display(df_final)

avg= df_final.iloc[:,1].mean()
stdev= df_final.iloc[:,1].std()
cv = (stdev/avg)*100
print(cv)
    
date = date.today()
with pd.ExcelWriter(f'shippment_{date}.xlsx') as writer:
    df_final.to_excel(writer, sheet_name='Shippment 0.8%CB Bag 1-3')
#df_cv_avg.sort_values(by=['Strip Repeat CV', 'Average Intrastrip CV, n=3'])


# In[233]:


#using SLICE_NAME and UNIQUE we can create bag labeling column

bag_name = slice_name(df_repeat_strip, 0, 13, 10)
unique_bag = unique(bag_name.iloc[:,0], 33)
print(unique_bag)
display(df_repeat_strip)


# In[237]:


#generate pd.df for repeat averages for strips ran

#pass previous df through functions to generate mean & stdev
bag_avg, bag_stdev = chunk_cv(df_repeat_strip['Strip Repeat Average RFU'], 11)
bag_avg_avg, bag_avg_stdev = chunk_cv(df_repeat_strip['Strip Repeat Average +/- 8% Average'], 11)

#generate dictionary for pd.df
bag_dict = {}
bag_dict['Bag Name'] = unique_bag
bag_dict['Bag Average RFU'] = bag_avg
bag_dict['Bag StDev'] = bag_stdev
bag_dict['Bag Average +/- 8% Average'] = bag_avg_avg
bag_dict['Bag Average +/- 8% StDev'] = bag_avg_stdev

#create pd.df & insert cv values
df_bag = pd.DataFrame(bag_dict)
df_bag.insert(3, 'Bag CV', round(((df_bag['Bag StDev']/df_bag['Bag Average RFU'])*100),4))
df_bag.insert(6, 'Strip Repeat Average +/- 8% CV', round(((df_bag['Bag Average +/- 8% StDev']/df_bag['Bag Average +/- 8% Average'])*100),4))
display(df_bag)


# In[238]:


date = date.today()
with pd.ExcelWriter(f'{date}.xlsx') as writer:
    df_dataset_sum.to_excel(writer, sheet_name='Channels')
    df_strip.to_excel(writer, sheet_name='Strip')
    df_repeat_strip.to_excel(writer, sheet_name='Strip Repeat')
    df_bag.to_excel(writer, sheet_name='Bag')


# In[ ]:




