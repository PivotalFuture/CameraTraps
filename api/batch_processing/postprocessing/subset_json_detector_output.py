#
# subset_json_detector_output.py
#
# Pulls a subset of a detector API output file (.json) where filenames match 
# a specified query (prefix), optionally replacing that prefix with a replacement token.  
# If the query is blank, can also be used to prepend content to all filenames.
#
# 1) Retrieve all elements where filenames contain a specified query string, 
#    optionally replacing that query with a replacement token. If the query is blank, 
#    can also be used to prepend content to all filenames.
#
# 2) Create separate .jsons for each unique path, optionally making the filenames 
#    in those .json's relative paths.  In this case, you specify an output directory, 
#    rather than an output path.  All images in the folder blah\foo\bar will end up 
#    in a .json file called blah_foo_bar.json.
#
###
#
# Sample invocations (splitting into multiple json's):
#
# Read from "1800_idfg_statewide_wolf_detections_w_classifications.json", split up into 
# individual .jsons in 'd:\temp\idfg\output', making filenames relative to their individual
# folders:
#
# python subset_json_detector_output.py "d:\temp\idfg\1800_idfg_statewide_wolf_detections_w_classifications.json" "d:\temp\idfg\output" --split_folders --make_folder_relative
#
# Now do the same thing, but instead of writing .json's to d:\temp\idfg\output, write them to *subfolders*
# corresponding to the subfolders for each .json file.
#
# python subset_json_detector_output.py "d:\temp\idfg\1800_detections_S2.json" "d:\temp\idfg\output_to_folders" --split_folders --make_folder_relative --copy_jsons_to_folders
#
###
#
# Sample invocations (creating a single subset matching a query):
#
# Read from "1800_detections.json", write to "1800_detections_2017.json"
#
# Include only images matching "2017", and change "2017" to "blah"
#
# python subset_json_detector_output.py "d:\temp\1800_detections.json" "d:\temp\1800_detections_2017_blah.json" --query 2017 --replacement blah
#
# Include all images, prepend with "prefix/"
#
# python subset_json_detector_output.py "d:\temp\1800_detections.json" "d:\temp\1800_detections_prefix.json" --replacement "prefix/"
#

#%% Constants and imports

import copy
import json
import os

from tqdm import tqdm

from data_management.annotations import annotation_constants


#%% Helper classes

class SubsetJsonDetectorOutputOptions:
    
    replacement = None
    query = ''
    
    
#%% Main function
                
def subset_json_detector_output(input_filename,output_filename,options):

    if options is None:    
        options = SubsetJsonDetectorOutputOptions()
            
    print('Reading json...',end='')
    with open(input_filename) as f:
        data = json.load(f)
    print(' ...done')
    
    images_in = data['images']
    
    images_out = []
    
    print('Searching json...',end='')
    
    # iImage = 0; im = images_in[0]
    for iImage,im in tqdm(enumerate(images_in),total=len(images_in)):
        
        fn = im['file']
        
        # Only take images that match the query
        if not ((len(options.query) == 0) or (fn.startswith(options.query))):
            continue
        
        if options.replacement is not None:
            if len(options.query) > 0:
                fn = fn.replace(options.query,options.replacement)
            else:
                fn = options.replacement + fn
            
        im['file'] = fn
        
        images_out.append(im)
        
    # ...for each image        
    
    print(' ...done')
    
    data['images'] = images_out
    
    print('Serializing back to .json...', end = '')    
    s = json.dumps(data, indent=1)
    print(' ...done')
    print('Writing output file...', end = '')    
    with open(output_filename, "w") as f:
        f.write(s)
    print(' ...done')

    return data

    print('Done, found {} matches (of {})'.format(len(data['images']),len(images_in)))
    
    
#%% Interactive driver
                
if False:

    #%%   
    
    input_filename = r"D:\temp\1800_detections.json"
    output_filename = r"D:\temp\1800_detections_2017.json"
     
    options = SubsetJsonDetectorOutputOptions()
    options.replacement = 'blah'
    options.query = '2017'
        
    data = subset_json_detector_output(input_filename,output_filename,options)
    print('Done, found {} matches'.format(len(data['images'])))

    
#%% Command-line driver

import argparse
import inspect
import sys

# Copy all fields from a Namespace (i.e., the output from parse_args) to an object.  
#
# Skips fields starting with _.  Does not check field existence in the target object.
def argsToObject(args, obj):
    
    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v);

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input .json filename')
    parser.add_argument('output_file', type=str, help='Output .json filename')
    parser.add_argument('--query', type=str, default='', help='Prefix to search for (omitting this matches all)')
    parser.add_argument('--replacement', type=str, default=None, help='Replace [query] with this')
    
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()    
    
    # Convert to an options object
    options = SubsetJsonDetectorOutputOptions()
    argsToObject(args,options)
    
    subset_json_detector_output(args.input_file,args.output_file,options)
    
if __name__ == '__main__':
    
    main()
