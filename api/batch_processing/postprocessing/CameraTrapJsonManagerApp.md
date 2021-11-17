# Batch processing output JSON Manager App




# Downloading the Camera Trap API Output Manager App

Download it <a href="https://lilablobssc.blob.core.windows.net/models/apps/CameraTrapApiOutputManager.1.1.zip">here</a>.  If you don&rsquo;t already know what this app does, it&rsquo;s probably not super-useful to you, but...


# Documentation for the Camera Trap API Output Manager App

Documentation coming soon; for now, <a href="mailto:cameratraps@microsoft.com">email us</a> with questions, or take a look at the corresponding <a href="https://github.com/microsoft/CameraTraps/blob/master/api/batch_processing/postprocessing/subset_json_detector_output.py">Python script</a> to understand what the options do.



Let's say the image file paths in your API output look like this:

```
A/B/C/D/image001.jpg
A/B/C/E/image002.jpg
A/B/C/F/image003.jpg
A/B/C/G/image004.jpg
E/Q/R/S/image005.jpg
E/Q/R/T/image006.jpg
```

#### Top

If you set "Split folder mode" to "Top", that means "split image entries into JSON files based on the <i>top</i>-level folder".  In the above example, you would get two JSON files, one for the folder "A", and one for the folder "E".  Those files would be called "A.json" and "E.json", respectively.


#### Bottom

If you set "Split folder mode" to "Bottom", the app would create JSON files for the bottom-level folders.  In the above example, you would get six JSON files, for the folders:

```
A/B/C/D
A/B/C/E
A/B/C/F
A/B/C/G
E/Q/R/S
E/Q/R/T
````
 
Those files would be called:

```
A_B_C_D.json
A_B_C_E.json
A_B_C_F.json
A_B_C_G.json
E_Q_R_S.json
E_Q_R_T.json
````

This is useful if you have one camera’s data in each bottom-most folder.


#### NFromTop

What if you want each JSON file to represent a subfolder that's not a top-most nor a bottom-most folder?

If all of them are a certain number of levels (`N`) below the top-most folder, you can choose the "NFromTop" option and specify `N` in the "Split parameter" text box. `N` is "how many folders _down from the top_". 

When "NFromTop" is specified, setting `N` to "0" would be the same as using the "Top" splitting mode.  Setting `N` to "1" would split one level down from the top, "2" would split two levels down, etc.  

In the above example, setting `N` to 1 and using "NFromTop" results in two JSON files, for the folders:

```
A/B
E/Q
```

#### NFromBottom

Same idea for "NFromBottom", where `N` is "how many folders _up from the bottom_". 

Here using a value of "0" for `N` would be the same as using the "Bottom" splitting mode (bottom-most folders), "1" would be one level up, etc. So in the above example, using "NFromBottom" and setting `N` to 1 would result in JSON files for these folders:

```
A/B/C
E/Q/R
```


### Once you've filled in the options...

Click `Process` to proceed. The output window will show a progress bar and the paths to the resulting, smaller JSON files. 

The `Help` button will bring you to this page.


## Other notes

You should not need to worry about the fact that the output file uses forward slashes `/`, except in specifying the Query and Replacement string/token. Both the Output Manager and Timelapse will handle the path separator correctly for the platform you're running on.


## Help

If you run into any issues whatsoever, email us at cameratraps@lila.science for help!
