# Bioacoustic Analysis of Bird Calls

The initially existing files in the repo are just to illustrate how I'd like the code organized. 

The project root will contain setup.py/setup.cfg. I also often put required data, such as soundfiles into a subdirectory of the project root. Python packages go into the src subdirectory directory. The modules go into the packages. Just as an example of the structure:

```
<proj-root>
     src
        utilities                    <----- package
            __init__.py
            file_utils.py            <----- module
            audio_utils.py           <----- module
                ...
        ml                           <----- package
            __init__.py
            embedding.py
            clustering.py
 ```          
            
