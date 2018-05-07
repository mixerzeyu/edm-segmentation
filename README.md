# edm-segmentation

This is my final project for the MUMT 621 Music Information Retrieval at McGill University in Winter 2018

Where is the Drop: Segment Boundary Detection in Electronic Dance Music
Final Project Report for MUMT 621 Music Information Retrieval

Author: Zeyu Li

4 code files are included

annotation_reader.py: the python subroutines that read annotation from txt files in SDIF format, or mp3 tags written by Serato DJ
segmentation.py: the python subroutines that completes tempo and beat tracking, downbeat tracking, boundary detection, and adjust obundaries
main.py: the python main routine for the project that evaluates each step of the segmentation process, and report results to csv files 

sdifreader.omp: the OpenMusic patch that reads SDIF format files to txt files
